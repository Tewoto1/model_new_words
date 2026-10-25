"""Neologism v0: replace a phrase with one new embedding row, train it, watch where it goes.

Run from the repo root:   python -m experiments.neologism
Colab: %cd to the repo, then  from experiments.neologism import main; main(WORD="syn_huge")

Per run, outputs/neologism/<word>/<tag>/:
  config.json   CFG used
  traj.pt       e_new every LOG_EVERY steps [n_logs, d] + reference rows, for PCA plots
  logs.json     per log step: loss, slot metrics on eval/competitor/nearmiss/control, geometry, equivalence
"""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import load_word, word_meta
from model import Model
from train import TrainConfig, train

CFG = dict(
    MODEL="roneneldan/TinyStories-33M",
    DEVICE="cuda",                       # "mps" / "cpu"
    DTYPE="float32",
    WORD="temporal_the_next_day",        # see data/words/; check syn_huge (calibration) first
    N_TRAIN=200,
    INITS=["mean_emb", "mean_h", "in_span"],   # "gold" is added automatically when the word has one
    LR_NEW=5e-3,
    LR_REST=0.0,                         # 0 = neologism learning; Phase B: 1e-6, 1e-5, 1e-4
    STEPS=300, BS=16, LOG_EVERY=10,
    N_EVAL_SLOTS=200,                    # rows per slot set
    N_EQUIV=64, EQUIV_EVERY=50,          # equivalence = two forwards per row, the slow metric
    K_SPAN=5,
    OUT="outputs/neologism",
    SEED=0,
)


# ---------------------------------------------------------------------------- inits
def make_inits(m: Model, train_rows, new_id: int, active, k_span: int = 5, gold_id=None):
    """Starting points for e_new. Returns (inits: {name: [d] float32}, info: dict).
      mean_emb : mean of the active rows                       (neutral; what HF's mean-resizing does)
      mean_h   : mean slot state h over train rows, rescaled to the mean norm of active rows
                 (tied model: d logit_new / d e_new = h, so this is one big output-side gradient step)
      in_span  : mean_h projected onto span of its k_span nearest active rows, rescaled the same way
                 (the 'new word = generalization of nearby words' hypothesis as a start)
      gold     : E[gold_id] if the word has a one-token equivalent (calibration: should not move)
    info = dict(mu=mean active row, row_norm=mean active-row norm, H_train=[N, d], span_ids=[k_span])
    Only use rows < new_id (the new row is garbage at this point).

    Hints:
        x.norm(dim=1).mean() = mean row norm; v / v.norm() * r rescales.
        Reuse m.slot_states(rows)["h"], m.neighbors(..., active=active, mu=mu, upto=new_id), m.project.
    """
    E = m.E.detach()[:new_id].float()
    R = E[active]
    mu, rn = R.mean(0), R.norm(dim=1).mean()
    H = m.slot_states(train_rows)["h"]
    mh = H.mean(0)
    mh = mh / mh.norm() * rn
    nb, _ = m.neighbors(mh, k=k_span, active=active, mu=mu, upto=new_id)
    sp = m.project(mh, nb)
    inits = {"mean_emb": mu.clone(), "mean_h": mh, "in_span": sp / sp.norm() * rn}
    if gold_id is not None:
        inits["gold"] = E[gold_id].clone()
    return inits, dict(mu=mu, row_norm=rn, H_train=H, span_ids=nb)


# ---------------------------------------------------------------------------- metrics
@torch.no_grad()
def slot_metrics(e_new, H, E_old, k_lose: int = 8) -> dict:
    """At fixed slot states H [N, d], with old rows E_old [V, d] and candidate new row e_new [d]:
    the softmax the model actually computes is over V+1 logits [H @ E_old.T, H @ e_new].
    Returns (means over the N slots):
      p_new      probability of the new word
      logp_new   mean log-probability of the new word
      margin     l_new - max old logit
      top1_frac  fraction of slots where the new word is the argmax
      losers     (ids [k_lose], dp [k_lose]): old tokens whose mean probability dropped most
                 when the new row was added (dp = p_with_new - p_without, most negative first)
    Sanity: if e_new == E_old[j] exactly, p_new == p0_j / (1 + p0_j) at every slot.

    Hints:
        torch.cat([L_old, l_new[:, None]], dim=1): [N, V] + [N, 1] -> [N, V+1] (x[:, None] adds a dim).
        log_softmax then .exp() for probabilities (numerically stable).
        x.topk(k, largest=False) gives the k most negative. Return losers as tensors; main() decodes them.
    """
    L_old = H @ E_old.T
    l_new = H @ e_new
    lp = torch.cat([L_old, l_new[:, None]], 1).log_softmax(-1)
    p0 = L_old.softmax(-1)
    dp = (lp[:, :-1].exp() - p0).mean(0)
    lose = dp.topk(k_lose, largest=False)
    return dict(p_new=lp[:, -1].exp().mean().item(), logp_new=lp[:, -1].mean().item(),
                margin=(l_new - L_old.max(1).values).mean().item(),
                top1_frac=(l_new > L_old.max(1).values).float().mean().item(),
                losers=(lose.indices, lose.values))


@torch.no_grad()
def equivalence(m: Model, rows, new_id: int, n_next: int = 5, bs: int = 32) -> dict:
    """Does {w} behave like the phrase it replaced? For each row, run both versions from
    m.encode_pair (same left, NEW vs phrase tokens, same right) and compare the next-token
    distributions over the first k = min(n_next, n_right) right tokens:
      in ids_new  the logits predicting right[0] sit at position slot
      in ids_orig they sit at position len(ids_orig) - n_right - 1   (the last phrase token)
    Drop the new row's logit (columns >= new_id) and renormalize before comparing.
    Returns {"kl": [N] KL(p_orig || p_new) averaged over the k positions,
             "nll_new": [N], "nll_orig": [N]} (mean NLL of the k right tokens). Rows with n_right=0 skipped.
    Sanity: if E[new_id] == E[gold] and the phrase is the single token gold, KL == 0.

    Hints:
        F.kl_div(input, target, log_target=True) = KL(target || input), with input = log q, target = log p.
        Argument order is the trap: KL(p_orig || p_new) -> input = log p_new, target = log p_orig.
        reduction="none" then .sum(-1) over the vocab, then mean over the k positions.
        logits[..., :new_id] drops the new word's column; log_softmax after that renormalizes.
        x.gather(1, idx[:, None]) picks each target's log-prob: [k, V] with [k, 1] -> [k, 1].
        Pad both versions with m.pad; index each row at its own positions (phrase lengths differ).
    """
    kls, nn_, no_ = [], [], []
    for i in range(0, len(rows), bs):
        pairs = [p for p in (m.encode_pair(r, new_id) for r in rows[i:i + bs]) if p[3] > 0]
        if not pairs:
            continue
        a, aa = m.pad([p[0] for p in pairs])
        b, ba = m.pad([p[1] for p in pairs])
        la = m.forward(a, aa)[..., :new_id].float().log_softmax(-1)
        lb = m.forward(b, ba)[..., :new_id].float().log_softmax(-1)
        for j, (ids_new, ids_orig, slot, nr) in enumerate(pairs):
            k = min(n_next, nr)
            pb = len(ids_orig) - nr - 1
            A, B = la[j, slot:slot + k], lb[j, pb:pb + k]
            tgt = torch.tensor(ids_new[slot + 1:slot + 1 + k], device=A.device)[:, None]
            kls.append(F.kl_div(A, B, log_target=True, reduction="none").sum(-1).mean().item())
            nn_.append(-A.gather(1, tgt).mean().item())
            no_.append(-B.gather(1, tgt).mean().item())
    t = torch.tensor
    return dict(kl=t(kls), nll_new=t(nn_), nll_orig=t(no_))


@torch.no_grad()
def geometry(m: Model, e_new, new_id, active, mu, ref_ids: dict, span_ids=None, k: int = 8) -> dict:
    E = m.E.detach()[:new_id].float()
    v = e_new.float() - mu
    out = dict(norm=e_new.norm().item(),
               neighbors=m.decode_ids(*m.neighbors(e_new, k=k, active=active, mu=mu, upto=new_id)),
               cos={w: F.cosine_similarity(v, E[i] - mu, dim=0).item() for w, i in ref_ids.items()})
    if ref_ids:
        out["span_frac_refs"] = m.span_fraction(e_new, list(ref_ids.values())).item()
    if span_ids is not None:
        out["span_frac_init_nbrs"] = m.span_fraction(e_new, span_ids).item()
    return out


def ref_words(m: Model, meta) -> dict:
    """Single-token reference words: gold + competitors + near-misses -> {word: id}."""
    refs = {}
    for w in ([meta["gold"].strip()] if meta.get("gold") else []) + meta["competitors"] + meta["nearmiss"]:
        ids = m.tok.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            refs[w] = ids[0]
    return refs


# ---------------------------------------------------------------------------- main
def main(m: Model | None = None, **over):
    cfg = {**CFG, **over}
    torch.manual_seed(cfg["SEED"])
    m = m or Model.load(cfg["MODEL"], device=cfg["DEVICE"], dtype=cfg["DTYPE"])
    meta = word_meta(cfg["WORD"])
    rows = load_word(cfg["WORD"], "train_pool", n=cfg["N_TRAIN"])
    sets = {s: load_word(cfg["WORD"], s)[:cfg["N_EVAL_SLOTS"]] for s in ["eval", "competitor", "nearmiss", "control"]}
    equiv_rows = sets["eval"][:cfg["N_EQUIV"]]

    new_id = m.add_row()
    act = m.active_ids(rows + sets["eval"] + sets["control"])
    gold_id = m.tok.encode(meta["gold"], add_special_tokens=False)[0] if meta.get("gold") else None
    inits, info = make_inits(m, rows, new_id, act, k_span=cfg["K_SPAN"], gold_id=gold_id)
    refs = ref_words(m, meta)
    E0 = m.E.detach().clone()                                         # base model, restored per init
    others0 = {n: p.detach().clone() for n, p in m.hf.named_parameters() if p is not m.E} if cfg["LR_REST"] > 0 else {}
    H = {s: m.slot_states(r)["h"] for s, r in sets.items()}          # fixed while LR_REST == 0
    print(f"{cfg['WORD']}: train {len(rows)}, slots", {s: len(v) for s, v in H.items()}, "refs", refs)

    def logger(step, m, e, loss):
        if cfg["LR_REST"] > 0:                                        # model moved: recompute
            Hs, Ecur = {s: m.slot_states(r)["h"] for s, r in sets.items()}, m.E.detach()[:new_id].float()
        else:
            Hs, Ecur = H, E0[:new_id].float()
        d = {}
        for s in Hs:
            sm = slot_metrics(e, Hs[s], Ecur)
            sm["losers"] = m.decode_ids(*sm["losers"])
            d[s] = sm
        d["geom"] = geometry(m, e, new_id, act, info["mu"], refs, span_ids=info["span_ids"])
        if step % cfg["EQUIV_EVERY"] == 0 or step == cfg["STEPS"]:
            q = equivalence(m, equiv_rows, new_id)
            d["equiv"] = dict(kl_mean=q["kl"].mean().item(), kl_median=q["kl"].median().item(),
                              nll_new=q["nll_new"].mean().item(), nll_orig=q["nll_orig"].mean().item())
        return d

    tc = TrainConfig(steps=cfg["STEPS"], bs=cfg["BS"], lr_new=cfg["LR_NEW"], lr_rest=cfg["LR_REST"],
                     log_every=cfg["LOG_EVERY"], seed=cfg["SEED"])
    names = cfg["INITS"] + (["gold"] if gold_id is not None and "gold" not in cfg["INITS"] else [])
    for name in names:
        with torch.no_grad():
            m.E.copy_(E0)
            for n, p in m.hf.named_parameters():
                if n in others0:
                    p.copy_(others0[n])
        t0 = time.time()
        traj, logs = train(m, rows, new_id, inits[name], tc, logger=logger)
        tag = f"{name}_lrrest{cfg['LR_REST']:g}_n{cfg['N_TRAIN']}_s{cfg['SEED']}"
        out = Path(cfg["OUT"]) / cfg["WORD"] / tag
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps({**cfg, "init": name}, indent=2))
        torch.save(dict(traj=traj, refs=refs, ref_rows=E0[list(refs.values())].float().cpu(),
                        span_ids=info["span_ids"].cpu(), mu=info["mu"].cpu(),
                        H_train_mean=info["H_train"].mean(0).cpu()), out / "traj.pt")
        (out / "logs.json").write_text(json.dumps(logs, indent=1, default=str))
        f = logs[-1]
        print(f"[{name:8s}] {time.time() - t0:5.0f}s loss {f['loss']:.3f} | p_new eval {f['eval']['p_new']:.3f} "
              f"comp {f['competitor']['p_new']:.3f} ctrl {f['control']['p_new']:.4f} | "
              f"KL {f.get('equiv', {}).get('kl_mean', float('nan')):.3f} | nbrs {f['geom']['neighbors'][:4]}")
    return m


if __name__ == "__main__":
    main()
