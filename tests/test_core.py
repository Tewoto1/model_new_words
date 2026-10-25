"""Checks for the functions you implement. Tiny random GPT-Neo, CPU, a few seconds.

    python tests/test_core.py            # from the repo root; prints PASS / FAIL / TODO per test
    python tests/test_core.py collate    # only tests whose name contains 'collate'

TODO = the function still raises NotImplementedError. The tests check properties, not my code,
so any correct implementation passes.
"""
import copy
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPTNeoConfig, GPTNeoForCausalLM

from model import Model
from train import TrainConfig, collate, lm_loss, make_examples, setup_params, train

CFG = GPTNeoConfig(vocab_size=50257, hidden_size=64, num_layers=2, num_heads=4,
                   attention_types=[[["global", "local"], 1]], max_position_embeddings=512,
                   embed_dropout=0.0, attention_dropout=0.0, resid_dropout=0.0)
_TOK = None


def tiny():
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(os.environ.get("NW_TOKENIZER", "EleutherAI/gpt-neo-125M"))
    torch.manual_seed(0)
    return Model(GPTNeoForCausalLM(copy.deepcopy(CFG)), copy.deepcopy(_TOK), "cpu")


ROW = dict(left="Tom was sad. ", phrase="All of a sudden", right=", a big dog ran into the park. It was fun.",
           text="Tom was sad. {w}, a big dog ran into the park. It was fun.", orig="", n_w=1)
ROW["orig"] = ROW["left"] + ROW["phrase"] + ROW["right"]
ROW_HUGE = dict(left="The dog was ", phrase="huge", right=" and very happy.", text="The dog was {w} and very happy.",
                orig="The dog was huge and very happy.", n_w=1)


# ------------------------------------------------------------------ model.py
def test_add_row():
    m = tiny(); E0 = m.E.detach().clone()
    i = m.add_row()
    assert i == E0.shape[0] and m.vocab_size == E0.shape[0] + 1, "new id / vocab size wrong"
    assert torch.equal(m.E[:i], E0), "old rows changed"
    m.assert_tied()


def test_pad():
    m = tiny()
    ids, att = m.pad([[5, 6, 7], [8]])
    assert ids.shape == (2, 3) and ids.dtype == torch.long
    assert att.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert ids[0].tolist() == [5, 6, 7] and ids[1, 0] == 8 and ids[1, 1] == m.tok.pad_token_id


def test_capture_h_gives_logits():
    m = tiny(); ids, att = m.pad([m.encode("Once upon a time there was a cat.")])
    with torch.no_grad(), m.capture() as st:
        logits = m.forward(ids, att)
    assert st["h"] is not None and st["h"].shape == (1, ids.shape[1], m.E.shape[1])
    assert torch.allclose(st["h"] @ m.E.T, logits, atol=1e-4), "h @ E.T != logits: wrong tensor captured"


def test_capture_resid_and_cleanup():
    m = tiny(); ids, att = m.pad([m.encode("The sun was hot.")])
    with torch.no_grad(), m.capture(layers=[0, -1]) as st:
        m.forward(ids, att)
    assert set(st["resid"]) == {0, -1} and st["resid"][-1].shape == st["h"].shape
    ln_f = m.hf.transformer.ln_f
    assert torch.allclose(ln_f(st["resid"][-1]), st["h"], atol=1e-4), "resid[-1] is not the last block output"
    n_hooks = sum(len(mod._forward_hooks) for mod in m.hf.modules())
    assert n_hooks == 0, f"{n_hooks} hooks still attached after the with-block"


def test_capture_removes_hooks_on_error():
    m = tiny()
    with m.capture(layers=[0]):                      # NotImplementedError subclasses RuntimeError: check it runs first
        pass
    try:
        with m.capture(layers=[0]):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert sum(len(mod._forward_hooks) for mod in m.hf.modules()) == 0, "hooks leak when the body raises"


def test_states_at_last_batched_equals_single():
    m = tiny()
    seqs = [m.encode(s) for s in ["A cat.", "The big dog ran to the park and played all day.", "Hi"]]
    st = m.states_at_last(seqs, bs=8, layers=[0])
    assert st["h"].shape == (3, m.E.shape[1]) and st["h"].dtype == torch.float32
    for i, s in enumerate(seqs):
        one = m.states_at_last([s], bs=1, layers=[0])
        assert torch.allclose(st["h"][i], one["h"][0], atol=1e-4), f"row {i}: padding changed the answer"
        assert torch.allclose(st["resid"][0][i], one["resid"][0][0], atol=1e-4)


def test_neighbors_finds_itself():
    m = tiny(); act = torch.arange(1000)
    ids, cos = m.neighbors(m.E[123].detach(), k=5, active=act)
    assert ids.shape == (5,) and int(ids[0]) == 123 and abs(float(cos[0]) - 1) < 1e-4
    assert torch.all(cos[:-1] >= cos[1:]), "not sorted"


def test_project():
    m = tiny(); ids = [10, 20, 30]
    V = m.E.detach()[ids].float()
    v_in = 2 * V[0] - V[2]
    assert torch.allclose(m.project(v_in, ids), v_in, atol=1e-4), "a vector in the span must project to itself"
    v = torch.randn(m.E.shape[1])
    r = v - m.project(v, ids)
    assert torch.allclose(V @ r, torch.zeros(3), atol=1e-3), "residual must be orthogonal to every E[id]"
    vb = torch.randn(4, m.E.shape[1])
    assert m.project(vb, ids).shape == (4, m.E.shape[1]), "batched input"


# ------------------------------------------------------------------ train.py
def test_collate():
    m = tiny(); new = m.add_row()
    ex = make_examples(m, [ROW, ROW_HUGE], new, use_orig=False, max_len=256)
    ids, att, lab = collate(m, ex, loss_from_slot=True)
    for i, (s, slot) in enumerate(ex):
        assert (lab[i, :slot] == -100).all(), "positions before the slot must be ignored"
        assert lab[i, slot] == new, "the label AT the slot (= NEW) must be kept"
        assert lab[i, slot:len(s)].tolist() == s[slot:], "labels after the slot = ids"
        assert (lab[i, len(s):] == -100).all(), "padding must be ignored"
    _, _, lab2 = collate(m, ex, loss_from_slot=False)
    assert lab2[0, 0] == ex[0][0][0], "loss_from_slot=False keeps the prefix"


def test_lm_loss_matches_hf():
    m = tiny(); new = m.add_row()
    ids, att, lab = collate(m, make_examples(m, [ROW, ROW_HUGE], new, False, 256), True)
    logits = m.forward(ids, att)
    ref = m.hf(input_ids=ids, attention_mask=att, labels=lab).loss
    assert torch.allclose(lm_loss(logits, lab), ref, atol=1e-5), "shift or ignore_index wrong"


def test_setup_params_row_only():
    m = tiny(); new = m.add_row()
    opt, hook = setup_params(m, new, lr_new=1e-2, lr_rest=0.0)
    assert hook is not None
    ids, att, lab = collate(m, make_examples(m, [ROW], new, False, 256), True)
    lm_loss(m.forward(ids, att), lab).backward()
    g = m.E.grad
    assert g is not None and g[new].abs().sum() > 0, "new row got no gradient"
    assert (g[:new] == 0).all(), "old rows must get zero gradient"
    assert all(not p.requires_grad for p in m.hf.parameters() if p is not m.E), "other params must be frozen"
    hook.remove()


def test_train_row_only_moves_only_new_row():
    m = tiny(); new = m.add_row()
    before = {n: p.detach().clone() for n, p in m.hf.named_parameters()}
    init = m.E.detach()[:new].mean(0)
    seen = []
    traj, logs = train(m, [ROW, ROW_HUGE], new, init, TrainConfig(steps=30, bs=2, lr_new=5e-2, log_every=10),
                       logger=lambda step, m, e, loss: seen.append(step) or {})
    for n, p in m.hf.named_parameters():
        if p is m.E:
            assert torch.equal(p[:new], before[n][:new]), "old rows moved"
            assert not torch.equal(p[new], before[n][new]), "new row did not move"
        else:
            assert torch.equal(p, before[n]), f"{n} moved"
    assert traj.shape == (4, m.E.shape[1]) and torch.allclose(traj[0], init.float()), "traj: log at 0,10,20 and end"
    assert seen == [0, 10, 20, 30] and len(logs) == 4 and "loss" in logs[-1]
    assert logs[-1]["loss"] < logs[1]["loss"], "loss did not go down"
    assert all(not p.requires_grad for p in m.hf.parameters()), "leave everything frozen at the end"


def test_train_lr_rest_moves_old_rows_slowly():
    m = tiny(); new = m.add_row()
    E0 = m.E.detach().clone()
    train(m, [ROW, ROW_HUGE], new, E0[:new].mean(0), TrainConfig(steps=5, bs=2, lr_new=1e-2, lr_rest=1e-4))
    d_old = (m.E[:new] - E0[:new]).abs().max().item()
    d_new = (m.E[new] - E0[new]).abs().max().item()
    assert 0 < d_old < 5 * 5 * 1e-4 * 1.01, f"old rows should move ~lr_rest per step (moved {d_old:.2e})"
    assert d_new > 10 * d_old, "new row should move much more than old rows"


# ------------------------------------------------------------------ experiments/neologism.py
def test_slot_metrics_duplicate_row():
    from experiments.neologism import slot_metrics
    torch.manual_seed(0)
    H, E = torch.randn(7, 16), torch.randn(50, 16)
    j = 3
    out = slot_metrics(E[j].clone(), H, E)
    p0 = (H @ E.T).softmax(-1)[:, j]
    assert abs(out["p_new"] - (p0 / (1 + p0)).mean().item()) < 1e-5, "duplicate row must take p0/(1+p0)"
    ids, dp = out["losers"]
    P0 = (H @ E.T).softmax(-1)
    want = (-P0 * (p0 / (1 + p0))[:, None]).mean(0)      # every old token is scaled by 1/(1+p0_j)
    want[j] = (-p0 ** 2 / (1 + p0)).mean()                # (same formula, written out for j)
    assert torch.allclose(dp, want[ids], atol=1e-6), "losers: dp = p_with_new - p_without, averaged over slots"
    assert torch.allclose(dp, want.topk(len(ids), largest=False).values, atol=1e-6), "losers not the most negative"
    assert -1e9 < out["margin"] <= 1e-5 and 0 <= out["top1_frac"] <= 1


def test_equivalence_zero_for_identical_row():
    from experiments.neologism import equivalence
    m = tiny(); new = m.add_row()
    gold = m.tok.encode(" huge", add_special_tokens=False)[0]
    m.set_row(new, m.E.detach()[gold].clone())
    q = equivalence(m, [ROW_HUGE], new)
    assert q["kl"].shape == (1,) and q["kl"][0].abs() < 1e-5, "same embedding => same next-token distributions"
    assert torch.allclose(q["nll_new"], q["nll_orig"], atol=1e-4)
    m.set_row(new, torch.randn(m.E.shape[1]))
    assert equivalence(m, [ROW_HUGE, ROW], new)["kl"].min() > 0


def test_make_inits():
    from experiments.neologism import make_inits
    m = tiny(); new = m.add_row()
    act = m.active_ids([ROW, ROW_HUGE])
    gold = m.tok.encode(" huge", add_special_tokens=False)[0]
    inits, info = make_inits(m, [ROW, ROW_HUGE], new, act, k_span=3, gold_id=gold)
    assert set(inits) == {"mean_emb", "mean_h", "in_span", "gold"}
    rn = m.E.detach()[act].norm(dim=1).mean()
    for k in ["mean_h", "in_span"]:
        assert abs(inits[k].norm() - rn) < 1e-3, f"{k} not rescaled to mean active-row norm"
    assert torch.allclose(inits["mean_emb"], m.E.detach()[act].mean(0))
    assert m.span_fraction(inits["in_span"], info["span_ids"]) > 0.999, "in_span must lie in the span"
    assert info["H_train"].shape == (2, m.E.shape[1]) and info["span_ids"].shape == (3,)


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else ""
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and pat in n]
    res = {"PASS": 0, "FAIL": 0, "TODO": 0}
    for n, f in tests:
        try:
            f(); r = "PASS"; msg = ""
        except NotImplementedError as e:
            r, msg = "TODO", str(e).split("\n")[0]
        except Exception as e:
            r = "FAIL"; tb = traceback.extract_tb(e.__traceback__)
            msg = f"{type(e).__name__}: {e} ({tb[-1].name}:{tb[-1].lineno})"
        res[r] += 1
        print(f"{r:4s}  {n:45s} {msg}")
    print(res)
