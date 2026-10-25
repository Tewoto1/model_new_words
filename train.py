"""Classical SFT loop (no Trainer, no HF loss), with one twist: what is allowed to move.

lr_rest = 0   : only the new row of E trains (neologism learning). Every other weight is frozen, so
                every h before the first NEW in a sequence is fixed; only e_new moves.
lr_rest > 0   : everything trains; old rows of E and all other weights at lr_rest, new row at lr_new.
use_orig=True : control run on the same rows with the real phrase instead of {w} (needs lr_rest > 0).

Why lr and not "smaller gradient": Adam divides by sqrt(running mean of g^2), so multiplying a
gradient by 0.01 leaves the update (almost) unchanged. To make something move less, lower its lr.
Old rows and the new row live in ONE tensor (E), so they cannot be in different param groups.
Trick: give E lr_new, and after each step pull the old rows back:
    E[:new] = before + (lr_rest / lr_new) * (after - before)
which is exactly an Adam step at lr_rest for those rows (Adam's update is linear in lr).
"""
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


@dataclass
class TrainConfig:
    steps: int = 300
    bs: int = 16
    lr_new: float = 5e-3
    lr_rest: float = 0.0
    log_every: int = 10
    max_len: int = 256
    loss_from_slot: bool = True      # ignore the loss on tokens before the slot (they don't depend on e_new)
    use_orig: bool = False
    seed: int = 0


def make_examples(m, rows, new_id, use_orig: bool, max_len: int):
    """rows -> list of (ids, slot). slot = index of the first NEW (or of the phrase start, use_orig)."""
    out = []
    for r in rows:
        if use_orig:
            ids = m.tok.encode(r["orig"], add_special_tokens=False)
            slot = len(m.tok.encode(r["left"].rstrip(" "), add_special_tokens=False))
        else:
            ids = m.encode(r["text"], new_id)
            slot = ids.index(new_id)
        out.append((ids[:max_len], slot))
    return out


def collate(m, batch, loss_from_slot: bool = True):
    """batch: list of (ids, slot) -> (ids [B, L], att [B, L], labels [B, L]).
    labels = ids, except -100 on padding and (if loss_from_slot) on every position before the slot.
    labels[i, slot] itself is KEPT: predicting NEW from the position before it is the output-side
    signal for e_new. (Shifting happens in lm_loss, not here.)
    """
    all_ids = [torch.tensor(ids, dtype=torch.long) for ids, _ in batch]   # pad_sequence needs tensors
    slots = torch.tensor([s for _, s in batch])
    lengths = torch.tensor([len(b) for b, _ in batch])
    all_ids = pad_sequence(all_ids, batch_first = True, padding_value = m.tok.pad_token_id)
    B, L = all_ids.shape
    pos = torch.arange(L)
    is_real = pos[None, :] < lengths[:, None]
    att = is_real                                      # attention: the model must still SEE the prefix
    keep = is_real & (pos[None, :] >= slots[:, None]) if loss_from_slot else is_real   # loss mask
    labels = all_ids.masked_fill(~keep, -100)
    return all_ids.to(m.device), att.long().to(m.device), labels.to(m.device)


def lm_loss(logits, labels):
    """Next-token cross entropy. logits [B, L, V], labels [B, L] with -100 = ignore.
    Position t predicts token t+1. Mean over non-ignored targets. Compute in float32.
    """
    x, y = logits[:, :-1].float(), labels[:, 1:]
    B, L, V = x.shape
    return F.cross_entropy(x.reshape(-1, V), y.reshape(-1))


def setup_params(m, new_id: int, lr_new: float, lr_rest: float):
    """Decide what trains. Returns (optimizer, hook_handle_or_None).
      lr_rest == 0: every parameter frozen except E; E's gradient masked so only row new_id is nonzero
                    (zero grad -> Adam moment stays 0 -> that row never moves, with weight_decay=0).
      lr_rest  > 0: all parameters train; E in a group at lr_new, everything else at lr_rest
                    (the old-row pull-back happens in train(), after each step).
    Adam, weight_decay=0.

    Hints:
        p.requires_grad_(False) freezes p in place. m.hf.parameters() yields the tied E once.
        Compare parameters by identity (p is m.E), never ==.
        tensor.register_hook(fn): fn(grad) -> modified grad, runs during backward; returns a handle (.remove()).
        A [V, 1] mask broadcasts over d.
        Per-group lr: torch.optim.Adam([{"params": [E], "lr": a}, {"params": others, "lr": b}], weight_decay=0.0).
    """
    E = m.E
    others = [param for param in m.hf.parameters() if param is not E]
    E.requires_grad_(True)

    if (lr_rest == 0):
        for p in others:
            p.requires_grad_(False)
        mask = torch.zeros(E.shape[0], 1, device = E.device, dtype = E.dtype)
        mask[new_id] = 1
        hook = E.register_hook(lambda g: g*mask)
        optim = torch.optim.Adam([E], lr=lr_new, weight_decay=0.0)
    else:
        for p in others:
            p.requires_grad_(True)
        hook = None
        optim = torch.optim.Adam(
            [{"params": [E], "lr": lr_new},
             {"params": others, "lr": lr_rest}],
            weight_decay=0.0,
        )
    return optim, hook


def train(m, rows, new_id: int, init: torch.Tensor, cfg: TrainConfig, logger=None):
    """The SFT loop. Returns (traj [n_logs, d] float32 cpu, logs: list of dicts).

    Steps:
      1. m.set_row(new_id, init); build examples; optimizer via setup_params.
      2. model.train() only if lr_rest > 0 (dropout), else eval().
      3. for step in range(cfg.steps):
           - every cfg.log_every steps (and once after the loop): snapshot E[new_id] into traj and,
             if logger, call logger(step, m, e_new, last_loss) -> dict, append (add step, loss).
             Put the model in eval() while logging, back afterwards.
           - sample cfg.bs examples with a seeded random.Random(cfg.seed) (with replacement)
           - collate -> forward -> lm_loss -> zero_grad -> backward -> (copy old rows if lr_rest>0)
             -> step -> (pull old rows back if lr_rest>0)
      4. remove the hook, model.eval(), requires_grad_(False) on everything.

    Hints:
        Classic step: opt.zero_grad(set_to_none=True); loss = lm_loss(m.forward(ids, att), labels);
        loss.backward(); opt.step().
        Pull-back: before = m.E.data[:new_id].clone() before opt.step(); after it, inside torch.no_grad(),
        m.E.data[:new_id] = before + ratio * (m.E.data[:new_id] - before).
        loss.item() -> Python float (forces a GPU sync; once per step is fine).
        m.hf.train()/eval() only toggles dropout; requires_grad decides what learns.
        torch.stack(list of [d]) -> [n, d].
    """
    m.set_row(new_id, init)
    examples = make_examples(m, rows, new_id, cfg.use_orig, cfg.max_len)
    optim, hook = setup_params(m, new_id, cfg.lr_new, cfg.lr_rest)
    rng = random.Random(cfg.seed)
    E = m.E

    def set_mode():                                    # Model wraps the HF model: m.hf.train(), not m.train()
        if cfg.lr_rest > 0:
            m.hf.train()
        else:
            m.hf.eval()
    set_mode()

    traj, logs, last_loss = [], [], float("nan")

    def log(step):
        e = E[new_id].detach().float().clone()        # clone: E[new_id] is a live view of E
        traj.append(e.cpu())
        if logger is not None:
            m.hf.eval()                                # metrics without dropout
            d = logger(step, m, e, last_loss)
            d.update(step=step, loss=last_loss)
            logs.append(d)
            set_mode()

    for step in range(cfg.steps):
        if step % cfg.log_every == 0:
            log(step)
        batch = [examples[rng.randrange(len(examples))] for _ in range(cfg.bs)]   # with replacement
        ids, att, labels = collate(m, batch, cfg.loss_from_slot)                  # padding lives in collate
        loss = lm_loss(m.forward(ids, att), labels)

        optim.zero_grad(set_to_none = True)
        loss.backward()
        if cfg.lr_rest > 0:
            before = E.data[:new_id].clone()
        optim.step()
        if cfg.lr_rest > 0:                            # old rows: Adam step at lr_rest instead of lr_new
            with torch.no_grad():
                E.data[:new_id] = before + (cfg.lr_rest / cfg.lr_new) * (E.data[:new_id] - before)
        last_loss = loss.item()
    log(cfg.steps)

    if hook is not None:
        hook.remove()
    m.hf.eval()
    for p in m.hf.parameters():
        p.requires_grad_(False)
    return torch.stack(traj), logs
