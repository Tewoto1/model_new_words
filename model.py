"""Model side: loading, the embedding matrix, new rows, internals, and geometry in embedding space.

Everything an experiment needs from the network goes through `Model`, so experiments never touch
HF internals directly.

Spaces (tied model): E = embedding matrix [V, d], used both to embed and to decode.
  input embedding     x_t = E[id_t]                 (Gemma-style models scale by sqrt(d); GPT-Neo/Qwen do not)
  pre-decode vector   h_t = final_norm(resid_L)     logits_t = h_t @ E.T
A new word is one extra row of E (no tokenizer change): encode() puts its id where the text has " {w}".

Models we use:
  roneneldan/TinyStories-33M : GPT-Neo, tied, tokenizer EleutherAI/gpt-neo-125M (50k BPE; TinyStories
                               only uses a few thousand of those tokens). Absolute position embeddings,
                               so always RIGHT-pad batches.
  Qwen/Qwen3-0.6B(-Base)     : tied, d=1024, vocab 152k, 28 layers.
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F

TOKENIZER_FOR = {"roneneldan/TinyStories-33M": "EleutherAI/gpt-neo-125M",
                 "roneneldan/TinyStories-8M": "EleutherAI/gpt-neo-125M",
                 "roneneldan/TinyStories-1M": "EleutherAI/gpt-neo-125M"}


class Model:
    def __init__(self, hf_model, tok, device: str = "cpu"):
        """Wrap an already-built HF causal LM + tokenizer. Use Model.load(name) for pretrained ones."""
        self.hf, self.tok, self.device = hf_model.to(device).eval(), tok, device
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.assert_tied()

    @classmethod
    def load(cls, name: str, device: str = "cuda", dtype: str = "float32"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER_FOR.get(name, name))
        hf = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype))
        return cls(hf, tok, device)

    # ------------------------------------------------------------------ embeddings
    @property
    def E(self) -> torch.nn.Parameter:
        """[V, d] embedding matrix (same tensor as the unembedding)."""
        return self.hf.get_input_embeddings().weight

    @property
    def vocab_size(self) -> int:
        return self.E.shape[0]

    @property
    def blocks(self):
        """The list of transformer blocks (GPT-Neo: transformer.h, Qwen/Llama: model.layers)."""
        inner = getattr(self.hf, "transformer", None) or getattr(self.hf, "model")
        return getattr(inner, "h", None) or getattr(inner, "layers")

    def assert_tied(self):
        a, b = self.hf.get_input_embeddings().weight, self.hf.get_output_embeddings().weight
        assert a.data_ptr() == b.data_ptr(), "embeddings are not tied"

    def add_row(self) -> int:
        """Append one row to the (tied) embedding matrix and return its id.

        After this, self.vocab_size == old + 1, the old rows are bit-identical, and input/output
        embeddings are still the same tensor. The value of the new row does not matter (set it with
        set_row); skip HF's default random-normal init.

        Hints:
            model.resize_token_embeddings(n, mean_resizing=False) resizes BOTH embedding matrices and re-ties them
            (mean_resizing=False skips HF's random init of the new row).
            x.data_ptr() is the address of x's storage: tied = same data_ptr (self.assert_tied() checks this).
        """
        new_id = self.vocab_size
        self.hf.resize_token_embeddings(new_id + 1, mean_resizing=False)
        self.assert_tied()
        return new_id

    def set_row(self, i: int, v: torch.Tensor):
        with torch.no_grad():
            self.E[i] = v.to(self.E.dtype)

    # ------------------------------------------------------------------ text <-> ids
    def encode(self, text: str, new_id: int | None = None) -> list[int]:
        """'left {w}right {w}more' -> enc(left) + [NEW] + enc(right) + [NEW] + enc(more).
        NEW plays the role of ' <phrase>' (with its leading space). Pieces are tokenized separately."""
        parts, ids = text.split(" {w}"), []
        for i, p in enumerate(parts):
            if i > 0:
                assert new_id is not None, "text has {w} but no new_id"
                ids.append(new_id)
            if p:
                ids += self.tok.encode(p, add_special_tokens=False)
        return ids

    def encode_pair(self, row: dict, new_id: int):
        """(ids_new, ids_orig, slot, n_right) for a data row: identical left tokens, then NEW vs the
        phrase tokens, then identical right tokens up to the next {w}. slot = index of NEW in ids_new."""
        left = self.tok.encode(row["left"].rstrip(" "), add_special_tokens=False)
        phr = self.tok.encode(" " + row["phrase"].strip(), add_special_tokens=False)
        right_first = row["text"].split(" {w}")[1] if row["text"] else row["right"]
        r = self.tok.encode(right_first, add_special_tokens=False) if right_first else []
        return left + [new_id] + r, left + phr + r, len(left), len(r)

    def pad(self, seqs: list[list[int]]):
        """Right-pad a list of id lists. Returns (ids [B, L] long, att [B, L] long) on self.device;
        att is 1 on real tokens, 0 on padding; pad value = tok.pad_token_id.

        Hints:
            torch.full((B, L), value, dtype=torch.long), torch.zeros_like(x).
            Slice-assign each row: ids[i, :n] = torch.tensor(s); move to self.device once at the end.
            Why right padding: GPT-Neo has absolute position embeddings; left padding shifts every position
            unless you also pass position_ids.
        """
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), self.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            att[i, :len(s)] = 1
        return ids.to(self.device), att.to(self.device)

    # ------------------------------------------------------------------ internals
    def forward(self, ids, att=None):
        """Plain forward. Returns logits [B, L, V]."""
        return self.hf(input_ids=ids, attention_mask=att).logits

    @contextmanager
    def capture(self, layers=()):
        """Record internals during any forward run inside the `with` block:
            with m.capture(layers=[0, 3]) as st:
                m.forward(ids, att)
            st["h"]          [B, L, d]  pre-decode vector (the input to lm_head), logits = h @ E.T
            st["resid"][l]   [B, L, d]  residual stream after block l (negative l allowed)
        Tensors are kept on device and NOT detached (so this also works inside training).
        Hooks must be removed when the block exits, even if the forward raised.

        Hints:
            module.register_forward_hook(fn) -> handle. fn(module, args, output) runs after module.forward;
            args = tuple of positional inputs, so for lm_head (self.hf.get_output_embeddings()) args[0] is h.
            handle.remove() detaches the hook.
            self.blocks[l % len(self.blocks)] handles negative l. GPT-Neo blocks return a tuple (hidden, ...)
            -> out[0]; some HF versions/models return a bare tensor: check isinstance(out, tuple).
            Hooks made in a loop: def hook(mod, args, out, l=l) - the default arg freezes the current l
            (a plain closure would see the last l for every hook).
            @contextmanager (already applied): code before `yield st` is setup, after is teardown. Wrap the
            yield in try/finally so the hooks are removed even if the body raises.
        """
        st = {"h": None, "resid": {}}
        hs = [self.hf.get_output_embeddings().register_forward_hook(
            lambda mod, args, out: st.__setitem__("h", args[0]))]
        n = len(self.blocks)
        for l in layers:
            def hook(mod, args, out, l=l):
                st["resid"][l] = out[0] if isinstance(out, tuple) else out
            hs.append(self.blocks[l % n].register_forward_hook(hook))
        try:
            yield st
        finally:
            for h in hs:
                h.remove()

    @torch.no_grad()
    def states_at_last(self, seqs: list[list[int]], bs: int = 64, layers=()) -> dict:
        """For each sequence, the internals at its LAST real token, batched.
        Returns {"h": [N, d] float32, "resid": {l: [N, d] float32}} on self.device.
        Must give the same answer as running each sequence alone (right padding + gather).

        Hints:
            Batch with self.pad, run self.forward inside self.capture(layers).
            Last real index per row: att.sum(1) - 1.
            One position per row: x[torch.arange(B, device=x.device), last] -> [B, d] (advanced indexing).
            .float() each piece, torch.cat at the end. @torch.no_grad() is already on the method.
        """
        hs, rs = [], {l: [] for l in layers}
        for i in range(0, len(seqs), bs):
            ids, att = self.pad(seqs[i:i + bs])
            with self.capture(layers) as st:
                self.forward(ids, att)
            last = att.sum(1) - 1
            b = torch.arange(ids.shape[0], device=ids.device)
            hs.append(st["h"][b, last].float())
            for l in layers:
                rs[l].append(st["resid"][l][b, last].float())
        return {"h": torch.cat(hs), "resid": {l: torch.cat(v) for l, v in rs.items()}}

    def slot_states(self, rows: list[dict], layers=()) -> dict:
        """Internals at the position that predicts the slot (end of `left`) for each data row.
        NEW is never in the context here, so these do not depend on the new row."""
        seqs = [self.tok.encode(r["left"].rstrip(" "), add_special_tokens=False) for r in rows]
        return self.states_at_last(seqs, layers=layers)

    # ------------------------------------------------------------------ geometry in embedding space
    def active_ids(self, rows: list[dict], field: str = "orig") -> torch.Tensor:
        """Token ids the corpus actually uses. Restrict neighbour search and norms to these:
        TinyStories never feeds most of GPT-Neo's 50k rows as inputs."""
        s = set()
        for r in rows:
            s.update(self.tok.encode(r[field], add_special_tokens=False))
        return torch.tensor(sorted(s), device=self.device)

    def neighbors(self, v: torch.Tensor, k: int = 8, active=None, mu=None, upto: int | None = None):
        """Top-k rows of E[:upto] (restricted to `active` ids if given) by CENTERED cosine with v:
        cos(E[i] - mu, v - mu), mu defaulting to the mean of the candidate rows.
        Returns (ids [k] long, cos [k] float32).

        Hints:
            F.normalize(x, dim=-1) = x / ||x||; then (normalized rows) @ (normalized v) is the cosine vector.
            values, idx = s.topk(k) is sorted descending; idx indexes the CANDIDATE list -> map back with ids[idx].
            self.E is a Parameter: .detach() before doing math on it.
        """
        E = self.E.detach()[:upto].float()
        ids = torch.arange(E.shape[0], device=E.device) if active is None else active
        R = E[ids]
        mu = R.mean(0) if mu is None else mu
        s = F.normalize(R - mu, dim=1) @ F.normalize(v.float() - mu, dim=0)
        t = s.topk(k)
        return ids[t.indices], t.values

    def decode_ids(self, ids, vals=None):
        ids = [int(i) for i in ids]
        if vals is None:
            return [self.tok.decode([i]) for i in ids]
        return [(self.tok.decode([i]), round(float(c), 3)) for i, c in zip(ids, vals)]

    def project(self, v: torch.Tensor, ids) -> torch.Tensor:
        """Least-squares projection of v [d] (or [N, d]) onto span(E[ids]). Same shape as v, float32.
        The rows are not orthogonal, so this is NOT a sum of 1-D projections.

        Hints:
            Coefficients c solve the normal equations (V V^T) c = V v, with V = E[ids] [k, d].
            torch.linalg.solve(A, B) solves A X = B (prefer it to torch.linalg.inv: more stable, no explicit inverse).
            Treat v as [N, d] internally (v.reshape(-1, d)) so one code path covers [d] and [N, d]; reshape back.
            If rows are nearly collinear, torch.linalg.lstsq(V.T, v2.T).solution is the robust alternative.
        """
        V = self.E.detach()[ids].float()
        h2 = v.float().reshape(-1, V.shape[1])
        c = torch.linalg.solve(V @ V.T, V @ h2.T).T
        return (c @ V).reshape(v.shape)

    def span_fraction(self, v, ids) -> torch.Tensor:
        """||P v||^2 / ||v||^2 for P = projection onto span(E[ids])."""
        p = self.project(v, ids)
        return p.pow(2).sum(-1) / v.float().pow(2).sum(-1)
