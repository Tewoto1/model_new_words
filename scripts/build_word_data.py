"""Build new-word datasets from TinyStories by replacing a real phrase with a nonce token.

In target sets (train_pool, eval) every occurrence of the phrase in the window becomes {w}; the
slot (end of `left`) is always the first one, so h at the slot is untouched by the new row.
Every example keeps the original text, so for each word we know what the nonce "should" mean:
  text = left + " {w}" + right       (what the model is trained / evaluated on; {w} = one new row)
  orig = left + phrase + right       (same context with the real phrase; for KL-equivalence and the
                                      synonym-swap control in Phase B)

Files per word, in data/words/<word>/:
  meta.json          spec, counts, GPT-Neo tokenization of the phrase
  train_pool.jsonl   TinyStories train occurrences, seeded order; train_N = rows with rank < N
  eval.jsonl         held-out occurrences (validation split; if it has < EVAL_MIN rows, the rest are train stories held out of the pool,
                     validation is too small, tagged source="train_heldout"); frame_seen marks
                     whether the previous word ever precedes the phrase in train_200 (unseen = harder)
  competitor.jsonl   validation slots where a competitor fills the slot (" big", " suddenly", ...);
                     here {w} in `text` marks the competitor's slot (swap test), `slot` names it
  nearmiss.jsonl     validation slots with a related but wrong word/phrase (same format)
  control.jsonl      random validation slots unrelated to the word (left/right only, phrase="")

Usage:
  python scripts/build_word_data.py --raw raw_tinystories --out data/words [--words syn_huge,...]
  raw_tinystories/ must hold the roneneldan/TinyStories parquet files (downloaded if missing).
"""
import argparse, json, random, re, urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/"
FILES = {
    "train": ["train-00000-of-00004-2d5a1467fff1081b.parquet", "train-00001-of-00004-5852b56a2bd28fd9.parquet",
              "train-00002-of-00004-a26307300439e943.parquet", "train-00003-of-00004-d243063613e5a057.parquet"],
    "val": ["validation-00000-of-00001-869c898b519ad725.parquet"],
}

# phrases: what the nonce replaces (matched case-insensitively, longest first).
# gold: an existing single token that should be (near) equivalent, if any.
WORDS = {
    "syn_huge": dict(
        rung="synonym", nonce="blick", phrases=["huge"], gold=" huge",
        competitors=["big", "giant", "enormous", "large"], nearmiss=["small", "tiny"],
        note="calibration: nonce = an existing token, so the answer is E[' huge']"),
    "idiom_all_of_a_sudden": dict(
        rung="phrase_idiom", nonce="zorp", phrases=["all of a sudden"], gold=" suddenly",
        competitors=["suddenly"], nearmiss=["then", "after that"],
        note="non-compositional idiom with a one-token paraphrase"),
    "comp_very_big": dict(
        rung="phrase_compositional", nonce="gloob", phrases=["very big"], gold=None,
        competitors=["huge", "enormous", "giant", "big"], nearmiss=["very small", "very happy"],
        note="compositional; is it ' big', ' huge', or between?"),
    "compound_ice_cream": dict(
        rung="phrase_lexicalized", nonce="frim", phrases=["ice cream"], gold=None,
        competitors=["cake", "cookies", "candy", "treat"], nearmiss=["ice", "cream"],
        note="lexicalized compound, meaning not ice + cream"),
    "temporal_the_next_day": dict(
        rung="phrase_temporal", nonce="tovan", phrases=["the next day"], gold=None,
        competitors=["the next morning", "tomorrow", "one day"], nearmiss=["that day", "the day"],
        note="includes the article; slot intention is mostly ' the'"),
    "blend_happy_and_sad": dict(
        rung="blend", nonce="glumpy", phrases=["happy and sad", "sad and happy", "happy but sad", "sad but happy"],
        gold=None, competitors=["happy", "sad"], nearmiss=["excited", "scared"],
        left_must_match=r"\b(?:was|were|is|are|am|be|been|felt|feel|feels|feeling|looked|looks|seemed|both|so|very)\s$",
        note="four surface forms merged into one word; only feeling frames (drops lists like "
             "'fast and slow, happy and sad'); small data"),
}

POOL_CAP, EVAL_MIN, EVAL_CAP, SLOT_CAP, CONTROL_N = 5000, 200, 600, 300, 500
N_LEFT_SENTS, N_RIGHT_SENTS = 2, 1
SENT_END = re.compile(r"""[.!?]["']?\s+""")


def fetch(raw: Path):
    raw.mkdir(parents=True, exist_ok=True)
    for split, names in FILES.items():
        for n in names:
            p = raw / n
            if not p.exists():
                print("downloading", n); urllib.request.urlretrieve(HF + n, p)


def read(raw: Path, split: str):
    import pandas as pd
    return pd.concat([pd.read_parquet(raw / n) for n in FILES[split]], ignore_index=True)["text"].tolist()


def phrase_re(phrases):
    alts = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
    return re.compile(r"\b(?:" + alts + r")\b", re.I)


def boundaries(text):
    return [0] + [m.end() for m in SENT_END.finditer(text)] + [len(text)]


def window(text, s, e, bnds):
    """Slice [left | phrase | right] around span (s, e), keeping exact original characters."""
    i = max(k for k, b in enumerate(bnds) if b <= s)          # sentence containing s starts at bnds[i]
    j = min(k for k, b in enumerate(bnds) if b >= e)          # first boundary at/after e
    ls = bnds[max(0, i - N_LEFT_SENTS)]
    re_ = bnds[min(len(bnds) - 1, j + N_RIGHT_SENTS)]
    return text[ls:s], text[s:e], text[e:re_].rstrip(), s == bnds[i]


def last_word(left):
    m = re.search(r"([A-Za-z']+)\W*$", left)
    return m.group(1).lower() if m else ""


def occurrences(stories, rx, source, block_rx=None, cap=None, seed=0, left_rx=None):
    """All usable occurrences of rx. block_rx: spans to exclude (e.g. the target phrase when mining
    competitor ' big' so ' very big' is not counted). Usable = preceded by a single space, nonempty left."""
    out, seen = [], set()
    order = list(range(len(stories))); random.Random(seed).shuffle(order)
    for si in order:
        t = stories[si]
        ms = list(rx.finditer(t))
        if not ms: continue
        blocked = [(m.start(), m.end()) for m in block_rx.finditer(t)] if block_rx else []
        bnds = None
        for m in ms:
            s, e = m.start(), m.end()
            if any(s < be and bs < e for bs, be in blocked): continue
            if s == 0 or t[s - 1] != " ": continue
            bnds = bnds or boundaries(t)
            left, phr, right, initial = window(t, s, e, bnds)
            if not left.strip(): continue
            if block_rx is None and rx.search(left): continue          # slot must be the first {w} in the window
            if left_rx is not None and not left_rx.search(left): continue
            key = (left[-80:], right[:40])
            if key in seen: continue
            seen.add(key)
            out.append(dict(source=source, story=si, left=left, phrase=phr, right=right,
                            frame=last_word(left), sent_initial=initial))
            break                                              # one example per story
        if cap and len(out) >= cap: break
    return out


def finish(rows, word, extra=None, rx=None):
    """rx given (target sets): later occurrences of the phrase in `right` also become {w}."""
    for k, r in enumerate(rows):
        r["id"] = f"{word}/{r['source']}/{r['story']}"
        right = re.sub(r" ?" + rx.pattern, " {w}", r["right"], flags=re.I) if rx is not None else r["right"]
        r["text"] = r["left"].rstrip(" ") + " {w}" + right if r["phrase"] else None
        r["n_w"] = r["text"].count("{w}") if r["text"] else 0
        r["orig"] = r["left"] + r["phrase"] + r["right"]
        if extra: r.update(extra)
    return rows


def controls(stories, bad_rx, n, seed):
    rng, out = random.Random(seed), []
    while len(out) < n:
        si = rng.randrange(len(stories)); t = stories[si]
        if bad_rx.search(t): continue
        sp = [m.start() for m in re.finditer(r"(?<= )[A-Za-z]", t)]
        if not sp: continue
        s = rng.choice(sp); bnds = boundaries(t)
        left, _, right, initial = window(t, s, s, bnds)
        if not left.strip(): continue
        out.append(dict(source="val", story=si, left=left, phrase="", right=right,
                        frame=last_word(left), sent_initial=initial))
    return out


def write_jsonl(p, rows):
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(word, spec, train, val, out_dir, tok=None):
    d = out_dir / word; d.mkdir(parents=True, exist_ok=True)
    rx = phrase_re(spec["phrases"])

    lrx = re.compile(spec["left_must_match"], re.I) if spec.get("left_must_match") else None
    ev = occurrences(val, rx, "val", cap=EVAL_CAP, left_rx=lrx)
    allr = occurrences(train, rx, "train", cap=POOL_CAP + EVAL_MIN, seed=1, left_rx=lrx)
    need = max(0, min(EVAL_MIN, len(allr) // 3) - len(ev))     # validation too small -> hold out train stories
    pool, held = allr[:len(allr) - need][:POOL_CAP], allr[len(allr) - need:] if need else []
    for r in held: r["source"] = "train_heldout"
    ev += held
    for k, r in enumerate(pool): r["rank"] = k
    finish(pool, word, rx=rx)

    pool_texts = {r["orig"] for r in pool}
    ev = [r for r in ev if r["left"] + r["phrase"] + r["right"] not in pool_texts]   # train/val dupes
    frames200 = {r["frame"] for r in pool if r["rank"] < 200}
    finish(ev, word, rx=rx)
    for r in ev: r["frame_seen_train200"] = r["frame"] in frames200

    comp = []
    for c in spec["competitors"]:
        comp += finish(occurrences(val, phrase_re([c]), "val", block_rx=rx, cap=SLOT_CAP), word, {"slot": c})
    near = []
    for c in spec["nearmiss"]:
        near += finish(occurrences(val, phrase_re([c]), "val", block_rx=rx, cap=SLOT_CAP), word, {"slot": c})
    bad = phrase_re(spec["phrases"] + spec["competitors"] + spec["nearmiss"])
    ctrl = finish(controls(val, bad, CONTROL_N, seed=3), word)

    for name, rows in [("train_pool", pool), ("eval", ev), ("competitor", comp), ("nearmiss", near), ("control", ctrl)]:
        write_jsonl(d / f"{name}.jsonl", rows)

    meta = dict(word=word, **spec, counts=dict(
        train_pool=len(pool), eval=len(ev),
        eval_val=sum(r["source"] == "val" for r in ev),
        eval_frame_unseen=sum(not r["frame_seen_train200"] for r in ev),
        competitor={c: sum(r["slot"] == c for r in comp) for c in spec["competitors"]},
        nearmiss={c: sum(r["slot"] == c for r in near) for c in spec["nearmiss"]},
        control=len(ctrl)),
        windows=dict(left_sents=N_LEFT_SENTS, right_sents=N_RIGHT_SENTS),
        source="roneneldan/TinyStories (train shards 0-3, validation)")
    if tok is not None:
        meta["gptneo_tokens"] = {p: tok.encode(" " + p).tokens for p in spec["phrases"]}
        if spec["gold"]: meta["gptneo_gold_id"] = tok.encode(spec["gold"]).ids
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(word, meta["counts"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw_tinystories")
    ap.add_argument("--out", default="data/words")
    ap.add_argument("--words", default=",".join(WORDS))
    ap.add_argument("--tokenizer", default=None, help="path to GPT-Neo tokenizer.json (optional)")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    fetch(raw)
    train, val = read(raw, "train"), read(raw, "val")
    tok = None
    if a.tokenizer:
        from tokenizers import Tokenizer; tok = Tokenizer.from_file(a.tokenizer)
    for w in a.words.split(","):
        build(w, WORDS[w], train, val, out, tok)


if __name__ == "__main__":
    main()
