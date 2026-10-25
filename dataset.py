"""Datasets. New-word datasets are built by scripts/build_word_data.py into data/words/<word>/:
  meta.json, train_pool.jsonl, eval.jsonl, competitor.jsonl, nearmiss.jsonl, control.jsonl
Row fields: id, source, story, left, phrase, right, text (with {w}), orig, frame, sent_initial,
  n_w, rank (train_pool), frame_seen_train200 (eval), slot (competitor/nearmiss).
"""
import json
from pathlib import Path

WORDS_DIR = "data/words"


def list_words(words_dir: str = WORDS_DIR) -> list[str]:
    return sorted(p.name for p in Path(words_dir).iterdir() if (p / "meta.json").exists())


def word_meta(word: str, words_dir: str = WORDS_DIR) -> dict:
    return json.loads((Path(words_dir) / word / "meta.json").read_text())


def load_word(word: str, split: str, n: int | None = None, words_dir: str = WORDS_DIR,
              slot: str | None = None, source: str | None = None) -> list[dict]:
    """split in {train_pool, eval, competitor, nearmiss, control}.
    train_pool is in seeded order, so load_word(w, 'train_pool', n=200) is the fixed train_200
    and smaller n are nested subsets (for data-scaling curves)."""
    rows = [json.loads(l) for l in open(Path(words_dir) / word / f"{split}.jsonl")]
    if slot is not None:
        rows = [r for r in rows if r.get("slot") == slot]
    if source is not None:
        rows = [r for r in rows if r["source"] == source]
    return rows[:n] if n else rows
