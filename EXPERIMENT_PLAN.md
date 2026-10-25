# Experiment plan: where does a new word go? (Qwen3-0.6B-Base)

For an agent running this on a rented GPU box. Written 2026-10-25 for Tony. Read all of it before touching the box.

---

## FILL IN BEFORE HANDOFF (Tony)

```
SSH:              ssh -p <PORT> <USER>@<HOST> -i <KEY_PATH>
GPU:              RTX 4080 16GB (see the 16 GB settings in S2.2)
HF_USER:          <your HF username or org>
HF_RESULTS_REPO:  <HF_USER>/model-new-words-runs      (private dataset repo; agent creates it if missing)
HF_TOKEN:         DO NOT write it in this file. Give it to the agent separately; it lives only in an env var on the box.
CODE_REPO:        https://github.com/Tewoto1/model_new_words   (push the 5 files in "Before handoff" first)
BUDGET:           no cap: run every tier, including Optional. Report spend as you go. Do not leave the GPU idle.
WEIGHT DELTAS:    none. Embedding change recorded in full for every touched row (section 6), plus hidden states.
E_DELTA_ALL_ROWS: no          (yes = also store all 151,937 rows exactly: +14 GB fp16)
END OF RUN:       destroy the box, after the final HF upload is verified
```

Agent: if any field above is still `<...>`, ask Tony before starting. Also ask before deleting anything on his Mac or HF, changing `data/words`, or changing the repo's existing functions (add new files instead).

### Before handoff (Tony)
Everything is in your repo working copy (`~/Desktop/Everything_Else/projs/model_new_words`), not committed yet. Commit and push:
```
cd ~/Desktop/Everything_Else/projs/model_new_words
git add EXPERIMENT_PLAN.md README.md scripts/word_list.py scripts/build_word_data.py data/experiments/words_manifest.sha256 notebooks/
git commit -m "word list v2 (116 words), experiment plan, sweep + internals notebooks"
git push
```
`data/words/` (491 MB, 116 words) is gitignored: the agent uploads it to `HF_RESULTS_REPO` under `inputs/data_words/` from the Mac (section 4, step 2).

## Key numbers and warnings (read before launching)

- **HF upload: ~7 GB total.** Embedding changes ~1.9 GB (45 full fine-tune runs x ~41 MB: every row the data touches stored exactly in fp32, the other ~148k rows as a rank-64 SVD + per-row norms + the reconstruction error); hidden states ~3.2 GB; everything else ~1.3 GB; input data 0.5 GB. No weight deltas. If you want every one of the 151,937 rows exact for those 45 runs instead, that is +14 GB (fp16) or +28 GB (fp32): set `E_DELTA_ALL_ROWS: yes`.
- **Box disk: >= 40 GB.** Upload after every stage, delete local copies only after the upload is verified.
- **GPU time: ~20-24 h on the RTX 4080** for all tiers + Optional (estimate, ~1.5x a 4090; the agent re-times after a smoke test). Tier 0 ~1.5 h, Tier 1 ~5 h, Tier 2 ~7 h, Tier 3 ~4.5 h, Optional ~4 h, uploads extra.
- **16 GB is enough for every stage, with S2.2 changed:** fp32 Adam on all 0.6B parameters needs >16 GB, so full fine-tune runs use 8-bit AdamW (`bitsandbytes`), gradient checkpointing and `BS=8, STEPS=600` (details in S2.2). Row-only runs fit as written in fp32.
- **What changes in which run:** in row-only runs (most runs) the ONLY weight that changes is the new word's row of `E` (tied: it is both the input embedding and the output vector). Hidden states before the new word stay bit-identical; states at and after it change because they read the new row. In full fine-tune runs every weight changes. Section 6 records both.
- **After the run: revoke the HF token** (the agent logs out and deletes it on the box first).

---

## 1. Questions this experiment answers

A new word = one new row `e_new` appended to the tied embedding matrix `E` (no tokenizer change). Each training sentence is a TinyStories sentence where a real phrase was replaced by the new token (e.g. "He was **very big**" -> "He was **NEW**").

- **Q0. Before learning.** At the place where the new word goes, what does the model already expect? Read the model's state at that position out through `E`: which existing tokens does it point to? Does it follow the expected semantics: synonym slots -> the synonym; antonyms ranked below synonyms; a blend ("tired but happy") -> between its two parts; an idiom -> its meaning ("suddenly"), not its parts ("all", "sudden")?
- **Q1. After learning: stick near old words, or make a new space?** After training (the model now predicts NEW in those slots), does `e_new` sit close to existing rows of `E` (a near-synonym of old words), or does it move into a direction no old word occupies? If the model's body is allowed to change, do the hidden states that predict NEW move as well, towards a new direction? Which regimes (word type, how much the body may change, amount of data, init) give which outcome?
- **Q2. Does the row move much from its init?** The default init is the mean hidden state at the slot (the vector the model already "points with" there), rescaled to a normal row norm. Does training move it far? From a random init, does it end up near the semantic init (same answer), or somewhere else that works just as well?
- **Q3. Is the location estimate biased by story context?** Tony's worry: the contexts in TinyStories may make the estimated location noisy or biased. Test it by resampling contexts, controlling for the preceding word, comparing against slots of same-meaning words, and swapping the new word into other contexts after training.

---

## 2. What already exists

Code root = the GitHub repo root (`model_new_words/`; on Tony's Mac: `~/Desktop/Everything_Else/projs/model_new_words`). Always run from the code root: `data/words`, `outputs/` are relative to it.

| file | what to use from it |
|---|---|
| `model.py` | `Model.load(name, device, dtype)`, `.E`, `.add_row()`, `.set_row()`, `.encode_pair(row, new_id)`, `.pad`, `.capture(layers)`, `.states_at_last(seqs, bs, layers)`, `.slot_states(rows)`, `.active_ids(rows)`, `.neighbors(v, k, active, mu, upto)` (centered cosine), `.project`, `.span_fraction` |
| `train.py` | `TrainConfig`, `train(m, rows, new_id, init, cfg, logger)`. `lr_rest=0`: only `e_new` trains. `lr_rest>0`: everything trains (old rows of E at `lr_rest` via the pull-back trick, body at `lr_rest`, new row at `lr_new`). |
| `experiments/neologism.py` | `make_inits` (mean_emb / mean_h / in_span / gold; mean_h and in_span are rescaled to the mean active-row norm), `slot_metrics`, `equivalence` (KL of the next tokens, NEW vs real phrase), `geometry` |
| `notebooks/neologism_sweep.ipynb` | reference implementations of extra metrics: `kl_drop` / `carried`, `slot_rep_gap`, per-slot-word competitor sampling (`per_slot`), `MIN_SLOT_ROWS`. Reuse that code, don't re-derive it. |
| `tests/test_core.py` | 16 property tests on a tiny random GPT-Neo (CPU). Must pass before and after your changes. |
| `data/words/<word>/` | 116 words, 9 types. `meta.json` (phrases, competitors, nearmiss, gold, type, status, counts), `train_pool.jsonl` (seeded order: first N rows = train_N), `eval.jsonl` (held out), `competitor.jsonl` / `nearmiss.jsonl` (validation slots where a same-meaning word / a phrase part or antonym fills the slot; `slot` names it), `control.jsonl` (random slots). |

Word types: `syn` (one-word synonym; calibration), `comp` ("very big"), `idiom` ("all of a sudden"), `cmp` (compound noun, "fire truck"), `act` (verb phrase, "fell asleep"), `tmp` ("the next day"), `spat` ("on top of"), `blend` ("tired but happy"), `abs` ("learned a lesson"). The original 6 words have no `type` field; map `rung` -> type as in the sweep notebook (`RUNG2TYPE`).

Verified on 2026-10-25 (CPU): `Model.load("Qwen/Qwen3-0.6B-Base")` works with this code: tied embeddings, `E` is 151936 x 1024 (151669 real tokens; the rest are padding rows that never appear in `active_ids`), 28 blocks, pad token `<|endoftext|>`. `add_row` gives `new_id = 151936`. Competitor words checked so far are single tokens in Qwen's tokenizer, but always re-check with Qwen's tokenizer at runtime (`len(tok.encode(" " + w)) == 1`), because `meta.json` single-token checks were done with GPT-Neo's.

Known issues to handle:
- `experiments/neologism.py: main()` takes `load_word(w, "competitor")[:N]`, which only sees the first competitor word (files are grouped by `slot`). Use `per_slot` from the sweep notebook.
- Memory: every forward returns full logits `[B, L, 151936]`. `states_at_last` defaults to `bs=64` and `equivalence` to `bs=32`; on Qwen use `bs=8..16` or you will OOM on 16-24 GB.
- Keep `dtype=float32` for row-only runs: a bf16 `E` rounds away small updates to `e_new`.
- The `mean_h` init (even rescaled) gives p(NEW) ~ 1.0 at slots before training, because the mean slot state shares a large common direction with all slot states. Record it; it is part of Q2, not a bug.
- In the smoke test, the nearest rows to `mean_h` for "very big" were ' a', ' the', ',', ' in', ' very': the slot state mostly encodes "a short function word or the phrase's first token comes next". That is why Q0 needs the phrase-end readout (S0.2) and not only the slot readout.

---

## 3. Two training modes (and LoRA)

- **Row-only** (`lr_rest = 0`): only `e_new` changes. Every other weight is bit-identical, so every hidden state *before* the new word, including the state that predicts it, is fixed. What can change is where `e_new` goes, plus states *after* the new word (since they read `e_new` as input). Storage per run: one 1024-number row plus its trajectory.
- **Full fine-tune** (`lr_rest > 0`): everything trains; the new row at `lr_new`, the rest at `lr_rest`. Now the state that predicts NEW can itself move, and old rows can move. This is the only mode where "the model makes a new space for the word" can happen on the hidden-state side. Storage: a full weight delta is ~1.2 GB in fp16 for Qwen 0.6B, so keep full deltas only for designated runs (section 6).
- **LoRA** (rank 16 on all attention + MLP linears, `peft`; `E` frozen except `e_new` via the same gradient-mask hook as row-only): the body may change, but only in a low-rank way, and old rows of E cannot. A middle regime for Q1. Implement it as a new function in a new file; do not change `train.py`.

---

## 4. Setup on the box

1. Check GPU: `nvidia-smi`. Use `tmux` for everything long.
2. Code + data:
   - Code: `git clone https://github.com/Tewoto1/model_new_words ~/model_new_words` (after the "Before handoff" push; check `scripts/word_list.py` exists).
   - Data, uploaded once from Tony's Mac: `huggingface-cli upload <HF_RESULTS_REPO> ~/Desktop/Everything_Else/projs/model_new_words/data/words inputs/data_words --repo-type dataset`. On the box: `huggingface-cli download <HF_RESULTS_REPO> --repo-type dataset --include "inputs/data_words/*" --local-dir /tmp/dl` and move `/tmp/dl/inputs/data_words` to `~/model_new_words/data/words`.
   - Verify: `cd ~/model_new_words && sha256sum -c --quiet data/experiments/words_manifest.sha256` must print nothing (696 files).
   - Fallback if the Mac is unreachable: `python scripts/build_word_data.py --raw raw_tinystories --out data/words` rebuilds all 116 words from TinyStories (seeded, ~30-40 min CPU, downloads ~1 GB). Then run the sha256 check; if files differ, tell Tony before using them.
   - Record `git rev-parse HEAD`, the manifest check result, and `pip freeze` into `outputs/word_study/inputs_manifest.json`.
3. Env: Python 3.11+, `torch` (cu12.x), `transformers>=4.51` (Qwen3), `safetensors`, `huggingface_hub`, `peft`, `bitsandbytes`, `pandas`, `matplotlib`, `scikit-learn`. Record `pip freeze` into the outputs.
4. HF login: `huggingface-cli login --token $HF_TOKEN` from an env var. Never write the token into a file in the repo, a log, a notebook, or the report.
5. Download `Qwen/Qwen3-0.6B-Base` once; set `HF_HOME` to a disk with space.
6. Run `python tests/test_core.py` (must be 16 PASS). Then a Qwen smoke test on `comp_very_big`: 20 train steps row-only, `bs=16`, check that `e_new` moves, p(NEW) on eval slots is finite, and memory use is under the GPU limit (`torch.cuda.max_memory_allocated`). Time one step and one evaluation; use those to rescale the time estimates below before committing to tiers.

---

## 5. Stages (run in tier order; each stage writes results and uploads before the next starts)

Word sets (all exist in `data/words`):
- **S9** (one per type): `syn_huge`, `comp_very_big`, `idiom_all_of_a_sudden`, `cmp_fire_truck`, `act_fell_asleep`, `temporal_the_next_day`, `spat_on_top_of`, `blend_tired_but_happy`, `abs_learned_a_lesson`
- **S27** (three per type): S9 plus `syn_scared`, `syn_quickly`, `comp_very_happy`, `comp_too_heavy`, `idiom_right_away`, `idiom_could_not_believe`, `cmp_teddy_bear`, `compound_ice_cream`, `act_shook_head`, `act_took_a_bite`, `tmp_after_a_while`, `tmp_for_a_long_time`, `spat_next_to`, `spat_far_away`, `blend_angry_and_sad`, `blend_dark_and_scary`, `abs_work_together`, `abs_tell_the_truth`
- **ALL**: every word with `status` in {keep, maybe} (116).

Default hyperparameters unless a stage says otherwise: `N_TRAIN=200`, `STEPS=300`, `BS=16`, `LOG_EVERY=25`, `seed=0`, eval sets: 200 eval rows, `per_slot` 100 rows per competitor / near-miss word (skip words with < 20 rows), 300 control rows, 64 rows for KL metrics. "State" means the pre-decode vector `h` (input of `lm_head`, `logits = h @ E.T`) unless a layer is named; also record the residual after blocks {7, 14, 21}.

Centered cosine everywhere: subtract `mu` = mean of the active rows (`make_inits` returns it). `active` = `m.active_ids(train + eval + control rows)` per word.

### Tier 0: no training (all 116 words; ~1 GPU-hour)

**S0.1 Slot readout (what the model expects where NEW will go).** For eval slots, states at the end of `left` (`m.slot_states`). Record per word:
- top-20 tokens of the mean predicted distribution `mean_i softmax(h_i E^T)`, and of `softmax(mean_h E^T)`;
- mean probability and mean rank of: the phrase's first token, every competitor, every near-miss, gold;
- top-20 centered-cos neighbours of `mean_h` among active rows;
- type-specific checks (save all numbers, report pass rates per type):
  - `syn`: P(gold) and P(competitors) > P(antonyms in nearmiss)? ranks too.
  - `blend` and `comp`: least-squares coefficients of `mean_h - mu` on the two component rows (blend: the two states, e.g. ' tired', ' happy'; comp: ' very' and the head word), `span_fraction` onto them, cos to each and to their midpoint. "Between" = both coefficients > 0.
  - all other types: does the best competitor outrank the best near-miss (meaning vs parts)?

**S0.2 Phrase-end readout (what the phrase means once it has been read).** For each eval row, the state at the phrase's last token in the real-phrase text (`ids_orig[:len(ids_orig) - n_right]` -> `states_at_last`, layers {7, 14, 21} + h). Do the same at the end of the filler in competitor slots and near-miss slots, and at a random token in control rows. Per layer: centered cos between the mean phrase-end state and the mean competitor-end state, mean near-miss-end state, mean control state, plus a per-row nearest-centroid accuracy (phrase rows vs competitor rows vs near-miss rows). This is the context-level "meaning location", free of the first-token bias in S0.1.

**S0.3 Is the location estimate stable? (Q3, part 1)**
- Bootstrap: split `train_pool` (first 2000 rows) into 10 disjoint halves-pairs; cos between the two halves' `mean_h`. Report median and 5th percentile.
- Frame control: restrict to slots whose previous word (`frame` field) also occurs as the frame of competitor slots; recompute `mean_h` for phrase and for competitors on those frames only. Compare cos(phrase, competitor) with and without the frame restriction. If it drops a lot, the "semantic" match was really a shared syntactic frame.
- Context length: recompute `mean_h` with `left` cut to the last sentence only, and to the last 8 tokens only. Cos to the full-context `mean_h`.
- Semantic vs context baseline: cos(`mean_h` phrase, `mean_h` competitor slots) vs cos(`mean_h` phrase, `mean_h` near-miss slots) vs cos(`mean_h` phrase, `mean_h` control slots).

### Tier 1: row-only training, all words (~3-4 GPU-hours)

**S1.0 Learning-rate check** (S9, ~10 min). `lr_new` in {1e-3, 5e-3, 2e-2}, `mean_h` init, 300 steps. Pick the largest rate where p(NEW) at eval slots plateaus by step 300 and p(NEW) on control slots does not rise above its init value. Use it for every later stage. Record the curves.

**S1.1 Main run** (ALL, `mean_h` init). Per run record (section 6) and compute, before (row = init) and after:
- usage: p(NEW) at eval / control slots, top-1 fraction, `lift_comp`, `lift_near` (as in the sweep notebook), p(NEW) per competitor and per near-miss word;
- function: `equivalence` KL(phrase || NEW) on the next 5 tokens, `kl_drop`, `carried = 1 - KL_new / KL_drop`;
- **Q1 geometry of the row**:
  - top-20 centered-cos neighbours; NN1 cos; `iso` = NN1 cos of `e_new` minus the median NN1 cos of the old active rows (each old row vs its own nearest other active row; compute once per word, cache);
  - `span_fraction` of `e_new - mu` onto its top-8 neighbours (minus mu), and onto the top-64 / top-256 principal directions of the centered active rows;
  - cos to gold, to the mean of competitor rows, to the mean of near-miss rows, to the phrase-token mean;
- **Q2 drift**: `||e_final - e_init|| / ||e_init||`, cos(e_final, e_init), norm ratio to the mean active-row norm;
- **downstream states**: per layer {7, 14, 21, h}, centered cos between the state at NEW and at the phrase end (matched minus shuffled, `slot_rep_gap`), after training only (before = init).

### Tier 2: inits (Q2) and body plasticity (Q1) (~4-5 GPU-hours)

**S2.1 Init study** (S27, row-only). Inits: `mean_h`, `in_span`, `mean_emb`, `gold` (only if single-token in Qwen), `rand` x3 seeds (Gaussian, rescaled to the mean active-row norm), `rand_out` x1 (Gaussian projected off the top-256 principal directions of the centered active rows, rescaled). Per word, beyond the S1.1 metrics:
- pairwise cos between the final rows of all inits; cos(final_rand, mean_h); cos(final_rand, final_mean_h);
- decomposition of the update `delta = e_final - e_init`: fraction of `||delta||^2` inside the span of the top-k principal directions (k = 16, 64, 256) of the states seen at training positions (slot states of the train rows and states at every position after NEW); fraction of the init's component outside that span that survives training;
- functional agreement between inits: mean KL between the next-token distributions of two trained rows on eval texts (same row positions), p(NEW) at eval slots. "Different place, same function" = high KL agreement (low KL) with low cos.

**S2.2 Body plasticity** (S9, N_TRAIN=200, `mean_h` init). Conditions: row-only (from S1.1), LoRA r=16 (lr 1e-4), full fine-tune with `lr_rest` in {1e-6, 1e-5, 1e-4}, and full fine-tune `lr_rest=1e-5` with `N_TRAIN=1000`. With the `use_orig` control below that is 5 full fine-tune runs per word, 45 in total. **16 GB settings for full fine-tune (the 4080):** fp32 weights, `bitsandbytes.optim.AdamW8bit` (weight_decay=0) in place of Adam, `m.hf.gradient_checkpointing_enable()`, `BS=8`, `STEPS=600` (same number of examples as 16 x 300). Implement it as a copy of `train()` in `experiments/word_study.py` with the optimizer swapped (keep the old-row pull-back trick exactly; it holds for any optimizer whose update is linear in lr, which AdamW8bit is). Check peak memory on the first run; record the settings. Also a `use_orig=True` control at `lr_rest=1e-5` (same sentences with the real phrase; tells you how much the body moves from fine-tuning alone). Per run, beyond S1.1:
- **state at the slot** (the one that predicts NEW): `dh = mean_h_after - mean_h_before` over eval slots; `||dh|| / ||mean_h_before||`; centered cos(dh, e_new - mu); fraction of `||dh||^2` outside the span of the top-256 principal directions of the old active rows ("moved into a new direction");
- **old rows**: per-row `||dE_i||` for active rows; top-20 moved rows (tokens), and their cos(dE_i, e_new - mu); do competitors / near-misses move towards or away from `e_new`?
- **row geometry**: all Q1 metrics from S1.1, computed against the *trained* old rows;
- **damage**: mean next-token loss on 300 control windows before vs after;
- **weights**: per-tensor `||dW|| / ||W||`, top-8 SVD of every 2-D weight delta (section 6).
- Regime summary table: one row per (word, condition) with `iso`, span fractions, `||dh||` and its out-of-span fraction, cos(dh, e_new), p(NEW) eval/control, `carried`, damage. Tentative labels (report the numbers, not only labels): **anchored** if `iso >= 0` and span_fraction(top-8 neighbours) >= 0.5; **new space** if `iso` < 10th percentile of old rows' NN1 distribution and the slot state moved with out-of-span fraction > 0.5; else **mixed**.

### Tier 3: data amount and counterfactuals (Q3, part 2) (~3 GPU-hours)

**S3.1 Data scaling** (S27, row-only, `mean_h`): `N_TRAIN` in {25, 100, 400, 1600} (cap at the word's pool size), seeds {0, 1} for N=100 and N=400 (different rows: shuffle `train_pool` with the seed, not the first N). Does the final row converge (cos to the N=1600 row), and how fast?

**S3.2 Swaps after training** (use the S1.1 trained rows, ALL words; no training). Put NEW into: competitor slots (rows already have `{w}` at the competitor), near-miss slots, control slots, and eval slots split by `frame_seen_train200` (seen vs unseen preceding word). For each: KL(original filler || NEW) on the next 5 tokens and p(NEW) at the slot. NEW should behave like competitors, unlike near-misses and controls, and generalize to unseen frames.

### Optional (run it: BUDGET has no cap)
- Repeat S1.1 with `seed` 1 on ALL (noise floor for every metric).

---

## 6. What to record (and where)

Local layout under `src/model_new_words/outputs/word_study/`, mirrored to HF `<user>/model-new-words-runs` under `word_study/`:

**Probe states (the hidden-state record).** Per word, fix a probe set once and save its token ids: 32 eval rows and 32 control rows (seeded). Positions: in eval rows, the position that predicts the slot (`slot-1`), the slot itself (NEW, or the phrase's last token in the real-phrase text), and the 5 positions after it; in control rows, 4 random positions (352 positions). Record residual streams from `output_hidden_states` (index 0 = embedding output, 28 = after the final norm, i.e. the `h` that projects onto `E`) at those positions, fp16.
- Layers {0, 7, 14, 21, 28} (3.6 MB per set) for base sets and row-only runs; all 29 layers (21 MB per set) for full fine-tune and LoRA runs.
- base (once per word): real-phrase text, and NEW text with the init row.
- after each run: row-only runs store only positions >= slot of the eval rows (the rest are provably unchanged; check it on one run per stage by recording everything and asserting equality). Full fine-tune and LoRA runs store the full set.

**Embedding record (in full).** Every run: `e_init`, `e_final`, `e_traj` (every 25 steps), fp32. In row-only and LoRA runs no other row of `E` changes (assert it, max |dE| over old rows must be exactly 0). Full fine-tune runs, `dE = E_after - E_before`:
- exact fp32 rows for every touched id: all tokens appearing in that run's train / eval / competitor / near-miss / control rows, plus the 2,000 rows with the largest `||dE_i||` (~5-8k rows, ~25 MB);
- every other row: rank-64 SVD of their `dE` (U fp16, S, V fp32, ~20 MB), plus `||dE_i||` for all 151,937 rows and the relative reconstruction error per row (so you know exactly what the low-rank part misses). These rows only get the output-softmax push, a sum of training hidden states, so a low-rank fit should be close; report the error;
- if `E_DELTA_ALL_ROWS = yes`: also the full `[151937, 1024]` dE (fp16, 311 MB);
- `E` trajectories of the 50 most-moved rows every 50 steps.
No other weight deltas are saved; for non-embedding weights keep only per-tensor norms and top-8 SVDs.

Layout under `outputs/word_study/` (code root), mirrored to HF `HF_RESULTS_REPO` under `word_study/`:

```
inputs_manifest.json        git commit, data manifest check, pip freeze, GPU, CUDA, torch/transformers versions
run_manifest.jsonl          one line per run: run_id, stage, word, mode, init, lr_new, lr_rest, N, seed, status, minutes, peak_mem
base/<word>/stage0.json     S0.1-S0.3 numbers
base/<word>/probe.json      probe rows (ids) and positions
base/<word>/probe_states_phrase.safetensors   fp16 [n_pos, n_layers, 1024], real-phrase text
base/<word>/probe_states_init.safetensors     fp16, NEW text with e_init (mean_h)
base/<word>/slot_states.safetensors   fp16: slot states of eval / competitor / nearmiss / control (layer 14 + h),
                                      phrase-end states, mean_h variants (bootstrap halves, frame-controlled, short-context)
base/ref.safetensors        once: E rows of every active id over all words (fp32) + ids; mu per word
runs/<run_id>/config.json, metrics.json, curve.json
runs/<run_id>/rows.safetensors          fp32: e_init, e_final, e_traj [n_logs, d]
runs/<run_id>/probe_states_after.safetensors   fp16 (see Probe states)
runs/<run_id>/E_delta.safetensors      full FT: exact fp32 rows for touched + top-2000 ids, rank-64 SVD of the rest, per-row ||dE_i|| and SVD error, top-50 row trajectories (~41 MB)
runs/<run_id>/delta_summary.json        full FT / LoRA: per-tensor ||dW||, ||dW||/||W||
runs/<run_id>/delta_svd.safetensors     full FT: top-8 U, S, V (fp16) per 2-D weight delta
runs/<run_id>/lora.safetensors          LoRA: adapter weights (~20 MB)
figures/*.png, tables/*.csv, REPORT.md
```

Expected size: embedding changes ~1.9 GB; probe states ~3.2 GB (base 0.8, row-only runs ~1.3, full FT / LoRA 1.1); slot states, weight-norm SVDs, rows, metrics ~1.3 GB; input data 0.5 GB. Total ~7 GB (+14 GB if `E_DELTA_ALL_ROWS = yes`).

Upload after every stage (`huggingface_hub.HfApi().upload_folder(..., repo_type="dataset", path_in_repo="word_study/...")`; large files go through LFS automatically), verify the files are listed on HF, then delete the local copies of large files (deltas, probe states) to keep disk free. HF is the copy the run can die without.

Make every run resumable: skip a run if its `metrics.json` exists. Write `run_manifest.jsonl` lines as runs start and finish so a crash leaves a trace.

---

## 7. Report (`REPORT.md`, also uploaded)

Answer Q0-Q3 directly, with numbers and figures, per word type. At least:
- Q0: per-type table of S0.1 checks (pass rates), a few readout examples per type (top tokens), S0.2 cos bars per layer (competitor vs near-miss vs control).
- Q1: scatter of `iso` vs span_fraction(top-8) for every S1.1 word (colour = type); S2.2 regime table and a plot of `||dh||` out-of-span fraction vs `lr_rest`; top moved old rows for 3 example words.
- Q2: drift histogram per type; for S27, cos(final_rand, final_mean_h) vs functional agreement; the update-decomposition fractions.
- Q3: S0.3 stability numbers; S3.1 convergence curves; S3.2 swap KLs (competitor < near-miss < control?), seen vs unseen frames.
- A short "surprises and caveats" section, and wall-clock + cost per stage.

Plots: one dot per word with the per-type median marked; for distributions show the distribution (histogram / violin / ECDF), not mean +- band.

---

## 8. Rules

- Ask Tony before anything irreversible or outside this plan. Never delete files on his Mac. On the box, delete only what you created.
- Token hygiene: the HF token lives only in an env var and `~/.cache/huggingface/token` on the box. At the end run `huggingface-cli logout` and remove the token file; tell Tony to revoke the token.
- Do not modify `model.py`, `train.py`, `dataset.py`, `experiments/neologism.py` or `data/words`. Put new code in `experiments/word_study.py` (+ `experiments/word_study_lora.py`), CLI style `python -m experiments.word_study --stage S1.1 --words ALL`. Run `tests/test_core.py` again after writing it.
- Fixed seeds everywhere; record them.
- Check every metric on one word by hand before running it on all words (for example: row-only must leave every `E[i]`, i < new_id, bit-identical; `equivalence` with `e_new = E[gold]` on a single-token synonym gives KL ~ 0).
- At the end: final upload, verify the HF repo lists every stage and file count matches the local manifest, copy `REPORT.md` and `run_manifest.jsonl` back into Tony's repo on his Mac (`~/Desktop/Everything_Else/projs/model_new_words/outputs/word_study/`, gitignored), log out of HF and delete the token file, then **destroy the box** (END OF RUN). Tell Tony the final cost and to revoke the token.
