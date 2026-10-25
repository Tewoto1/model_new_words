# model_new_words

```
model.py                 Model: load, embeddings (E, add_row, set_row), encode {w}, pad,
                         internals (capture h / residuals, states_at_last, slot_states),
                         geometry in embedding space (neighbors, project, span_fraction)
train.py                 classical SFT loop: make_examples, collate, lm_loss, setup_params, train
dataset.py               load_word / word_meta for data/words/
experiments/neologism.py inits, slot_metrics, equivalence, geometry, main (CFG at the top)
tests/test_core.py       checks for every function you implement (tiny random GPT-Neo, CPU)
scripts/build_word_data.py   TinyStories -> data/words/<word>/ (seeded, ~4 min)
data/words/              built datasets (gitignored)
```

Functions with `raise NotImplementedError` are yours to write; their docstrings say what they must
return and give PyTorch hints. Suggested order (each step's tests unlock the next):

1. model.py: pad, add_row, capture, states_at_last, neighbors, project
2. train.py: collate, lm_loss, setup_params, train
3. experiments/neologism.py: slot_metrics, make_inits, equivalence

```
python tests/test_core.py            # PASS / FAIL / TODO per test
python tests/test_core.py capture    # only tests matching 'capture'
python -m experiments.neologism      # the real run (set DEVICE in CFG)
```
