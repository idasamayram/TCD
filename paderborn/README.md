# Paderborn Bearing Dataset — CNN1D pipeline (Zennit-LRP friendly)

## 1. How many classes should you use?

The 32 PU bearings break down like this:

| Group | Bearing codes | Count |
|---|---|---|
| Healthy | K001–K006 | 6 |
| OR damage, artificial | KA01,03,05,06,07,08,09 | 7 |
| OR damage, real | KA04,15,16,22,30 | 5 |
| IR damage, artificial | KI01,03,05,07,08 | 5 |
| IR damage, real | KI04,14,16,17,18,21 | 6 |
| Combined (IR+OR), real | KB23,24,27 | 3 |

`preprocessing.py` ships with several ready-made label schemes (`--scheme`), so you can switch with one flag instead of rewriting code:

- **`location3` (default, recommended)** — Healthy / IR / OR, 3 classes, 29 bearings (drops the 3 combined-damage bearings, which don't cleanly belong to either fault). This is the scheme used by most CNN-on-PU papers and is a good default if you just want "can the model tell where the fault is."
- **`location4`** — adds a 4th "Combined" class for KB23/24/27.
- **`healthy_vs_faulty`** — binary, easiest task, least informative.
- **`origin_location_6class`** — location × (real vs. artificial damage), 6 classes. Use this if you also care about damage severity/origin, not just location — real damage (natural pitting) looks meaningfully different from artificial (EDM/drilled) damage.
- **`bearing32`** — one class per physical bearing. Not recommended for a generalization-focused study — with only one or two bearings per class you can't leave a bearing out for testing, so this scheme all but forces file-level (not bearing-level) splitting and risks leaking the specific defect's signature into your accuracy.

**Recommendation:** start with `location3`. If accuracy comes out unrealistically high (>99%), you're likely leaking bearing identity — that's what the splitting section below addresses.

## 2. Avoiding data leakage

There are two distinct leakage risks in this dataset:

1. **Window-level leakage**: if you segment a 4-second recording into overlapping windows and then randomly shuffle windows into train/val/test, near-duplicate windows end up on both sides of the split.
2. **Bearing-identity leakage** (the sneakier one): even with non-overlapping windows, if different *runs* of the same physical bearing (different operating conditions, different run numbers) are split across train and test, the model can learn to recognize that specific bearing's individual defect signature rather than the fault type in general. This inflates test accuracy in a way that won't generalize to a new bearing.

`preprocessing.py` solves both by splitting **at the file level, before windowing**, using `StratifiedGroupKFold`:

- `--group-by bearing` (default): every file from the same physical bearing stays entirely in one split. This is the rigorous choice — it tests whether the model generalizes to bearings it has never seen, which is the realistic deployment scenario. The trade-off is that some classes only have 5–7 physical bearings, so the split can't be perfectly proportioned; the script warns you if a class has fewer bearings than `--n-splits-outer`.
- `--group-by file`: groups by individual recording instead of bearing. Looser (same bearing's other runs may appear in a different split), but works fine if you don't have enough bearings per class for `bearing`-level grouping to be practical, or if within-bearing generalization is an acceptable evaluation goal for your use case.

After splitting, the script asserts that no bearing (or file) appears in more than one split, and windows are generated independently per split — so nothing computed on train ever touches val/test, including normalization statistics if you use `--normalize global`.

## 3. Pipeline

```
pu_bearing_cnn/
  preprocessing.py   # scan .mat files -> label -> leakage-safe split -> window -> save .npz
  dataset.py          # PyTorch Dataset over the saved .npz files
  model.py             # CNN1D, no BatchNorm/GroupNorm, LRP-friendly
  train.py             # training loop, early stopping, test-set report
  explain_lrp.py       # example: zennit LRP on a trained model
```

### Step 0 — sanity-check the .mat structure

Before running on the whole dataset, check that field names match what the script expects (they're standard for the official PU download, but confirm once):

```bash
python preprocessing.py --inspect /path/to/one/N15_M07_F10_K001_1.mat
```

If the vibration channel isn't found, this prints all channel names in the file so you can see what's different.

### Step 1 — preprocess

```bash
python preprocessing.py \
    --data-root /path/to/paderborn_dataset \
    --out-dir ./processed \
    --scheme location3 \
    --window-size 4096 \
    --stride-train 2048 \
    --stride-eval 4096 \
    --group-by bearing \
    --normalize per_window
```

This writes `train.npz`, `val.npz`, `test.npz`, and `meta.json` to `./processed`.

Notes on the defaults:
- `--window-size 4096` at 64 kHz ≈ 64 ms per window — enough to capture several shaft rotations at typical PU test speeds (900–1500 rpm).
- Training windows overlap (`stride-train` < `window-size`) for more training data; val/test windows don't overlap, so each evaluation window is an independent segment of signal.
- `--normalize per_window` z-scores each window independently using only its own mean/std — this can't leak information across samples. `--normalize global` instead fits mean/std on the training set only and applies it everywhere, which is also leakage-safe but preserves cross-window amplitude differences (useful if you care about severity, not just location).

### Step 2 — train

```bash
python train.py --data-dir ./processed --out-dir ./runs/exp1 --epochs 50
```

Trains with class-weighted cross-entropy (handles the mild class imbalance), Adam + `ReduceLROnPlateau`, and early stopping on val loss. At the end it reloads the best checkpoint and reports precision/recall/F1 and a confusion matrix on the held-out test set. Outputs: `best_model.pt`, `history.json`, `test_metrics.json`, `run_config.json`.

### Step 3 — explain with Zennit LRP

```bash
pip install zennit
python explain_lrp.py --run-dir ./runs/exp1 --data-dir ./processed --index 0
```

Why the model is LRP-friendly by construction:
- No BatchNorm / GroupNorm / LayerNorm anywhere — these require canonizers (e.g. `SequentialMergeBatchNorm`) to attribute correctly; simplest to just not use them, as you asked.
- No skip/residual connections — additions of two branches need explicit rule handling; a plain feed-forward stack sidesteps that.
- `ReLU(inplace=False)` everywhere, since in-place ops can overwrite activations that LRP hooks need to read on the backward pass.
- Only `Conv1d`, `Linear`, `MaxPool1d`, `AdaptiveAvgPool1d`, `Flatten`, `Dropout` — all natively recognized by zennit's built-in composites (`Conv1d`/`AdaptiveAvgPool1d` verified to match `zennit.types.Convolution` / `AvgPool` in the installed version).
- `explain_lrp.py` uses `EpsilonPlusFlat` (Epsilon in the dense layers, ZPlus in the conv layers, Flat in the very first layer). Flat is the right choice for the first-layer rule here because our input is z-scored, unbounded real-valued signal — not pixel data in `[0, 1]` — so the box-constrained composites made for images (e.g. `EpsilonGammaBox`) don't apply directly.

## 4. Extending this

- To try a different class scheme, just change `--scheme` — no code changes needed unless you want a scheme not already in `LABEL_SCHEMES()` in `preprocessing.py`, in which case add a new dict there.
- To add cross-condition evaluation (e.g. train on N15_M07_F10, test on N09_M07_F10) instead of / in addition to the leakage-safe split, filter `meta` by the `condition` column before calling `stratified_group_split`.
- The model's `base_channels` and `dropout` are exposed via `train.py` CLI flags if you want to scale the network up or down.
