"""
Preprocessing pipeline for the Paderborn University (PU) Bearing DataCenter dataset.

What this script does
----------------------
1. Recursively scans a directory for the official ``.mat`` files
   (e.g. ``N15_M07_F10_KA04_1.mat``) and parses bearing code / operating
   condition / run number straight from the filename (works no matter how
   you've organised the folders after download).
2. Assigns a class label to every file based on the bearing code, using a
   configurable labeling scheme (see ``LABEL_SCHEMES`` below).
3. Splits the dataset at the *file* (or *bearing*) level, stratified by
   class, BEFORE any windowing happens. This is the key anti-leakage step:
   every window generated from a given recording (and, if you choose
   bearing-level grouping, every recording from a given physical bearing)
   ends up entirely inside a single split.
4. Segments the raw vibration signal of each file into fixed-length windows
   (with overlap allowed for the training split only, by default) and saves
   the result as train/val/test ``.npz`` files ready to be consumed by
   ``dataset.py``.

Run ``python preprocessing.py --help`` for all options, or
``python preprocessing.py --inspect path/to/one_file.mat`` to sanity-check
that the internal .mat structure matches what this script expects before
you run it on the whole dataset.

Dataset background / label schemes
-----------------------------------
The PU dataset has 32 bearings: 6 healthy, 12 with artificially induced
damage (EDM / drilling / engraving) and 14 with real damage from
accelerated lifetime tests. Three of those (KB23, KB24, KB27) have BOTH
inner- and outer-race damage simultaneously.

    HEALTHY        : K001, K002, K003, K004, K005, K006
    OR (artificial): KA01, KA03, KA05, KA06, KA07, KA08, KA09
    OR (real)      : KA04, KA15, KA16, KA22, KA30
    IR (artificial): KI01, KI03, KI05, KI07, KI08
    IR (real)      : KI04, KI14, KI16, KI17, KI18, KI21
    Combined (real): KB23, KB24, KB27   (both IR + OR damage)

Not sure how many classes to use? See the README for the full discussion.
The default here, ``location3``, is the scheme most commonly used in the
CNN-on-PU literature: Healthy / Inner-race / Outer-race (3 classes, 29 of
the 32 bearings, the 3 combined-damage bearings excluded because they
don't cleanly belong to either fault class).
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from sklearn.model_selection import StratifiedGroupKFold

SAMPLING_RATE_HZ = 64_000

# ---------------------------------------------------------------------------
# Bearing code -> condition groupings
# ---------------------------------------------------------------------------
HEALTHY = {"K001", "K002", "K003", "K004", "K005", "K006"}
OR_ARTIFICIAL = {"KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09"}
OR_REAL = {"KA04", "KA15", "KA16", "KA22", "KA30"}
IR_ARTIFICIAL = {"KI01", "KI03", "KI05", "KI07", "KI08"}
IR_REAL = {"KI04", "KI14", "KI16", "KI17", "KI18", "KI21"}
COMBINED_REAL = {"KB23", "KB24", "KB27"}  # both IR and OR damage

ALL_KNOWN_CODES = HEALTHY | OR_ARTIFICIAL | OR_REAL | IR_ARTIFICIAL | IR_REAL | COMBINED_REAL


def LABEL_SCHEMES():
    """
    Returns a dict of scheme_name -> {bearing_code: class_name}.
    Bearing codes not present in the mapping are excluded from that scheme.
    """
    location3 = {}
    location3.update({c: "Healthy" for c in HEALTHY})
    location3.update({c: "IR" for c in (IR_ARTIFICIAL | IR_REAL)})
    location3.update({c: "OR" for c in (OR_ARTIFICIAL | OR_REAL)})
    # COMBINED_REAL deliberately excluded

    location4 = dict(location3)
    location4.update({c: "Combined" for c in COMBINED_REAL})

    healthy_vs_faulty = {}
    healthy_vs_faulty.update({c: "Healthy" for c in HEALTHY})
    healthy_vs_faulty.update({c: "Faulty" for c in ALL_KNOWN_CODES - HEALTHY})

    origin_location_6class = {}
    origin_location_6class.update({c: "Healthy" for c in HEALTHY})
    origin_location_6class.update({c: "IR_artificial" for c in IR_ARTIFICIAL})
    origin_location_6class.update({c: "IR_real" for c in IR_REAL})
    origin_location_6class.update({c: "OR_artificial" for c in OR_ARTIFICIAL})
    origin_location_6class.update({c: "OR_real" for c in OR_REAL})
    origin_location_6class.update({c: "Combined_real" for c in COMBINED_REAL})

    bearing32 = {c: c for c in ALL_KNOWN_CODES}

    return {
        "location3": location3,                    # Healthy / IR / OR            (recommended default)
        "location4": location4,                     # + Combined
        "healthy_vs_faulty": healthy_vs_faulty,      # binary
        "origin_location_6class": origin_location_6class,  # location x real-vs-artificial
        "bearing32": bearing32,                      # one class per physical bearing (not recommended for generalization)
    }


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
FNAME_RE = re.compile(
    r"^(?P<condition>N\d{2}_M\d{2}_F\d{2})_(?P<bearing>[A-Za-z]+\d+)_(?P<run>\d+)$"
)


def parse_filename(path: Path):
    stem = path.stem
    m = FNAME_RE.match(stem)
    if m is None:
        # Fallback: generic split, in case of minor naming variations.
        tokens = stem.split("_")
        if len(tokens) < 5:
            return None
        return {
            "condition": "_".join(tokens[:3]),
            "bearing_code": tokens[3],
            "run": tokens[4],
        }
    d = m.groupdict()
    return {"condition": d["condition"], "bearing_code": d["bearing"], "run": int(d["run"])}


def scan_directory(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*.mat")):
        parsed = parse_filename(path)
        if parsed is None:
            warnings.warn(f"Could not parse filename, skipping: {path.name}")
            continue
        rows.append({"path": str(path), **parsed})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No .mat files found under {root}")
    return df


# ---------------------------------------------------------------------------
# .mat loading
# ---------------------------------------------------------------------------
def inspect_mat_file(path: Path):
    """Print the structure of one .mat file so you can confirm field names
    before running the full pipeline. The PU files are a MATLAB struct
    named after the file itself, with a 'Y' field holding one sub-struct
    per channel (Force, Phase_Current_1, Phase_Current_2, Speed,
    Temp_2_Bearing_Module, Torque, Vibration_1), each with 'Name' and
    'Data' fields.
    """
    mat = scipy.io.loadmat(str(path), simplify_cells=True)
    top_keys = [k for k in mat.keys() if not k.startswith("__")]
    print(f"Top-level keys: {top_keys}")
    key = top_keys[0]
    struct = mat[key]
    print(f"Fields of '{key}': {list(struct.keys()) if isinstance(struct, dict) else type(struct)}")
    if isinstance(struct, dict) and "Y" in struct:
        channels = struct["Y"]
        if isinstance(channels, dict):
            channels = [channels]
        for ch in channels:
            name = ch.get("Name", "?")
            data = np.asarray(ch.get("Data", [])).squeeze()
            print(f"  channel '{name}': shape={data.shape}, dtype={data.dtype if data.size else 'n/a'}")


def load_vibration_signal(path: Path):
    """Load the vibration channel from one PU .mat file.

    Returns:
        np.ndarray: 1D float32 vibration signal
        None: if the .mat file is empty, corrupted, or has no vibration channel
    """
    try:
        mat = scipy.io.loadmat(str(path), simplify_cells=True)
    except Exception as e:
        warnings.warn(
            f"Could not load .mat file, skipping: {path.name} "
            f"({type(e).__name__}: {e})"
        )
        return None

    try:
        top_keys = [k for k in mat.keys() if not k.startswith("__")]

        if not top_keys:
            warnings.warn(f"Empty .mat structure, skipping: {path.name}")
            return None

        struct = mat[top_keys[0]]

        if not isinstance(struct, dict) or "Y" not in struct:
            warnings.warn(f"No valid 'Y' structure, skipping: {path.name}")
            return None

        channels = struct["Y"]

        if isinstance(channels, dict):
            channels = [channels]

        for ch in channels:
            name = str(ch.get("Name", "")).lower()

            if "vibration" in name:
                data = np.asarray(
                    ch.get("Data", []),
                    dtype=np.float32
                ).squeeze()

                if data.size == 0:
                    warnings.warn(
                        f"Empty vibration signal, skipping: {path.name}"
                    )
                    return None

                return data

        warnings.warn(
            f"No vibration channel found, skipping: {path.name}"
        )
        return None

    except Exception as e:
        warnings.warn(
            f"Invalid .mat structure, skipping: {path.name} "
            f"({type(e).__name__}: {e})"
        )
        return None

# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def segment_signal(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    n = len(signal)
    if n < window_size:
        return np.empty((0, window_size), dtype=np.float32)
    starts = range(0, n - window_size + 1, stride)
    windows = np.stack([signal[s:s + window_size] for s in starts]).astype(np.float32)
    return windows


def normalize_windows(windows: np.ndarray, mode: str, train_mean=None, train_std=None):
    if mode == "none":
        return windows, train_mean, train_std
    if mode == "per_window":
        mu = windows.mean(axis=1, keepdims=True)
        sd = windows.std(axis=1, keepdims=True) + 1e-8
        return (windows - mu) / sd, None, None
    if mode == "global":
        if train_mean is None or train_std is None:
            train_mean = float(windows.mean())
            train_std = float(windows.std() + 1e-8)
        return (windows - train_mean) / train_std, train_mean, train_std
    raise ValueError(f"Unknown normalize mode: {mode}")


# ---------------------------------------------------------------------------
# Leakage-safe split
# ---------------------------------------------------------------------------
def stratified_group_split(meta: pd.DataFrame, group_col: str, label_col: str,
                            n_splits_outer: int, n_splits_inner: int, seed: int):
    for cls, count in meta.groupby(label_col)[group_col].nunique().items():
        if count < n_splits_outer:
            warnings.warn(
                f"Class '{cls}' only has {count} distinct groups (by '{group_col}') "
                f"but n_splits_outer={n_splits_outer}. Consider lowering --n-splits-outer "
                f"or using --group-by file instead of bearing for this class."
            )

    sgkf_outer = StratifiedGroupKFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf_outer.split(meta, meta[label_col], meta[group_col]))
    train_val = meta.iloc[train_val_idx].reset_index(drop=True)
    test = meta.iloc[test_idx].reset_index(drop=True)

    sgkf_inner = StratifiedGroupKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf_inner.split(train_val, train_val[label_col], train_val[group_col]))
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)

    # Explicit leakage assertions: no group should ever appear in two splits.
    g_train, g_val, g_test = set(train[group_col]), set(val[group_col]), set(test[group_col])
    assert g_train.isdisjoint(g_val), "Leakage: groups shared between train and val!"
    assert g_train.isdisjoint(g_test), "Leakage: groups shared between train and test!"
    assert g_val.isdisjoint(g_test), "Leakage: groups shared between val and test!"

    return train, val, test


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_windows_for_split(meta_split: pd.DataFrame, window_size: int, stride: int,
                             label_to_idx: dict):
    X_list, y_list, bearing_list, condition_list = [], [], [], []

    for _, row in meta_split.iterrows():
        signal = load_vibration_signal(Path(row["path"]))

        # Skip empty/corrupted/invalid .mat files
        if signal is None:
            continue

        windows = segment_signal(signal, window_size, stride)

        if windows.shape[0] == 0:
            warnings.warn(
                f"File shorter than window_size, skipped: {row['path']}"
            )
            continue

        X_list.append(windows)

        y_list.append(
            np.full(
                windows.shape[0],
                label_to_idx[row["label"]],
                dtype=np.int64
            )
        )

        bearing_list.append(
            np.full(
                windows.shape[0],
                row["bearing_code"],
                dtype=object
            )
        )

        condition_list.append(
            np.full(
                windows.shape[0],
                row["condition"],
                dtype=object
            )
        )

    X = (
        np.concatenate(X_list, axis=0)
        if X_list
        else np.empty((0, window_size), dtype=np.float32)
    )

    y = (
        np.concatenate(y_list, axis=0)
        if y_list
        else np.empty((0,), dtype=np.int64)
    )

    bearings = (
        np.concatenate(bearing_list, axis=0)
        if bearing_list
        else np.empty((0,), dtype=object)
    )

    conditions = (
        np.concatenate(condition_list, axis=0)
        if condition_list
        else np.empty((0,), dtype=object)
    )

    return X, y, bearings, conditions

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=str, help="Root directory containing the downloaded .mat files (any nesting).")
    parser.add_argument("--out-dir", type=str, default="./processed", help="Where to write train/val/test .npz files.")
    parser.add_argument("--scheme", type=str, default="location3", choices=list(LABEL_SCHEMES().keys()))
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--stride-train", type=int, default=2048, help="Stride for training windows (< window-size means overlap).")
    parser.add_argument("--stride-eval", type=int, default=4096, help="Stride for val/test windows (non-overlapping by default).")
    parser.add_argument("--group-by", type=str, default="bearing", choices=["bearing", "file"],
                         help="'bearing' = strict, splits by physical bearing (recommended). "
                              "'file' = splits by individual recording, allows different runs of the "
                              "same physical bearing to land in different splits (weaker leakage guarantee).")
    parser.add_argument("--n-splits-outer", type=int, default=5, help="Controls test fraction (~1/n).")
    parser.add_argument("--n-splits-inner", type=int, default=4, help="Controls val fraction of the remaining train+val (~1/n).")
    parser.add_argument("--normalize", type=str, default="per_window", choices=["per_window", "global", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect", type=str, default=None, help="Path to a single .mat file to inspect instead of running the pipeline.")
    args = parser.parse_args()

    if args.inspect:
        inspect_mat_file(Path(args.inspect))
        return

    if not args.data_root:
        parser.error("--data-root is required unless --inspect is used")

    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {root} for .mat files ...")
    meta = scan_directory(root)
    print(f"Found {len(meta)} files across {meta['bearing_code'].nunique()} bearing codes.")

    label_map = LABEL_SCHEMES()[args.scheme]
    meta["label"] = meta["bearing_code"].map(label_map)
    n_before = len(meta)
    meta = meta.dropna(subset=["label"]).reset_index(drop=True)
    if n_before != len(meta):
        print(f"Dropped {n_before - len(meta)} files whose bearing code isn't part of scheme '{args.scheme}'.")

    print("\nClass distribution (by file):")
    print(meta["label"].value_counts())

    classes = sorted(meta["label"].unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}

    group_col = "bearing_code" if args.group_by == "bearing" else "path"
    train_meta, val_meta, test_meta = stratified_group_split(
        meta, group_col=group_col, label_col="label",
        n_splits_outer=args.n_splits_outer, n_splits_inner=args.n_splits_inner, seed=args.seed,
    )
    print(f"\nSplit sizes (files): train={len(train_meta)}, val={len(val_meta)}, test={len(test_meta)}")
    for name, split in [("train", train_meta), ("val", val_meta), ("test", test_meta)]:
        print(f"  {name}: {split['label'].value_counts().to_dict()}")

    print("\nBuilding windows ...")
    X_train, y_train, bear_train, cond_train = build_windows_for_split(
        train_meta, args.window_size, args.stride_train, label_to_idx)
    X_val, y_val, bear_val, cond_val = build_windows_for_split(
        val_meta, args.window_size, args.stride_eval, label_to_idx)
    X_test, y_test, bear_test, cond_test = build_windows_for_split(
        test_meta, args.window_size, args.stride_eval, label_to_idx)

    X_train, mean_, std_ = normalize_windows(X_train, args.normalize)
    X_val, _, _ = normalize_windows(X_val, args.normalize, mean_, std_)
    X_test, _, _ = normalize_windows(X_test, args.normalize, mean_, std_)

    # add channel dim: (N, window_size) -> (N, 1, window_size)
    X_train = X_train[:, None, :]
    X_val = X_val[:, None, :]
    X_test = X_test[:, None, :]

    print(f"\nWindow counts: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")

    np.savez_compressed(out_dir / "train.npz", X=X_train, y=y_train, bearing=bear_train, condition=cond_train)
    np.savez_compressed(out_dir / "val.npz", X=X_val, y=y_val, bearing=bear_val, condition=cond_val)
    np.savez_compressed(out_dir / "test.npz", X=X_test, y=y_test, bearing=bear_test, condition=cond_test)

    meta_out = {
        "scheme": args.scheme,
        "classes": classes,
        "label_to_idx": label_to_idx,
        "window_size": args.window_size,
        "stride_train": args.stride_train,
        "stride_eval": args.stride_eval,
        "group_by": args.group_by,
        "normalize": args.normalize,
        "global_mean": mean_,
        "global_std": std_,
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "seed": args.seed,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nSaved processed data + meta.json to {out_dir}")


if __name__ == "__main__":
    main()