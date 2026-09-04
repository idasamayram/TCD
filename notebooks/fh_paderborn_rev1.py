
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paderborn_cnn1d.py
==================
Fault classification on the Paderborn University (KAt-DataCenter) rolling
bearing dataset with a 1D CNN in PyTorch.

Pipeline
--------
1. Walk a data root, find every ``*.mat`` measurement and parse the file name
   ``N15_M07_F10_KA01_1.mat`` -> operating condition / bearing code / trial.
2. Pull the requested channel(s) out of the MATLAB struct
   (``vibration_1`` @ 64 kHz by default; currents also available).
3. Cut every record into fixed-length windows -> arrays of shape (N, C, L)
   and cache them in one ``.npz`` (the .mat parsing is the slow part).
4. Split **by measurement file** (or **by bearing**), never randomly over
   windows -- windows from the same record are almost duplicates, a random
   split gives you fake 99.9 % accuracy.
5. Train a 1D CNN (wide first kernel + BN + GAP/GMP head), report window-level
   and measurement-level (majority vote) metrics + confusion matrix.

Data layout (nesting does not matter, only file names do)
---------------------------------------------------------
    data_root/
      K001/N09_M07_F10_K001_1.mat ...      # healthy
      KA01/N15_M07_F10_KA01_1.mat ...      # artificial outer-ring damage
      KI04/...                             # real inner-ring damage
      ...
Download: https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/

Usage
-----
# 0) look at what is inside a .mat file (channel names, lengths)
python paderborn_cnn1d.py --mode inspect --data-root /data/paderborn

# 1) quick baseline: healthy vs outer vs inner ring, real damages, one op. condition
python paderborn_cnn1d.py --data-root /data/paderborn --task 3class \
    --damage-source real --conditions N15_M07_F10 --epochs 30

# 2) harder / more honest: hold out entire bearings, all 4 operating conditions
python paderborn_cnn1d.py --data-root /data/paderborn --task 3class \
    --damage-source real --conditions all --split-by bearing --epochs 40

# 3) two channels, longer windows
python paderborn_cnn1d.py --data-root /data/paderborn \
    --channels vibration_1,phase_current_1 --window 4096 --max-windows-per-file 24

# 4) predict on single files with a trained checkpoint
python paderborn_cnn1d.py --mode predict --ckpt runs/exp1/best.pt \
    --predict-files /data/paderborn/KI04/N15_M07_F10_KI04_1.mat

Requirements: numpy, scipy, torch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.io import loadmat
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scipy is required:  pip install scipy") from exc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset



'''
# 1. sanity check the .mat parsing (prints channel names: force, phase_current_1/2,
#    speed, temp_2_bearing_module, torque, vibration_1)
python paderborn_cnn1d.py --mode inspect --data-root /data/paderborn

# 2. train
python paderborn_cnn1d.py --data-root /data/paderborn --task 3class \
       --damage-source real --conditions N15_M07_F10 --epochs 30 --out runs/real_3c

'''

# ----------------------------------------------------------------------------
# 1. Dataset meta information
# ----------------------------------------------------------------------------
# damage location: "healthy" | "OR" (outer ring) | "IR" (inner ring) | "IR+OR"
# source:          "healthy" | "artificial" (EDM/engraver/drilling) | "real" (run-to-failure)
BEARINGS: Dict[str, Tuple[str, str]] = {
    # --- undamaged -----------------------------------------------------------
    **{f"K00{i}": ("healthy", "healthy") for i in range(1, 7)},
    # --- artificial damages --------------------------------------------------
    "KA01": ("OR", "artificial"), "KA03": ("OR", "artificial"),
    "KA05": ("OR", "artificial"), "KA06": ("OR", "artificial"),
    "KA07": ("OR", "artificial"), "KA08": ("OR", "artificial"),
    "KA09": ("OR", "artificial"),
    "KI01": ("IR", "artificial"), "KI03": ("IR", "artificial"),
    "KI05": ("IR", "artificial"), "KI07": ("IR", "artificial"),
    "KI08": ("IR", "artificial"),
    # --- real (accelerated lifetime test) damages ---------------------------
    "KA04": ("OR", "real"), "KA15": ("OR", "real"), "KA16": ("OR", "real"),
    "KA22": ("OR", "real"), "KA30": ("OR", "real"),
    "KB23": ("IR+OR", "real"), "KB24": ("IR+OR", "real"), "KB27": ("IR+OR", "real"),
    "KI04": ("IR", "real"), "KI14": ("IR", "real"), "KI16": ("IR", "real"),
    "KI17": ("IR", "real"), "KI18": ("IR", "real"), "KI21": ("IR", "real"),
}

ALL_CONDITIONS = ["N09_M07_F10", "N15_M07_F10", "N15_M01_F10", "N15_M07_F04"]

FILE_RE = re.compile(r"(N\d{2}_M\d{2}_F\d{2})_([A-Z]{1,2}\d{2,3})_(\d+)", re.IGNORECASE)


@dataclass
class FileRec:
    path: Path
    condition: str
    code: str
    trial: int


# ----------------------------------------------------------------------------
# 2. MATLAB file reading
# ----------------------------------------------------------------------------
def _unwrap(obj):
    """Peel 1-element object arrays produced by scipy until we hit the payload."""
    while isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
        obj = obj.ravel()[0]
    return obj


def _field(obj, *names):
    for n in getattr(obj, "_fieldnames", []):
        if n.lower() in names:
            return getattr(obj, n)
    return None


def _entry_name(entry) -> str:
    v = _field(entry, "name")
    if v is None:
        return ""
    v = _unwrap(v)
    return str(v).strip()


def _entry_data(entry) -> Optional[np.ndarray]:
    v = _field(entry, "data", "raw", "signal", "values")
    if v is not None and np.size(v) > 1:
        return np.asarray(v).ravel()
    # fallback: largest numeric array among the fields
    best = None
    for n in getattr(entry, "_fieldnames", []):
        val = np.asarray(getattr(entry, n))
        if val.dtype.kind in "fiub" and val.size > 1 and (best is None or val.size > best.size):
            best = val
    return None if best is None else best.ravel()


def read_mat_channels(path: Path) -> Dict[str, np.ndarray]:
    """Return {channel_name: 1-D signal} for one Paderborn .mat file."""
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)

    rec = None
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        v = _unwrap(v)
        if hasattr(v, "_fieldnames"):
            rec = v
            break
    if rec is None:
        raise ValueError(f"{path.name}: no MATLAB struct found")

    channels: Dict[str, np.ndarray] = {}
    for fname in rec._fieldnames:                       # usually only 'Y'
        for entry in np.atleast_1d(getattr(rec, fname)).ravel():
            entry = _unwrap(entry)
            if not hasattr(entry, "_fieldnames"):
                continue
            data = _entry_data(entry)
            if data is None:
                continue
            name = _entry_name(entry) or f"{fname}"
            channels[name] = np.asarray(data, dtype=np.float32)
    if not channels:
        raise ValueError(f"{path.name}: could not locate any signal")
    return channels


def load_channels(path: Path, wanted: Sequence[str]) -> np.ndarray:
    """-> array (C, T) float32, channels truncated to the shortest length."""
    chans = read_mat_channels(path)
    lut = {k.lower(): v for k, v in chans.items()}
    sigs = []
    for ch in wanted:
        if ch.lower() not in lut:
            raise KeyError(f"{path.name}: channel {ch!r} not found. Available: {sorted(chans)}")
        sigs.append(lut[ch.lower()].ravel())
    n = min(len(s) for s in sigs)
    if n < 16:
        raise ValueError(f"{path.name}: signal too short ({n})")
    return np.stack([s[:n] for s in sigs], axis=0)


# ----------------------------------------------------------------------------
# 3. File index, labels, windowing, caching
# ----------------------------------------------------------------------------
def build_label_map(codes: Sequence[str], task: str) -> Tuple[Dict[str, int], List[str]]:
    """code -> class id, plus class names. Codes not in the map are dropped."""
    if task == "binary":
        names = ["healthy", "damaged"]
        m = {c: (0 if BEARINGS[c][0] == "healthy" else 1) for c in codes}
    elif task == "3class":
        names = ["healthy", "outer_ring", "inner_ring"]
        m = {}
        for c in codes:
            loc = BEARINGS[c][0]
            if loc == "healthy":
                m[c] = 0
            elif loc == "OR":
                m[c] = 1
            elif loc == "IR":
                m[c] = 2
            # "IR+OR" (KB*) intentionally dropped in 3-class mode
    elif task == "4class":
        names = ["healthy", "outer_ring", "inner_ring", "inner+outer"]
        loc2id = {"healthy": 0, "OR": 1, "IR": 2, "IR+OR": 3}
        m = {c: loc2id[BEARINGS[c][0]] for c in codes}
    elif task == "code":
        names = sorted(codes)
        m = {c: i for i, c in enumerate(names)}
    else:
        raise ValueError(task)
    return m, names


def index_files(root: Path, conditions: Sequence[str], allowed_codes: set,
                limit_per_code: Optional[int]) -> List[FileRec]:
    seen, recs = set(), []
    for p in sorted(root.rglob("*.mat")):
        m = FILE_RE.search(p.stem)
        if not m:
            continue
        cond, code, trial = m.group(1).upper(), m.group(2).upper(), int(m.group(3))
        if code not in allowed_codes or (conditions and cond not in conditions):
            continue
        key = (cond, code, trial)
        if key in seen:                      # duplicate copies in nested folders
            continue
        seen.add(key)
        recs.append(FileRec(p, cond, code, trial))

    if limit_per_code:
        per = defaultdict(int)
        keep = []
        for r in recs:
            if per[(r.code, r.condition)] < limit_per_code:
                keep.append(r)
                per[(r.code, r.condition)] += 1
        recs = keep
    return recs


def cut_windows(sig: np.ndarray, window: int, stride: int, max_windows: int) -> np.ndarray:
    """sig (C, T) -> (n, C, window); evenly spread over the record if capped."""
    T = sig.shape[1]
    if T < window:
        return np.empty((0, sig.shape[0], window), dtype=np.float32)
    starts = np.arange(0, T - window + 1, stride)
    if max_windows and len(starts) > max_windows:
        sel = np.unique(np.linspace(0, len(starts) - 1, max_windows).round().astype(int))
        starts = starts[sel]
    return np.stack([sig[:, s:s + window] for s in starts]).astype(np.float32)


def build_or_load_cache(recs: List[FileRec], channels: List[str], window: int, stride: int,
                        max_windows: int, cache_dir: Optional[Path]) -> dict:
    key = hashlib.sha1(json.dumps({
        "channels": channels, "window": window, "stride": stride, "max_windows": max_windows,
        "files": sorted(str(r.path) for r in recs),
    }, sort_keys=True).encode()).hexdigest()[:16]

    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"pu_{key}.npz"
        if cache_path.exists():
            print(f"[cache] loading {cache_path}")
            d = np.load(cache_path, allow_pickle=False)
            return {k: d[k] for k in d.files}

    codes = sorted({r.code for r in recs})
    conds = sorted({r.condition for r in recs})
    Xs, code_idx, cond_idx, file_idx, kept_files = [], [], [], [], []

    t0 = time.time()
    for i, r in enumerate(recs):
        try:
            sig = load_channels(r.path, channels)
            win = cut_windows(sig, window, stride, max_windows)
        except Exception as exc:                                    # noqa: BLE001
            print(f"  [warn] skipping {r.path.name}: {exc}")
            continue
        if len(win) == 0:
            print(f"  [warn] {r.path.name}: record shorter than window, skipped")
            continue
        fid = len(kept_files)
        kept_files.append(str(r.path))
        Xs.append(win)
        code_idx.append(np.full(len(win), codes.index(r.code), np.int32))
        cond_idx.append(np.full(len(win), conds.index(r.condition), np.int32))
        file_idx.append(np.full(len(win), fid, np.int32))
        if (i + 1) % 25 == 0 or i + 1 == len(recs):
            el = time.time() - t0
            print(f"  [read] {i + 1}/{len(recs)} files  {el:5.1f}s  "
                  f"(eta {el / (i + 1) * (len(recs) - i - 1):5.1f}s)")

    if not Xs:
        raise RuntimeError("No data was read -- check --data-root / --conditions / --channels.")

    out = {
        "X": np.concatenate(Xs).astype(np.float32),
        "code_idx": np.concatenate(code_idx),
        "cond_idx": np.concatenate(cond_idx),
        "file_idx": np.concatenate(file_idx),
        "codes": np.array(codes, dtype="<U8"),
        "conds": np.array(conds, dtype="<U16"),
        "files": np.array(kept_files, dtype="<U512"),
    }
    print(f"[data] windows: {out['X'].shape}  ({out['X'].nbytes / 2**20:.0f} MiB)")
    if cache_path is not None:
        np.savez(cache_path, **out)
        print(f"[cache] saved {cache_path}")
    return out


# ----------------------------------------------------------------------------
# 4. Splitting (group aware!)
# ----------------------------------------------------------------------------
def grouped_split(groups: np.ndarray, y: np.ndarray, val_frac: float, test_frac: float,
                  seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    glabel = {g: int(y[groups == g][0]) for g in uniq}
    by_class = defaultdict(list)
    for g in uniq:
        by_class[glabel[g]].append(g)

    tr, va, te = [], [], []
    for cls in sorted(by_class):
        gs = np.array(by_class[cls])
        rng.shuffle(gs)
        n = len(gs)
        n_te = min(max(1, round(test_frac * n)), max(0, n - 2)) if test_frac > 0 else 0
        n_va = min(max(1, round(val_frac * n)), max(0, n - n_te - 1)) if val_frac > 0 else 0
        te += list(gs[:n_te]); va += list(gs[n_te:n_te + n_va]); tr += list(gs[n_te + n_va:])

    idx = lambda sel: np.where(np.isin(groups, np.array(sel)))[0]  # noqa: E731
    return idx(tr), idx(va), idx(te)


# ----------------------------------------------------------------------------
# 5. torch Dataset
# ----------------------------------------------------------------------------
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, norm: str = "sample",
                 mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None,
                 augment: bool = False, seed: int = 0):
        self.X, self.y = X, y.astype(np.int64)
        self.norm, self.mean, self.std = norm, mean, std
        self.augment, self.seed, self.epoch = augment, seed, 0

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        x = np.array(self.X[i], dtype=np.float32, copy=True)        # (C, L)
        if self.augment:
            g = np.random.default_rng((self.seed * 1_000_003 + self.epoch * 7919 + i) % 2**32)
            if g.random() < 0.5:                                    # circular time shift
                x = np.roll(x, int(g.integers(x.shape[-1])), axis=-1)
            if g.random() < 0.5:                                    # random gain
                x *= np.float32(g.uniform(0.8, 1.25))
            if g.random() < 0.5:                                    # polarity is arbitrary
                x = -x
            if g.random() < 0.3:                                    # additive noise
                x += g.normal(0.0, 0.02 * (x.std() + 1e-8), x.shape).astype(np.float32)
        if self.norm == "sample":
            x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)
        elif self.norm == "global":
            x = (x - self.mean) / (self.std + 1e-8)
        return torch.from_numpy(x), int(self.y[i])


# ----------------------------------------------------------------------------
# 6. Model
# ----------------------------------------------------------------------------
class CNN1D(nn.Module):
    """Wide-first-kernel 1D CNN (WDCNN-ish) with BN and a GAP+GMP head."""

    def __init__(self, in_ch: int = 1, n_classes: int = 3, width: int = 32, n_blocks: int = 5,
                 first_kernel: int = 64, first_stride: int = 8, max_width: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv1d(in_ch, width, first_kernel, first_stride, first_kernel // 2, bias=False),
            nn.BatchNorm1d(width), nn.GELU(), nn.MaxPool1d(2),
        ]
        c = width
        for _ in range(max(0, n_blocks - 1)):
            co = min(c * 2, max_width)
            layers += [
                nn.Conv1d(c, co, 3, padding=1, bias=False), nn.BatchNorm1d(co), nn.GELU(),
                nn.Conv1d(co, co, 3, padding=1, bias=False), nn.BatchNorm1d(co), nn.GELU(),
                nn.MaxPool1d(2),
            ]
            c = co
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)                                   # (B, C, L')
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)         # (B, 2C)
        return self.head(h)


# ----------------------------------------------------------------------------
# 7. Metrics / helpers
# ----------------------------------------------------------------------------
def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def prf(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1e-12)
    rec = tp / np.maximum(cm.sum(1), 1e-12)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return prec, rec, f1


def report(cm: np.ndarray, names: Sequence[str], title: str) -> Dict[str, float]:
    prec, rec, f1 = prf(cm)
    acc = np.trace(cm) / max(cm.sum(), 1)
    w = max(12, max(len(n) for n in names) + 2)
    print(f"\n== {title} ==  acc={acc:.4f}  macro-F1={f1.mean():.4f}")
    print(f"{'class':<{w}}{'prec':>8}{'rec':>8}{'f1':>8}{'n':>8}")
    for i, n in enumerate(names):
        print(f"{n:<{w}}{prec[i]:8.3f}{rec[i]:8.3f}{f1[i]:8.3f}{cm[i].sum():8d}")
    print("confusion matrix (rows = true, cols = pred):")
    print("  " + "".join(f"{n[:7]:>8}" for n in names))
    for i, n in enumerate(names):
        print(f"{n[:10]:<10}" + "".join(f"{v:8d}" for v in cm[i]))
    return {"accuracy": float(acc), "macro_f1": float(f1.mean())}


def amp_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def amp_autocast(device_type: str, enabled: bool):
    try:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# 8. Train / eval loops
# ----------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    train = optimizer is not None
    model.train(train)
    tot_loss, n, preds, trues = 0.0, 0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), amp_autocast(device.type, use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        tot_loss += float(loss) * len(yb)
        n += len(yb)
        preds.append(logits.detach().float().argmax(1).cpu().numpy())
        trues.append(yb.cpu().numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    return tot_loss / n, float((preds == trues).mean()), trues, preds


@torch.no_grad()
def predict_probs(model, loader, device, use_amp=False) -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in loader:
        with amp_autocast(device.type, use_amp):
            out.append(torch.softmax(model(xb.to(device)).float(), 1).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------
# 9. Modes
# ----------------------------------------------------------------------------
def mode_inspect(args) -> None:
    root = Path(args.data_root)
    paths = [Path(args.inspect_file)] if args.inspect_file else sorted(root.rglob("*.mat"))[:3]
    if not paths:
        raise SystemExit(f"no .mat files under {root}")
    for p in paths:
        print(f"\n--- {p} ---")
        try:
            for name, sig in read_mat_channels(p).items():
                print(f"  {name:<28} len={sig.size:>8}  dtype={sig.dtype}  "
                      f"min={sig.min():+.4g} max={sig.max():+.4g} std={sig.std():.4g}")
        except Exception as exc:                                   # noqa: BLE001
            print(f"  !! {exc}")
    m = FILE_RE.search(paths[0].stem)
    if m:
        print(f"\nparsed name -> condition={m.group(1)} bearing={m.group(2)} trial={m.group(3)}")


def mode_predict(args) -> None:
    ck = torch.load(args.ckpt, map_location="cpu")
    cfg, names = ck["config"], ck["class_names"]
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CNN1D(in_ch=len(cfg["channels"]), n_classes=len(names), width=cfg["width"],
                  n_blocks=cfg["blocks"], first_kernel=cfg["first_kernel"],
                  first_stride=cfg["first_stride"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    for f in args.predict_files:
        sig = load_channels(Path(f), cfg["channels"])
        win = cut_windows(sig, cfg["window"], cfg["window"], args.max_windows_per_file or 0)
        ds = WindowDataset(win, np.zeros(len(win)), norm=cfg["norm"],
                           mean=np.array(ck.get("mean", 0.0), np.float32).reshape(-1, 1),
                           std=np.array(ck.get("std", 1.0), np.float32).reshape(-1, 1))
        probs = predict_probs(model, DataLoader(ds, batch_size=256), device)
        mean_p = probs.mean(0)
        print(f"\n{Path(f).name}: {names[int(mean_p.argmax())]}  "
              f"(windows={len(win)})")
        for i, n in enumerate(names):
            print(f"    {n:<14} {mean_p[i]:.3f}")


def mode_train(args) -> None:
    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    conditions = ALL_CONDITIONS if args.conditions.lower() == "all" else \
        [c.strip().upper() for c in args.conditions.split(",") if c.strip()]

    # --- which bearing codes are we allowed to use -------------------------
    src = args.damage_source
    allowed = {c for c, (_, s) in BEARINGS.items()
               if s == "healthy" or src == "all" or s == src}
    label_map, class_names = build_label_map(sorted(allowed), args.task)
    allowed = set(label_map)                       # 3class drops KB*

    recs = index_files(Path(args.data_root), conditions, allowed, args.limit_files_per_code)
    if not recs:
        raise SystemExit("No matching .mat files found -- check --data-root/--conditions.")
    print(f"[files] {len(recs)} measurements, {len({r.code for r in recs})} bearings, "
          f"conditions={sorted({r.condition for r in recs})}")

    data = build_or_load_cache(recs, channels, args.window, args.stride or args.window,
                              args.max_windows_per_file,
                              None if args.no_cache else Path(args.cache_dir or (Path(args.data_root) / "_cache")))

    codes = [str(c) for c in data["codes"]]
    y = np.array([label_map[codes[i]] for i in data["code_idx"]], dtype=np.int64)
    X = data["X"]
    groups = data["file_idx"] if args.split_by == "file" else data["code_idx"]

    idx_tr, idx_va, idx_te = grouped_split(groups, y, args.val_frac, args.test_frac, args.seed)
    print(f"[split] by {args.split_by}: train={len(idx_tr)} val={len(idx_va)} test={len(idx_te)} windows")
    for nm, idx in (("train", idx_tr), ("val", idx_va), ("test", idx_te)):
        cnt = np.bincount(y[idx], minlength=len(class_names))
        print(f"    {nm:<5} " + "  ".join(f"{class_names[i]}={cnt[i]}" for i in range(len(class_names)))
              + f"   bearings={sorted({codes[c] for c in data['code_idx'][idx]})}")

    # --- normalisation stats (train only) ---------------------------------
    mean = std = None
    if args.norm == "global":
        sub = X[idx_tr[:: max(1, len(idx_tr) // 20000)]]
        mean = sub.mean((0, 2), keepdims=False).reshape(-1, 1).astype(np.float32)
        std = sub.std((0, 2), keepdims=False).reshape(-1, 1).astype(np.float32)

    ds_tr = WindowDataset(X[idx_tr], y[idx_tr], args.norm, mean, std, args.augment, args.seed)
    ds_va = WindowDataset(X[idx_va], y[idx_va], args.norm, mean, std)
    ds_te = WindowDataset(X[idx_te], y[idx_te], args.norm, mean, std)

    kw = dict(num_workers=args.workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=args.workers > 0)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=len(ds_tr) > args.batch_size, **kw)
    dl_va = DataLoader(ds_va, batch_size=512, shuffle=False, **kw) if len(ds_va) else None
    dl_te = DataLoader(ds_te, batch_size=512, shuffle=False, **kw) if len(ds_te) else None

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CNN1D(len(channels), len(class_names), args.width, args.blocks,
                  args.first_kernel, args.first_stride, dropout=args.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] CNN1D  in_ch={len(channels)}  classes={len(class_names)}  params={n_par/1e3:.1f}k  dev={device}")

    weights = None
    if args.class_weights:
        cnt = np.bincount(y[idx_tr], minlength=len(class_names)).astype(np.float32)
        weights = torch.tensor(cnt.sum() / np.maximum(cnt, 1) / len(cnt), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=max(1, len(dl_tr)), pct_start=0.25)
    use_amp = args.amp and device.type == "cuda"
    scaler = amp_scaler(use_amp)

    best, best_ep, bad = -1.0, -1, 0
    hist = []
    for ep in range(1, args.epochs + 1):
        ds_tr.epoch = ep
        t0 = time.time()
        tr_loss, tr_acc, *_ = run_epoch(model, dl_tr, criterion, device, optimizer, scaler, use_amp)
        sched.step()
        if dl_va is not None:
            va_loss, va_acc, yv, pv = run_epoch(model, dl_va, criterion, device)
            score = prf(confusion(yv, pv, len(class_names)))[2].mean()
        else:
            va_loss, va_acc, score = float("nan"), float("nan"), tr_acc
        hist.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc,
                         val_loss=va_loss, val_acc=va_acc, val_macro_f1=float(score)))
        flag = ""
        if score > best:
            best, best_ep, bad = float(score), ep, 0
            torch.save({"model": model.state_dict(), "class_names": class_names,
                        "config": vars(args) | {"channels": channels},
                        "mean": None if mean is None else mean.ravel().tolist(),
                        "std": None if std is None else std.ravel().tolist()},
                       out_dir / "best.pt")
            flag = " *"
        else:
            bad += 1
        print(f"ep {ep:3d}/{args.epochs}  lr {optimizer.param_groups[0]['lr']:.2e}  "
              f"train {tr_loss:.4f}/{tr_acc:.4f}  val {va_loss:.4f}/{va_acc:.4f}  "
              f"F1 {score:.4f}  {time.time()-t0:4.1f}s{flag}")
        if args.patience and bad >= args.patience:
            print(f"[early stop] no val improvement for {args.patience} epochs")
            break

    # --- final evaluation with the best checkpoint --------------------------
    metrics = {"best_epoch": best_ep, "best_val_macro_f1": best, "history": hist}
    if (out_dir / "best.pt").exists():
        model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["model"])
    if dl_te is not None:
        probs = predict_probs(model, dl_te, device)
        pred, true = probs.argmax(1), y[idx_te]
        metrics["test_window"] = report(confusion(true, pred, len(class_names)), class_names,
                                        "TEST (window level)")
        # measurement-level decision: average softmax over all windows of a file
        fid = data["file_idx"][idx_te]
        agg_t, agg_p = [], []
        for f in np.unique(fid):
            m = fid == f
            agg_t.append(true[m][0])
            agg_p.append(probs[m].mean(0).argmax())
        metrics["test_file"] = report(confusion(np.array(agg_t), np.array(agg_p), len(class_names)),
                                      class_names, "TEST (measurement level, mean-softmax vote)")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[done] checkpoint + metrics in {out_dir}")


# ----------------------------------------------------------------------------
# 10. CLI
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Paderborn bearing dataset -> 1D-CNN classification",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["train", "inspect", "predict"], default="train")
    # data
    p.add_argument("--data-root", default="./paderborn", help="folder containing K001/, KA01/, ...")
    p.add_argument("--channels", default="vibration_1",
                   help="comma separated, e.g. vibration_1,phase_current_1,phase_current_2")
    p.add_argument("--conditions", default="N15_M07_F10",
                   help="'all' or comma separated subset of " + ",".join(ALL_CONDITIONS))
    p.add_argument("--task", choices=["binary", "3class", "4class", "code"], default="3class")
    p.add_argument("--damage-source", choices=["real", "artificial", "all"], default="real")
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--stride", type=int, default=0, help="0 = non-overlapping (=window)")
    p.add_argument("--max-windows-per-file", type=int, default=32, help="0 = keep all")
    p.add_argument("--limit-files-per-code", type=int, default=0, help="0 = all 20 trials")
    p.add_argument("--norm", choices=["sample", "global", "none"], default="sample")
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--no-cache", action="store_true")
    # split
    p.add_argument("--split-by", choices=["file", "bearing"], default="file")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.20)
    # model
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--first-kernel", type=int, default=64)
    p.add_argument("--first-stride", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.3)
    # optim
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--patience", type=int, default=10, help="0 disables early stopping")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs/exp1")
    # inspect / predict
    p.add_argument("--inspect-file", default="")
    p.add_argument("--ckpt", default="runs/exp1/best.pt")
    p.add_argument("--predict-files", nargs="*", default=[])
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.mode == "inspect":
        mode_inspect(args)
    elif args.mode == "predict":
        mode_predict(args)
    else:
        mode_train(args)


if __name__ == "__main__":
    main()


