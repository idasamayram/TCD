#!/usr/bin/env python
"""
export_pytorch.py — Paderborn (KAt) bearing dataset -> PyTorch-ready windowed tensors.

Replaces notebook 02. Stages:
    inventory -> export -> splits -> norm -> smoke -> verify

Key guarantees
  * inventory comes from an ALL-FILE artifact (file_inventory.csv / inventory.parquet);
    features.parquet (256-row subsample) is never allowed to drive export or N_KEEP.
  * read_errors.csv is filtered to fatal rows and ignored if it would drop >20% of files.
  * fast channels (64 kHz) and slow channels (4 kHz) are kept on their own time axes;
    conditioning values are time-aligned window means of the slow channels.
  * setpoint torque (`torque_set`) never collides with measured torque (`torque_nm`).
  * read failures / missing channels / non-finite / short files are recorded as skips and
    the output memmap is truncated, so skipped files leave no zero-filled samples.
  * 4-class target (healthy / OR / IR / IR+OR); binary is derived at training time.
  * three split strategies: bearing_holdout, artificial_to_real, cross_condition.

Usage
    python export_pytorch.py --stage all
    python export_pytorch.py --stage export --max-files 40 --win 8192 --hop 8192
    python export_pytorch.py --stage smoke  --strategy bearing_holdout
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------------------------

LABELS4 = {"healthy": 0, "OR": 1, "IR": 2, "IR+OR": 3}
NAMES4 = {v: k for k, v in LABELS4.items()}

CH_MAIN = ("vibration_1", "phase_current_1", "phase_current_2")   # fast, 64 kHz
CH_SLOW = ("speed", "torque", "force")                            # slow, 4 kHz -> conditioning
CH_AUX = ("temp_2_bearing_module",)                               # ~1 Hz, per-file scalar
COND_KEYS = ("speed_rpm", "torque_nm", "force_n")                 # conditioning feature order

INV_CANDIDATES = ("file_inventory.csv", "inventory.parquet", "file_inventory.parquet",
                  "files.parquet", "manifest.parquet", "file_index.parquet", "inventory.csv")
LEN_CANDIDATES = ("n_samples", "n_fast", "n_vibration", "n_samples_fast", "length", "len",
                  "n", "samples", "num_samples", "n_vib", "vibration_1__n")
MIN_INVENTORY_ROWS = 300     # anything smaller is a subsample
MAX_DROP_FRAC = 0.20         # read_errors.csv guard


@dataclass
class Cfg:
    root: Path = Path("../data_paderborn")
    eda: Optional[Path] = None
    out: Optional[Path] = None
    win: int = 8192
    hop: int = 8192
    dtype: str = "float16"
    channels: Tuple[str, ...] = CH_MAIN
    allow_short: bool = True       # short file -> fewer windows instead of being dropped
    max_files: int = 0             # 0 = all
    max_gb: float = 24.0
    seed: int = 1337
    test_cond: str = ""            # cross-condition holdout; "" -> auto
    strategy: str = "bearing_holdout"
    batch_size: int = 64
    workers: int = 0               # keep 0 on Windows unless you know you want more
    smoke_steps: int = 40
    splits_source: str = "auto"    # auto | artifact | derive
    augment: bool = True

    def __post_init__(self):
        self.root = Path(self.root)
        self.eda = Path(self.eda) if self.eda else self.root / "eda_out"
        self.out = Path(self.out) if self.out else self.root / "pt_export"
        self.out.mkdir(parents=True, exist_ok=True)
        assert self.win > 0 and self.hop > 0, "win/hop must be positive"
        assert self.dtype in ("float16", "float32")

    @property
    def np_dtype(self):
        return np.float16 if self.dtype == "float16" else np.float32

    @property
    def dat_path(self) -> Path:
        return self.out / "windows.dat"

    @property
    def meta_path(self) -> Path:
        return self.out / "meta.parquet"

    @property
    def export_meta_path(self) -> Path:
        return self.out / "export_meta.json"

    @property
    def splits_path(self) -> Path:
        return self.out / "splits_export.json"

    def norm_path(self, strategy: str) -> Path:
        return self.out / f"norm_{strategy}.npz"


def log(msg: str) -> None:
    print(msg, flush=True)


def pick(df: pd.DataFrame, *cands: str, required: bool = True) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    if required:
        raise KeyError(f"none of {cands} in columns {list(df.columns)}")
    return None


# ----------------------------------------------------------------------------------------
# 1. INVENTORY  (fixes the Cell-3 FileNotFoundError / empty-inventory bugs)
# ----------------------------------------------------------------------------------------

REAL_BEARINGS = {"KA04", "KA15", "KA16", "KA22", "KA30", "KI04", "KI14", "KI16", "KI17",
                 "KI18", "KI21", "KB23", "KB24", "KB27"}
ARTI_BEARINGS = {"KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09",
                 "KI01", "KI03", "KI05", "KI07", "KI08"}


def norm_component(s) -> str:
    t = str(s).strip().lower().replace(" ", "").replace("_", "")
    if t in {"healthy", "k0", "none", "ok", "0", "nan"}:
        return "healthy"
    if t in {"ir+or", "or+ir", "combined", "both", "irandor", "3"}:
        return "IR+OR"
    if t.startswith("ir") or "inner" in t or t == "2":
        return "IR"
    if t.startswith("or") or "outer" in t or t == "1":
        return "OR"
    raise ValueError(f"unmapped component {s!r}")


def _component_from_code(code: str) -> str:
    c = code.upper()
    if c.startswith("K0"):
        return "healthy"
    if c.startswith("KA"):
        return "OR"
    if c.startswith("KI"):
        return "IR"
    if c.startswith("KB"):
        return "IR+OR"
    raise ValueError(f"unknown bearing code {code!r}")


def _origin_from_code(code: str) -> str:
    c = code.upper()
    if c.startswith("K0"):
        return "healthy"
    return "real" if c in REAL_BEARINGS else ("artificial" if c in ARTI_BEARINGS else "unknown")


def scan_raw(cfg: Cfg) -> pd.DataFrame:
    """Last-resort inventory: parse N15_M07_F10_KA01_1.mat filenames under root."""
    import re
    skip = {cfg.eda.resolve(), cfg.out.resolve()}
    rows = []
    for p in cfg.root.rglob("*.mat"):
        rp = p.resolve()
        if any(s in rp.parents for s in skip):
            continue
        q = p.stem.split("_")
        if len(q) < 5:
            continue
        n, m, f, code, tr = q[0].upper(), q[1].upper(), q[2].upper(), q[-2].upper(), q[-1]
        if not (n.startswith("N") and m.startswith("M") and f.startswith("F") and code.startswith("K")):
            continue
        rows.append(dict(path=p.relative_to(cfg.root).as_posix(), bearing=code,
                         cond=f"{n}_{m}_{f}", trial=int(re.sub(r"\D", "", tr) or 0),
                         component=_component_from_code(code), origin=_origin_from_code(code),
                         rpm=int(n[1:]) * 100, torque=int(m[1:]) / 10, radial_force=int(f[1:]) * 100))
    return pd.DataFrame(rows)


def _apply_read_errors(cfg: Cfg, files: pd.DataFrame) -> pd.DataFrame:
    """read_errors.csv in this project logs EVERY read attempt -> filter to fatal rows only."""
    p = cfg.eda / "read_errors.csv"
    if not p.exists():
        return files
    err = pd.read_csv(p)
    pc = pick(err, "path", "filepath", "file", required=False)
    rc = pick(err, "reason", "error", "err", "message", "msg", "status", "note", required=False)
    if pc is None:
        log("read_errors: no path column -> ignored")
        return files

    sub = err
    if rc is not None:
        s = err[rc].astype(str).str.strip().str.lower()
        benign = s.isin({"", "nan", "none", "ok", "success", "0", "false", "-"})
        fatal_kw = ("error", "fail", "cannot", "could not", "unable", "corrupt", "truncat",
                    "empty", "missing", "exception", "traceback", "invalid", "short")
        sub = err[~benign & s.str.contains("|".join(fatal_kw), regex=True)]
        log(f"read_errors: {len(err)} rows -> {len(sub)} fatal (column '{rc}')")
    else:
        log(f"read_errors: no reason column; treating all {len(err)} rows as fatal (guarded)")

    bad = {Path(x).name for x in sub[pc].astype(str)}
    hit = files["path_abs"].map(lambda x: Path(x).name in bad)
    frac = float(hit.mean()) if len(files) else 0.0
    if frac > MAX_DROP_FRAC:
        log(f"!! read_errors would drop {int(hit.sum())} files ({frac:.0%}) > {MAX_DROP_FRAC:.0%} "
            f"-> IGNORED (inspect {p})")
        return files
    if hit.any():
        log(f"read_errors: excluded {int(hit.sum())} files")
        return files[~hit].reset_index(drop=True)
    log("read_errors: no matching files")
    return files


def load_inventory(cfg: Cfg) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    src, files = None, None
    for name in INV_CANDIDATES:
        p = cfg.eda / name
        if not p.exists():
            continue
        df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_parquet(p)
        if len(df) < MIN_INVENTORY_ROWS:
            log(f"skip {p.name}: only {len(df)} rows (subsample, not a full inventory)")
            continue
        src, files = p.name, df
        break

    if files is None:
        log(f"no usable inventory artifact in {cfg.eda} -> rebuilding from filesystem")
        files = scan_raw(cfg)
        assert len(files), f"no Paderborn .mat files under {cfg.root.resolve()} — check --root"
        files.to_parquet(cfg.eda / "inventory.parquet", index=False)
        src = "inventory.parquet (rebuilt)"
    log(f"inventory <- {src}  {files.shape}")

    COL = {k: pick(files, *v, required=req) for k, v, req in [
        ("path",       ("path", "filepath", "file", "fullpath", "rel_path"),  True),
        ("bearing",    ("bearing", "bearing_code", "code", "bearing_id"),     True),
        ("cond",       ("cond", "condition", "setting", "op_cond"),           True),
        ("trial",      ("trial", "rep", "measurement", "run"),                False),
        ("component",  ("component", "damage_component", "fault"),            False),
        ("origin",     ("origin", "damage_origin", "kind", "source"),         False),
        ("label",      ("label4", "label", "y"),                              False),
        ("mode",       ("mode",),                                             False),
        ("severity",   ("severity",),                                         False),
        ("rpm_set",    ("rpm", "speed_set", "speed_rpm_nominal"),             False),
        ("torque_set", ("torque", "torque_set", "torque_nm_nominal"),         False),
        ("force_set",  ("radial_force", "force_set", "force_n_nominal"),      False)]}
    log(f"resolved columns: {COL}")

    # ---- labels (prefer notebook-01's label4 to avoid a divergent mapping) ----
    if COL["label"] and pd.api.types.is_integer_dtype(files[COL["label"]]):
        files["label"] = files[COL["label"]].astype(int)
        files["label_name"] = files["label"].map(NAMES4)
        assert files["label_name"].notna().all(), "label4 contains values outside 0..3"
    else:
        src_col = COL["label"] or COL["component"]
        assert src_col, "inventory has neither a label nor a component column"
        files["label_name"] = files[src_col].map(norm_component)
        files["label"] = files["label_name"].map(LABELS4).astype(int)

    files["path_abs"] = files[COL["path"]].map(
        lambda p: str(Path(p) if Path(p).is_absolute() else (cfg.root / p)))
    files["bearing"] = files[COL["bearing"]].astype(str).str.upper()
    files["cond"] = files[COL["cond"]].astype(str)
    files["trial"] = files[COL["trial"]].astype(int) if COL["trial"] else 0
    files["origin"] = (files[COL["origin"]].astype(str) if COL["origin"]
                       else files["bearing"].map(_origin_from_code))
    for tgt, key in (("torque_set", "torque_set"), ("rpm_set", "rpm_set"), ("force_set", "force_set")):
        files[tgt] = pd.to_numeric(files[COL[key]], errors="coerce") if COL[key] else np.nan
    files["mode"] = files[COL["mode"]].astype(str) if COL["mode"] else "unknown"
    files["severity"] = files[COL["severity"]] if COL["severity"] else np.nan

    files = _apply_read_errors(cfg, files)

    missing = [p for p in files["path_abs"] if not Path(p).exists()]
    assert not missing, f"{len(missing)} inventory paths do not exist, e.g. {missing[:3]}"
    assert not files.duplicated(subset="path_abs").any(), "duplicate paths in inventory"
    assert len(files) >= MIN_INVENTORY_ROWS, f"only {len(files)} files survived filtering — aborting"

    files = files.sort_values(["bearing", "cond", "trial"]).reset_index(drop=True)
    files["file_id"] = np.arange(len(files), dtype=np.int64)
    if cfg.max_files:
        files = files.head(cfg.max_files).copy()
        log(f"--max-files: using first {len(files)} files")

    log(f"OK: {len(files)} files | {files.bearing.nunique()} bearings | "
        f"{files.cond.nunique()} conditions | unknown-origin={int((files.origin == 'unknown').sum())}")
    log("\nlabel x condition:\n" + str(files.pivot_table(index="label_name", columns="cond",
                                                         aggfunc="size", fill_value=0)))
    log("\norigin x label:\n" + str(files.groupby(["origin", "label_name"]).size().unstack(fill_value=0)))
    return files, COL


# ----------------------------------------------------------------------------------------
# 2. READER ADAPTER (pdb_io.py if available, else scipy fallback)
# ----------------------------------------------------------------------------------------

_READER = None
_READER_NAME = "none"


def _import_pdb_io(cfg: Cfg):
    for base in (Path.cwd(), Path(__file__).resolve().parent, cfg.root, cfg.root.parent, cfg.eda):
        p = Path(base) / "pdb_io.py"
        if p.exists():
            spec = importlib.util.spec_from_file_location("pdb_io", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)          # type: ignore[union-attr]
            log(f"pdb_io <- {p}")
            return mod
    log("pdb_io.py not found -> scipy fallback reader")
    return None


def _init_reader(cfg: Cfg) -> None:
    global _READER, _READER_NAME
    mod = _import_pdb_io(cfg)
    if mod is not None:
        for fn in ("read_mat", "read_file", "load_mat", "read_channels", "load_channels",
                   "read_paderborn", "read"):
            f = getattr(mod, fn, None)
            if callable(f):
                _READER, _READER_NAME = f, f"pdb_io.{fn}"
                log(f"reader = {_READER_NAME}")
                return
        log("pdb_io.py has no known reader function -> scipy fallback")
    _READER, _READER_NAME = _scipy_read, "scipy.io.loadmat"
    log(f"reader = {_READER_NAME}")


def _scipy_read(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    from scipy.io import loadmat
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    key = next(k for k in m if not k.startswith("__"))
    entries = np.atleast_1d(getattr(m[key], "Y"))
    sig, fs = {}, {}
    for e in entries:
        name = str(np.atleast_1d(getattr(e, "Name"))[0]).strip()
        data = np.asarray(getattr(e, "Data"), dtype=np.float64).ravel()
        rate = next((float(np.atleast_1d(getattr(e, a))[0])
                     for a in ("sampling_rate", "samplingRate", "Sampling_Rate", "fs", "rate")
                     if hasattr(e, a)), np.nan)
        sig[name], fs[name] = data, rate
    return sig, fs


def _normalize_reader_output(obj) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Accept dict[name]->array, dict[name]->{data,fs}, (data,fs) tuple, or DataFrame."""
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], dict):
        data, fs = obj
        return ({k: np.asarray(v, dtype=np.float64).ravel() for k, v in data.items()},
                {k: float(v) for k, v in (fs or {}).items() if np.isscalar(v)})
    if isinstance(obj, pd.DataFrame):
        return {c: obj[c].to_numpy(np.float64) for c in obj.columns}, {}
    if isinstance(obj, dict):
        sig, fs = {}, {}
        for k, v in obj.items():
            if isinstance(v, dict):
                arr = v.get("data", v.get("Data"))
                if arr is None:
                    continue
                sig[k] = np.asarray(arr, dtype=np.float64).ravel()
                r = v.get("fs", v.get("sampling_rate", v.get("rate")))
                if r is not None and np.isscalar(r):
                    fs[k] = float(r)
            elif np.isscalar(v):
                continue
            else:
                sig[k] = np.asarray(v, dtype=np.float64).ravel()
        return sig, fs
    raise TypeError(f"unsupported reader output type {type(obj)}")


def read_channels(path: str, wanted: Sequence[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    sig, fs = _normalize_reader_output(_READER(path))
    alias = {k.strip().lower().replace(" ", "_"): k for k in sig}
    out_s, out_f = {}, {}
    for w in wanted:
        k = sig.get(w, None)
        key = w if k is not None else alias.get(w.lower())
        if key is None:
            out_s[w] = np.empty(0, dtype=np.float64)
            out_f[w] = np.nan
        else:
            out_s[w] = np.asarray(sig[key], dtype=np.float64).ravel()
            out_f[w] = float(fs.get(key, np.nan))
    return out_s, out_f


# ----------------------------------------------------------------------------------------
# 3. WINDOWING / TIME ALIGNMENT
# ----------------------------------------------------------------------------------------

def windows_of(x: np.ndarray, n_w: int, win: int, hop: int) -> np.ndarray:
    x = np.ascontiguousarray(x)
    s = x.strides[0]
    return np.lib.stride_tricks.as_strided(x, shape=(n_w, win), strides=(s * hop, s))


def time_axis(name: str, sig: Dict[str, np.ndarray], fs: Dict[str, float],
              ref: str, fs_ref: float) -> np.ndarray:
    """Own time axis per channel; infer fs from length ratio when the reader gives none."""
    n = len(sig[name])
    if n == 0:
        return np.empty(0)
    f = fs.get(name, np.nan)
    if not np.isfinite(f) or f <= 0:
        n_ref = max(len(sig[ref]), 1)
        f = fs_ref * n / n_ref
    return np.arange(n, dtype=np.float64) / f


def window_means(x: np.ndarray, t: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Mean of a slow channel over each fast window's [t0, t1) interval."""
    if len(x) == 0:
        return np.full(len(t0), np.nan)
    lo = np.searchsorted(t, t0, side="left")
    hi = np.searchsorted(t, t1, side="left")
    csum = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    cnt = np.maximum(hi - lo, 1)
    out = (csum[hi] - csum[lo]) / cnt
    empty = hi <= lo
    if empty.any():                                  # window shorter than slow sample period
        out[empty] = x[np.clip(lo[empty], 0, len(x) - 1)]
    return out


# ----------------------------------------------------------------------------------------
# 4. N_KEEP RESOLUTION (all-file artifact first, subsample only as warned fallback)
# ----------------------------------------------------------------------------------------

def resolve_n_keep(cfg: Cfg, files: pd.DataFrame) -> Tuple[int, str]:
    def snap(n: int) -> int:
        return (max(n - cfg.win, 0) // cfg.hop) * cfg.hop + cfg.win

    # tier 1: census.parquet
    p = cfg.eda / "census.parquet"
    if p.exists():
        cen = pd.read_parquet(p)
        col = pick(cen, *LEN_CANDIDATES, required=False)
        if col is None:
            num = [c for c in cen.select_dtypes("number").columns
                   if cen[c].min() >= cfg.win and "n" in c.lower()]
            col = num[0] if num else None
        if col is not None and len(cen) >= MIN_INVENTORY_ROWS:
            n_min = int(pd.to_numeric(cen[col], errors="coerce").dropna().min())
            if n_min >= cfg.win:
                return snap(n_min), f"census.parquet['{col}'] (n={len(cen)}, min={n_min})"
            log(f"census min length {n_min} < win {cfg.win} -> next tier")
        else:
            log(f"census.parquet unusable (rows={len(cen)}, col={col}) -> next tier")

    # tier 2: eda_summary.json
    p = cfg.eda / "eda_summary.json"
    if p.exists():
        try:
            js = json.loads(p.read_text())
            flat: Dict[str, float] = {}

            def walk(d, pre=""):
                for k, v in (d.items() if isinstance(d, dict) else []):
                    walk(v, f"{pre}{k}.") if isinstance(v, dict) else flat.update(
                        {f"{pre}{k}": v} if isinstance(v, (int, float)) else {})
            walk(js)
            for k, v in flat.items():
                kl = k.lower()
                if any(t in kl for t in ("n_keep", "min_len", "min_samples", "n_min", "min_n")):
                    if int(v) >= cfg.win:
                        return snap(int(v)), f"eda_summary.json['{k}']={int(v)}"
        except Exception as e:
            log(f"eda_summary.json unreadable: {e!r}")

    # tier 3: probe the raw files
    log("WARNING: no all-file length artifact -> probing raw files for min length")
    k = min(48, len(files))
    idx = np.linspace(0, len(files) - 1, k).astype(int)
    lens = []
    for i in idx:
        try:
            sig, _ = read_channels(files.path_abs.iloc[i], cfg.channels)
            lens.append(min(len(sig[c]) for c in cfg.channels))
        except Exception:
            pass
    assert lens, "could not read any file while probing lengths"
    n_min = int(min(lens))
    assert n_min >= cfg.win, f"probed min length {n_min} < win {cfg.win}"
    return snap(n_min), f"probe of {len(lens)} files (min={n_min}); allow_short handles the rest"


# ----------------------------------------------------------------------------------------
# 5. EXPORT
# ----------------------------------------------------------------------------------------

def _tqdm(it, **kw):
    try:
        from tqdm.auto import tqdm
        return tqdm(it, **kw)
    except Exception:
        return it


def stage_export(cfg: Cfg) -> None:
    _init_reader(cfg)
    files, COL = load_inventory(cfg)
    n_keep, how = resolve_n_keep(cfg, files)
    n_win_full = 1 + (n_keep - cfg.win) // cfg.hop
    C = len(cfg.channels)
    n_upper = n_win_full * len(files)
    gb = n_upper * C * cfg.win * np.dtype(cfg.np_dtype).itemsize / 1e9

    log(f"\nN_KEEP={n_keep} via {how}")
    log(f"win={cfg.win} hop={cfg.hop} -> {n_win_full} windows/file, upper bound {n_upper} windows")
    log(f"channels={list(cfg.channels)} dtype={cfg.dtype} -> {gb:.2f} GB")
    assert gb <= cfg.max_gb, f"estimated {gb:.1f} GB > --max-gb {cfg.max_gb}; raise it or increase --hop"

    all_ch = tuple(dict.fromkeys(tuple(cfg.channels) + CH_SLOW + CH_AUX))
    X = np.memmap(cfg.dat_path, mode="w+", dtype=cfg.np_dtype, shape=(n_upper, C, cfg.win))
    meta_rows: List[dict] = []
    skipped: List[dict] = []
    cur = 0
    ref = cfg.channels[0]

    for r in _tqdm(list(files.itertuples()), total=len(files), desc="export"):
        try:
            sig, fs = read_channels(r.path_abs, all_ch)
            for c in cfg.channels:
                if len(sig[c]) == 0:
                    raise ValueError(f"missing fast channel '{c}'")
            n_av = min(len(sig[c]) for c in cfg.channels)
            if n_av < cfg.win:
                raise ValueError(f"only {n_av} samples < win {cfg.win}")
            if n_av < n_keep and not cfg.allow_short:
                raise ValueError(f"{n_av} < N_KEEP {n_keep} and allow_short=False")

            n_f = min(n_av, n_keep)
            n_w = 1 + (n_f - cfg.win) // cfg.hop
            main = np.stack([sig[c][:n_f] for c in cfg.channels])
            if not np.isfinite(main).all():
                raise ValueError("non-finite samples in fast channels")
            blk = np.stack([windows_of(main[i], n_w, cfg.win, cfg.hop) for i in range(C)], axis=1)

            fs_ref = fs.get(ref, np.nan)
            if not np.isfinite(fs_ref) or fs_ref <= 0:
                fs_ref = 64000.0
            t_ref = np.arange(n_f, dtype=np.float64) / fs_ref
            t0 = t_ref[np.arange(n_w) * cfg.hop]
            t1 = t0 + cfg.win / fs_ref
            ops = {c: window_means(sig[c], time_axis(c, sig, fs, ref, fs_ref), t0, t1) for c in CH_SLOW}
            aux = sig.get(CH_AUX[0], np.empty(0))
            temp = float(np.mean(aux)) if len(aux) else np.nan
        except Exception as e:
            skipped.append({"file_id": int(r.file_id), "path": r.path_abs, "reason": repr(e),
                            "trace": traceback.format_exc(limit=1)})
            continue

        X[cur:cur + n_w] = blk.astype(cfg.np_dtype, copy=False)
        for w in range(n_w):
            meta_rows.append(dict(
                idx=cur + w, file_id=int(r.file_id), window=w, start=int(w * cfg.hop),
                n_win_file=n_w, n_samples_used=int(n_f),
                bearing=r.bearing, cond=r.cond, trial=int(r.trial), origin=r.origin,
                mode=r.mode, label=int(r.label), label_name=r.label_name,
                speed_rpm=float(ops["speed"][w]), torque_nm=float(ops["torque"][w]),
                force_n=float(ops["force"][w]), temp_c=temp,
                rpm_set=float(r.rpm_set) if pd.notna(r.rpm_set) else np.nan,
                torque_set=float(r.torque_set) if pd.notna(r.torque_set) else np.nan,
                force_set=float(r.force_set) if pd.notna(r.force_set) else np.nan))
        cur += n_w

    X.flush()
    del X
    os.truncate(cfg.dat_path, cur * C * cfg.win * np.dtype(cfg.np_dtype).itemsize)

    meta = pd.DataFrame(meta_rows)
    assert len(meta) == cur, f"metadata rows {len(meta)} != written windows {cur}"
    meta.to_parquet(cfg.meta_path, index=False)
    files.to_parquet(cfg.out / "files_used.parquet", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(cfg.out / "export_skipped.csv", index=False)

    info = dict(shape=[int(cur), C, cfg.win], dtype=cfg.dtype, channels=list(cfg.channels),
                slow_channels=list(CH_SLOW), aux_channels=list(CH_AUX), cond_keys=list(COND_KEYS),
                win=cfg.win, hop=cfg.hop, n_keep=int(n_keep), n_keep_source=how,
                n_windows_per_file_full=int(n_win_full), allow_short=cfg.allow_short,
                n_files_in=int(len(files)), n_files_exported=int(meta.file_id.nunique()),
                n_skipped=len(skipped), labels=LABELS4, seed=cfg.seed,
                reader=_READER_NAME, dat=cfg.dat_path.name)
    cfg.export_meta_path.write_text(json.dumps(info, indent=2))

    log(f"\nwrote {cfg.dat_path.name}: {cur} windows, "
        f"{cur * C * cfg.win * np.dtype(cfg.np_dtype).itemsize / 1e9:.2f} GB")
    log(f"skipped files: {len(skipped)}"
        + (f" -> {cfg.out / 'export_skipped.csv'}" if skipped else ""))
    log("\nwindows per class:\n" + str(meta.label_name.value_counts()))


# ----------------------------------------------------------------------------------------
# 6. SPLITS  (bearing_holdout | artificial_to_real | cross_condition)
# ----------------------------------------------------------------------------------------

def _split_bearings(bearings: Sequence[str], rng: np.random.Generator,
                    frac=(0.6, 0.2, 0.2)) -> Tuple[List[str], List[str], List[str]]:
    b = list(bearings)
    rng.shuffle(b)
    n = len(b)
    if n >= 3:
        n_tr = max(1, int(round(frac[0] * n)))
        n_va = max(1, int(round(frac[1] * n)))
        n_tr = min(n_tr, n - 2)
        n_va = min(n_va, n - n_tr - 1)
        return b[:n_tr], b[n_tr:n_tr + n_va], b[n_tr + n_va:]
    if n == 2:
        return b[:1], [], b[1:]
    return b, [], []


def build_splits(cfg: Cfg, meta: pd.DataFrame) -> Dict[str, Dict[str, List[int]]]:
    fmeta = meta.drop_duplicates("file_id")[["file_id", "bearing", "cond", "trial",
                                             "origin", "label_name"]].reset_index(drop=True)
    rng = np.random.default_rng(cfg.seed)
    ids = lambda mask: sorted(int(x) for x in fmeta.file_id[mask])
    splits: Dict[str, Dict[str, List[int]]] = {}

    # --- 1. bearing holdout: stratified over bearings by class -------------------------------
    tr, va, te = [], [], []
    for lbl, grp in fmeta.groupby("label_name"):
        a, b, c = _split_bearings(sorted(grp.bearing.unique()), rng)
        tr += a; va += b; te += c
    splits["bearing_holdout"] = {"train": ids(fmeta.bearing.isin(tr)),
                                 "val": ids(fmeta.bearing.isin(va)),
                                 "test": ids(fmeta.bearing.isin(te)),
                                 "_bearings": {"train": tr, "val": va, "test": te}}

    # --- 2. artificial -> real (healthy bearings split, never shared) ------------------------
    h_tr, h_va, h_te = _split_bearings(sorted(fmeta.bearing[fmeta.origin == "healthy"].unique()), rng)
    arti = sorted(fmeta.bearing[fmeta.origin == "artificial"].unique())
    real = sorted(fmeta.bearing[fmeta.origin == "real"].unique())
    rng.shuffle(arti)
    n_va = max(1, int(round(0.15 * len(arti)))) if len(arti) >= 4 else 0
    a_va, a_tr = arti[:n_va], arti[n_va:]
    splits["artificial_to_real"] = {"train": ids(fmeta.bearing.isin(a_tr + h_tr)),
                                    "val": ids(fmeta.bearing.isin(a_va + h_va)),
                                    "test": ids(fmeta.bearing.isin(real + h_te)),
                                    "_bearings": {"train": a_tr + h_tr, "val": a_va + h_va,
                                                  "test": real + h_te}}

    # --- 3. cross-condition holdout (val = held-out trials of the train conditions) ----------
    conds = sorted(fmeta.cond.unique())
    test_cond = cfg.test_cond or ("N09_M07_F10" if "N09_M07_F10" in conds else conds[0])
    assert test_cond in conds, f"--test-cond {test_cond!r} not in {conds}"
    in_test = fmeta.cond == test_cond
    trials = sorted(fmeta.trial[~in_test].unique())
    val_trials = set(trials[-max(1, len(trials) // 10):]) if len(trials) > 2 else set()
    splits["cross_condition"] = {"train": ids(~in_test & ~fmeta.trial.isin(val_trials)),
                                 "val": ids(~in_test & fmeta.trial.isin(val_trials)),
                                 "test": ids(in_test),
                                 "_test_cond": test_cond, "_val_trials": sorted(val_trials)}
    return splits


def load_or_build_splits(cfg: Cfg, meta: pd.DataFrame) -> Dict[str, Dict[str, List[int]]]:
    if cfg.splits_path.exists() and cfg.splits_source != "derive":
        js = json.loads(cfg.splits_path.read_text())
        log(f"splits <- {cfg.splits_path.name}")
        return js

    art = cfg.eda / "splits.json"
    if cfg.splits_source in ("auto", "artifact") and art.exists():
        try:
            js = json.loads(art.read_text())
            fmeta = meta.drop_duplicates("file_id")[["file_id", "bearing", "path_abs"]] \
                if "path_abs" in meta.columns else meta.drop_duplicates("file_id")[["file_id", "bearing"]]
            out: Dict[str, Dict[str, List[int]]] = {}
            for strat, d in js.items():
                if not isinstance(d, dict):
                    continue
                got = {}
                for part in ("train", "val", "test"):
                    vals = {str(v).upper() for v in d.get(part, [])}
                    if not vals:
                        continue
                    hit = fmeta.bearing.str.upper().isin(vals)
                    if hit.any():
                        got[part] = sorted(int(x) for x in fmeta.file_id[hit])
                if {"train", "test"} <= set(got):
                    got.setdefault("val", [])
                    out[strat] = got
            if out:
                log(f"splits <- {art.name} (reused, strategies: {list(out)})")
                cfg.splits_path.write_text(json.dumps(out, indent=2))
                return out
            log(f"{art.name} could not be mapped to file_ids -> deriving")
        except Exception as e:
            log(f"{art.name} unusable ({e!r}) -> deriving")

    splits = build_splits(cfg, meta)
    cfg.splits_path.write_text(json.dumps(splits, indent=2))
    log(f"splits derived -> {cfg.splits_path.name}")
    return splits


def stage_splits(cfg: Cfg) -> None:
    meta = pd.read_parquet(cfg.meta_path)
    splits = load_or_build_splits(cfg, meta)
    rows = []
    for strat, d in splits.items():
        for part in ("train", "val", "test"):
            f = d.get(part, [])
            m = meta[meta.file_id.isin(f)]
            rows.append(dict(strategy=strat, split=part, files=len(f), windows=len(m),
                             bearings=m.bearing.nunique(), conds=m.cond.nunique(),
                             classes=m.label_name.nunique()))
    log("\n" + str(pd.DataFrame(rows).to_string(index=False)))
    check_leakage(meta, splits)


def check_leakage(meta: pd.DataFrame, splits: Dict[str, Dict[str, List[int]]]) -> None:
    for strat, d in splits.items():
        parts = {p: set(d.get(p, [])) for p in ("train", "val", "test")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            inter = parts[a] & parts[b]
            assert not inter, f"{strat}: file_id overlap {a}/{b}: {sorted(inter)[:5]}"
        bset = {p: set(meta.bearing[meta.file_id.isin(ids)]) for p, ids in parts.items()}
        if strat in ("bearing_holdout", "artificial_to_real"):
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                inter = bset[a] & bset[b]
                assert not inter, f"{strat}: bearing overlap {a}/{b}: {sorted(inter)}"
        if strat == "cross_condition":
            ctr = set(meta.cond[meta.file_id.isin(parts["train"])])
            cte = set(meta.cond[meta.file_id.isin(parts["test"])])
            assert not (ctr & cte), f"cross_condition: condition overlap {sorted(ctr & cte)}"
        if strat == "artificial_to_real":
            o_te = set(meta.origin[meta.file_id.isin(parts["test"])])
            assert "artificial" not in o_te, "artificial_to_real: artificial bearings leaked into test"
        log(f"leakage OK: {strat}")


# ----------------------------------------------------------------------------------------
# 7. NORMALIZATION (train split only)
# ----------------------------------------------------------------------------------------

def open_memmap(cfg: Cfg) -> Tuple[np.memmap, dict]:
    info = json.loads(cfg.export_meta_path.read_text())
    X = np.memmap(cfg.dat_path, mode="r", dtype=np.dtype(info["dtype"]),
                  shape=tuple(info["shape"]))
    return X, info


def stage_norm(cfg: Cfg) -> None:
    meta = pd.read_parquet(cfg.meta_path)
    splits = load_or_build_splits(cfg, meta)
    X, info = open_memmap(cfg)
    C = info["shape"][1]

    for strat, d in splits.items():
        idx = np.asarray(meta.index[meta.file_id.isin(d.get("train", []))], dtype=np.int64)
        assert len(idx), f"{strat}: empty train split"
        n = np.zeros(C); s = np.zeros(C); ss = np.zeros(C)
        for chunk in np.array_split(idx, max(1, len(idx) // 4096)):
            b = np.asarray(X[np.sort(chunk)], dtype=np.float64)
            n += b.shape[0] * b.shape[2]
            s += b.sum(axis=(0, 2))
            ss += (b ** 2).sum(axis=(0, 2))
        mean = s / n
        std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-12))

        cm = meta.loc[idx, list(COND_KEYS)].to_numpy(np.float64)
        cmean = np.nanmean(cm, axis=0)
        cstd = np.nanstd(cm, axis=0)
        cstd[~np.isfinite(cstd) | (cstd < 1e-9)] = 1.0

        np.savez(cfg.norm_path(strat), mean=mean.astype(np.float32), std=std.astype(np.float32),
                 cond_mean=cmean.astype(np.float32), cond_std=cstd.astype(np.float32),
                 n_windows=len(idx), channels=np.array(info["channels"]),
                 cond_keys=np.array(list(COND_KEYS)))
        log(f"{strat}: train windows={len(idx)}  mean={np.round(mean, 5)}  std={np.round(std, 5)}")
    del X


# ----------------------------------------------------------------------------------------
# 8. DATASET / DATALOADERS  (torch)
# ----------------------------------------------------------------------------------------

def _require_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception as e:
        log(f"torch unavailable ({e!r}) -> skipping dataset/smoke stages")
        return False


def build_torch_bits():
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

    class WindowDataset(Dataset):
        """Lazy per-worker memmap; returns (x[C,W], cond[3], label)."""

        def __init__(self, dat: Path, shape, dtype, idx, labels, cond, mean, std,
                     cond_mean, cond_std, augment=False, shift=0.05, noise=0.01, seed=0):
            self.dat, self.shape, self.dtype = str(dat), tuple(shape), np.dtype(dtype)
            self.idx = np.asarray(idx, dtype=np.int64)
            self.labels = np.asarray(labels, dtype=np.int64)
            self.cond = np.nan_to_num(np.asarray(cond, dtype=np.float32))
            self.mean = np.asarray(mean, np.float32)[:, None]
            self.std = np.asarray(std, np.float32)[:, None]
            self.cond_mean = np.asarray(cond_mean, np.float32)
            self.cond_std = np.asarray(cond_std, np.float32)
            self.augment, self.shift, self.noise, self.seed = augment, shift, noise, seed
            self._X = None

        def __len__(self):
            return len(self.idx)

        @property
        def X(self):
            if self._X is None:
                self._X = np.memmap(self.dat, mode="r", dtype=self.dtype, shape=self.shape)
            return self._X

        def __getitem__(self, i):
            j = int(self.idx[i])
            x = np.asarray(self.X[j], dtype=np.float32)
            if self.augment:
                rng = np.random.default_rng((self.seed * 1_000_003 + j) & 0x7FFFFFFF)
                if self.shift:
                    x = np.roll(x, int(rng.integers(-int(self.shift * x.shape[-1]),
                                                    int(self.shift * x.shape[-1]) + 1)), axis=-1)
                if self.noise:
                    x = x + rng.normal(0.0, self.noise, x.shape).astype(np.float32) * self.std
            x = (x - self.mean) / self.std
            c = (self.cond[i] - self.cond_mean) / self.cond_std
            return (torch.from_numpy(np.ascontiguousarray(x)),
                    torch.from_numpy(np.ascontiguousarray(c)),
                    torch.tensor(int(self.labels[i])))

    class CondCNN(nn.Module):
        def __init__(self, in_ch: int, n_cls: int = 4, n_cond: int = 3, width: int = 32):
            super().__init__()
            def blk(i, o, k=7, s=4):
                return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2, bias=False),
                                     nn.BatchNorm1d(o), nn.ReLU(inplace=True))
            self.body = nn.Sequential(blk(in_ch, width), blk(width, width * 2),
                                      blk(width * 2, width * 4), blk(width * 4, width * 4),
                                      nn.AdaptiveAvgPool1d(1), nn.Flatten())
            self.head = nn.Sequential(nn.Linear(width * 4 + n_cond, 128), nn.ReLU(inplace=True),
                                      nn.Dropout(0.3), nn.Linear(128, n_cls))

        def forward(self, x, c):
            return self.head(torch.cat([self.body(x), c], dim=1))

    return torch, nn, Dataset, DataLoader, WeightedRandomSampler, WindowDataset, CondCNN


def make_loaders(cfg: Cfg, strategy: str):
    torch, nn, _, DataLoader, WeightedRandomSampler, WindowDataset, CondCNN = build_torch_bits()
    meta = pd.read_parquet(cfg.meta_path)
    splits = load_or_build_splits(cfg, meta)
    assert strategy in splits, f"unknown strategy {strategy!r}; have {list(splits)}"
    info = json.loads(cfg.export_meta_path.read_text())
    npz_path = cfg.norm_path(strategy)
    assert npz_path.exists(), f"missing {npz_path.name}; run --stage norm first"
    z = np.load(npz_path, allow_pickle=False)

    loaders, sets = {}, {}
    for part in ("train", "val", "test"):
        sel = meta.file_id.isin(splits[strategy].get(part, []))
        idx = np.asarray(meta.index[sel], dtype=np.int64)
        if not len(idx):
            log(f"{strategy}/{part}: empty -> skipped")
            continue
        ds = WindowDataset(cfg.dat_path, info["shape"], info["dtype"], idx,
                           meta.loc[idx, "label"].to_numpy(), meta.loc[idx, list(COND_KEYS)].to_numpy(),
                           z["mean"], z["std"], z["cond_mean"], z["cond_std"],
                           augment=(part == "train" and cfg.augment), seed=cfg.seed)
        if part == "train":
            y = meta.loc[idx, "label"].to_numpy()
            cnt = np.bincount(y, minlength=len(LABELS4)).astype(np.float64)
            w = (1.0 / np.maximum(cnt, 1))[y]
            sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                            num_samples=len(idx), replacement=True)
            dl = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler,
                            num_workers=cfg.workers, pin_memory=torch.cuda.is_available(),
                            drop_last=True, persistent_workers=cfg.workers > 0)
        else:
            dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.workers, pin_memory=torch.cuda.is_available(),
                            persistent_workers=cfg.workers > 0)
        loaders[part], sets[part] = dl, ds
        log(f"{strategy}/{part}: {len(idx)} windows, {int(sel.sum() and meta.file_id[sel].nunique())} files")
    return loaders, sets, meta, splits, info


def stage_smoke(cfg: Cfg) -> None:
    if not _require_torch():
        return
    torch, nn, *_ , CondCNN = build_torch_bits()
    loaders, sets, meta, splits, info = make_loaders(cfg, cfg.strategy)
    torch.manual_seed(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CondCNN(in_ch=info["shape"][1], n_cls=len(LABELS4), n_cond=len(COND_KEYS)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    log(f"device={dev}  params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    xb, cb, yb = next(iter(loaders["train"]))
    assert xb.shape[1:] == (info["shape"][1], cfg.win), f"bad batch shape {tuple(xb.shape)}"
    assert torch.isfinite(xb).all() and torch.isfinite(cb).all(), "non-finite batch"
    log(f"batch x={tuple(xb.shape)} cond={tuple(cb.shape)} y={tuple(yb.shape)} "
        f"x.mean={xb.mean():.3f} x.std={xb.std():.3f}")

    model.train()
    losses = []
    it = iter(loaders["train"])
    for step in range(cfg.smoke_steps):
        try:
            xb, cb, yb = next(it)
        except StopIteration:
            it = iter(loaders["train"]); xb, cb, yb = next(it)
        xb, cb, yb = xb.to(dev), cb.to(dev), yb.to(dev)
        loss = crit(model(xb, cb), yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(float(loss))
        if step % 10 == 0 or step == cfg.smoke_steps - 1:
            log(f"step {step:3d}  loss {loss:.4f}")
    assert np.isfinite(losses).all(), "non-finite loss"
    first, last = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
    log(f"loss {first:.4f} -> {last:.4f}")
    if last >= first:
        warnings.warn("loss did not decrease over the smoke run — check lr / normalization")

    part = "val" if "val" in loaders else "test"
    model.eval()
    hit = tot = 0
    with torch.no_grad():
        for k, (xb, cb, yb) in enumerate(loaders[part]):
            p = model(xb.to(dev), cb.to(dev)).argmax(1).cpu()
            hit += int((p == yb).sum()); tot += len(yb)
            if k >= 20:
                break
    log(f"{part} accuracy on {tot} windows (untrained-ish): {hit / max(tot, 1):.3f}")
    log("SMOKE TEST OK")


# ----------------------------------------------------------------------------------------
# 9. VERIFY
# ----------------------------------------------------------------------------------------

def stage_verify(cfg: Cfg) -> None:
    info = json.loads(cfg.export_meta_path.read_text())
    meta = pd.read_parquet(cfg.meta_path)
    X, _ = open_memmap(cfg)

    assert len(meta) == info["shape"][0], f"meta {len(meta)} != tensor rows {info['shape'][0]}"
    assert (meta.idx.to_numpy() == np.arange(len(meta))).all(), "meta.idx is not 0..N-1"
    exp_bytes = int(np.prod(info["shape"])) * np.dtype(info["dtype"]).itemsize
    assert cfg.dat_path.stat().st_size == exp_bytes, "dat file size does not match declared shape"

    cnt = meta.groupby("file_id").size()
    assert cnt.between(1, info["n_windows_per_file_full"]).all(), "impossible windows/file"
    log(f"files={cnt.size}  windows/file {cnt.min()}-{cnt.max()}  "
        f"short-of-N_KEEP={int((cnt < info['n_windows_per_file_full']).sum())}")
    assert (cnt.values == meta.groupby('file_id').n_win_file.first().values).all(), \
        "n_win_file disagrees with actual row count"

    for k in COND_KEYS:
        bad = int((~np.isfinite(meta[k])).sum())
        log(f"{k}: [{np.nanmin(meta[k]):.2f}, {np.nanmax(meta[k]):.2f}]  non-finite={bad}")
        assert bad == 0, f"non-finite conditioning value in {k}"
    assert meta.speed_rpm.between(0, 5000).all(), "speed_rpm out of plausible range"
    assert "torque_set" in meta.columns and "torque_nm" in meta.columns, \
        "setpoint/measured torque collision"

    rng = np.random.default_rng(cfg.seed)
    probe = rng.choice(len(meta), size=min(256, len(meta)), replace=False)
    blk = np.asarray(X[np.sort(probe)], dtype=np.float32)
    assert np.isfinite(blk).all(), "non-finite samples in tensor"
    assert (blk.std(axis=(0, 2)) > 0).all(), "a channel is constant (zero-filled rows?)"
    log(f"probe {len(probe)} windows: per-channel std={np.round(blk.std(axis=(0, 2)), 5)}")

    log("\nlabel distribution (windows):\n" + str(meta.label_name.value_counts()))
    log("\nlabel x cond:\n" + str(meta.pivot_table(index="label_name", columns="cond",
                                                   aggfunc="size", fill_value=0)))
    sk = cfg.out / "export_skipped.csv"
    if sk.exists():
        s = pd.read_csv(sk)
        log(f"\nskipped {len(s)} files; top reasons:\n"
            + str(s.reason.str.slice(0, 80).value_counts().head(5)))

    check_leakage(meta, load_or_build_splits(cfg, meta))
    del X
    log("\nVERIFY OK")


# ----------------------------------------------------------------------------------------
# 10. CLI
# ----------------------------------------------------------------------------------------

def parse_args(argv=None) -> Tuple[Cfg, str]:
    p = argparse.ArgumentParser(description="Paderborn -> PyTorch window export")
    p.add_argument("--stage", default="all",
                   choices=["inventory", "export", "splits", "norm", "smoke", "verify", "all"])
    p.add_argument("--root", default="../data_paderborn")
    p.add_argument("--eda", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--win", type=int, default=8192)
    p.add_argument("--hop", type=int, default=8192)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--channels", nargs="+", default=list(CH_MAIN))
    p.add_argument("--no-allow-short", dest="allow_short", action="store_false")
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--max-gb", type=float, default=24.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--test-cond", default="")
    p.add_argument("--strategy", default="bearing_holdout",
                   choices=["bearing_holdout", "artificial_to_real", "cross_condition"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--smoke-steps", type=int, default=40)
    p.add_argument("--splits-source", default="auto", choices=["auto", "artifact", "derive"])
    p.add_argument("--no-augment", dest="augment", action="store_false")
    a = p.parse_args(argv)
    cfg = Cfg(root=a.root, eda=a.eda, out=a.out, win=a.win, hop=a.hop, dtype=a.dtype,
              channels=tuple(a.channels), allow_short=a.allow_short, max_files=a.max_files,
              max_gb=a.max_gb, seed=a.seed, test_cond=a.test_cond, strategy=a.strategy,
              batch_size=a.batch_size, workers=a.workers, smoke_steps=a.smoke_steps,
              splits_source=a.splits_source, augment=a.augment)
    return cfg, a.stage


def main(argv=None) -> int:
    cfg, stage = parse_args(argv)
    log(f"root={cfg.root.resolve()}\neda ={cfg.eda.resolve()}\nout ={cfg.out.resolve()}")
    log(json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
                   indent=2, default=str))
    order = ["export", "splits", "norm", "smoke", "verify"] if stage == "all" else [stage]
    for s in order:
        log(f"\n{'=' * 78}\n== {s.upper()}\n{'=' * 78}")
        if s == "inventory":
            _init_reader(cfg); load_inventory(cfg)
        elif s == "export":
            stage_export(cfg)
        elif s == "splits":
            stage_splits(cfg)
        elif s == "norm":
            stage_norm(cfg)
        elif s == "smoke":
            stage_smoke(cfg)
        elif s == "verify":
            stage_verify(cfg)
    log("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())