
#!/usr/bin/env python
"""
export_pytorch.py — Paderborn (KAt) bearing dataset -> PyTorch-ready windowed tensors.

Pipeline:
    inventory -> export -> splits -> norm -> smoke -> verify

This version is aligned with the validated notebook pipeline:

    fast channels:
        vibration_1
        phase_current_1
        phase_current_2

    sampling rate:
        64 kHz

    window:
        4096 samples

    hop:
        4096 samples

    dtype:
        float32

    target:
        4 classes:
            healthy = 0
            OR      = 1
            IR      = 2
            IR+OR   = 3

Important:
    EDA artifacts are used only for the full inventory and N_KEEP.
    eda_out/splits.json is NEVER reused.

    Splits are always derived from the exported metadata and written to:
        pt_export/splits_export.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 0. CONFIG
# =============================================================================

LABELS4 = {
    "healthy": 0,
    "OR": 1,
    "IR": 2,
    "IR+OR": 3,
}

NAMES4 = {v: k for k, v in LABELS4.items()}

CH_MAIN = (
    "vibration_1",
    "phase_current_1",
    "phase_current_2",
)

CH_SLOW = (
    "speed",
    "torque",
    "force",
)

CH_AUX = (
    "temp_2_bearing_module",
)

COND_KEYS = (
    "speed_rpm",
    "torque_nm",
    "force_n",
    "temp_c",
)

FS_MAIN = 64_000

INV_CANDIDATES = (
    "file_inventory.csv",
    "inventory.parquet",
    "file_inventory.parquet",
    "files.parquet",
    "manifest.parquet",
    "file_index.parquet",
    "inventory.csv",
)

LEN_CANDIDATES = (
    "n_samples",
    "n_fast",
    "n_vibration",
    "n_samples_fast",
    "length",
    "len",
    "n",
    "samples",
    "num_samples",
    "n_vib",
    "vibration_1__n",
)

MIN_INVENTORY_ROWS = 300
MAX_DROP_FRAC = 0.20


# =============================================================================
# PADERBORN BEARING ORIGIN
# =============================================================================

REAL_BEARINGS = {
    "KA04", "KA15", "KA16", "KA22", "KA30",
    "KI04", "KI14", "KI16", "KI17", "KI18", "KI21",
    "KB23", "KB24", "KB27",
}

ARTI_BEARINGS = {
    "KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09",
    "KI01", "KI03", "KI05", "KI07", "KI08",
}


def log(msg: str) -> None:
    print(msg, flush=True)


# =============================================================================
# 1. CONFIG OBJECT
# =============================================================================

@dataclass
class Cfg:
    root: Path = Path("../data_paderborn")
    eda: Optional[Path] = None
    out: Optional[Path] = None

    # Aligned with working notebook
    win: int = 4096
    hop: int = 4096
    dtype: str = "float32"

    channels: Tuple[str, ...] = CH_MAIN

    allow_short: bool = True
    max_files: int = 0
    max_gb: float = 24.0

    seed: int = 0

    # Cross-condition test condition
    test_cond: str = ""

    strategy: str = "bearing_holdout"

    batch_size: int = 128
    workers: int = 0
    smoke_steps: int = 10

    # IMPORTANT:
    # Always derive splits. Do not consume stale EDA splits.
    splits_source: str = "derive"

    augment: bool = True

    def __post_init__(self):
        self.root = Path(self.root)

        self.eda = (
            Path(self.eda)
            if self.eda
            else self.root / "eda_out"
        )

        self.out = (
            Path(self.out)
            if self.out
            else self.root / "pt_export"
        )

        self.out.mkdir(parents=True, exist_ok=True)

        assert self.win > 0
        assert self.hop > 0
        assert self.dtype in ("float16", "float32")

    @property
    def np_dtype(self):
        return (
            np.float16
            if self.dtype == "float16"
            else np.float32
        )

    @property
    def dat_path(self):
        return self.out / "windows.dat"

    @property
    def meta_path(self):
        return self.out / "meta.parquet"

    @property
    def export_meta_path(self):
        return self.out / "export_meta.json"

    @property
    def splits_path(self):
        return self.out / "splits_export.json"

    def norm_path(self, strategy: str):
        return self.out / f"norm_{strategy}.npz"


# =============================================================================
# 2. HELPERS
# =============================================================================

def pick(
    df: pd.DataFrame,
    *cands: str,
    required: bool = True,
) -> Optional[str]:

    for c in cands:
        if c in df.columns:
            return c

    low = {c.lower(): c for c in df.columns}

    for c in cands:
        if c.lower() in low:
            return low[c.lower()]

    if required:
        raise KeyError(
            f"none of {cands} in columns {list(df.columns)}"
        )

    return None


def norm_component(s) -> str:

    t = (
        str(s)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
    )

    if t in {
        "healthy", "k0", "none", "ok", "0", "nan"
    }:
        return "healthy"

    if t in {
        "ir+or", "or+ir", "combined",
        "both", "irandor", "3"
    }:
        return "IR+OR"

    if (
        t.startswith("ir")
        or "inner" in t
        or t == "2"
    ):
        return "IR"

    if (
        t.startswith("or")
        or "outer" in t
        or t == "1"
    ):
        return "OR"

    raise ValueError(
        f"unmapped component {s!r}"
    )


def component_from_code(code: str) -> str:

    c = code.upper()

    if c.startswith("K0"):
        return "healthy"

    if c.startswith("KA"):
        return "OR"

    if c.startswith("KI"):
        return "IR"

    if c.startswith("KB"):
        return "IR+OR"

    raise ValueError(
        f"unknown bearing code {code!r}"
    )


def origin_from_code(code: str) -> str:

    c = code.upper()

    if c.startswith("K0"):
        return "healthy"

    if c in REAL_BEARINGS:
        return "real"

    if c in ARTI_BEARINGS:
        return "artificial"

    return "unknown"


# =============================================================================
# 3. INVENTORY
# =============================================================================

def scan_raw(cfg: Cfg) -> pd.DataFrame:

    import re

    skip = {
        cfg.eda.resolve(),
        cfg.out.resolve(),
    }

    rows = []

    for p in cfg.root.rglob("*.mat"):

        rp = p.resolve()

        if any(s in rp.parents for s in skip):
            continue

        q = p.stem.split("_")

        if len(q) < 5:
            continue

        n = q[0].upper()
        m = q[1].upper()
        f = q[2].upper()
        code = q[-2].upper()
        tr = q[-1]

        if not (
            n.startswith("N")
            and m.startswith("M")
            and f.startswith("F")
            and code.startswith("K")
        ):
            continue

        rows.append(
            dict(
                path=p.relative_to(cfg.root).as_posix(),
                bearing=code,
                cond=f"{n}_{m}_{f}",
                trial=int(
                    re.sub(r"\D", "", tr) or 0
                ),
                component=component_from_code(code),
                origin=origin_from_code(code),
                rpm=int(n[1:]) * 100,
                torque=int(m[1:]) / 10,
                radial_force=int(f[1:]) * 100,
            )
        )

    return pd.DataFrame(rows)


def apply_read_errors(
    cfg: Cfg,
    files: pd.DataFrame,
) -> pd.DataFrame:

    p = cfg.eda / "read_errors.csv"

    if not p.exists():
        return files

    err = pd.read_csv(p)

    pc = pick(
        err,
        "path",
        "filepath",
        "file",
        required=False,
    )

    rc = pick(
        err,
        "reason",
        "error",
        "err",
        "message",
        "msg",
        "status",
        "note",
        required=False,
    )

    if pc is None:
        log(
            "read_errors: no path column -> ignored"
        )
        return files

    sub = err

    if rc is not None:

        s = (
            err[rc]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        benign = s.isin({
            "",
            "nan",
            "none",
            "ok",
            "success",
            "0",
            "false",
            "-",
        })

        fatal_kw = (
            "error",
            "fail",
            "cannot",
            "could not",
            "unable",
            "corrupt",
            "truncat",
            "empty",
            "missing",
            "exception",
            "traceback",
            "invalid",
            "short",
        )

        sub = err[
            ~benign
            & s.str.contains(
                "|".join(fatal_kw),
                regex=True,
            )
        ]

        log(
            f"read_errors: {len(err)} rows -> "
            f"{len(sub)} fatal "
            f"(column '{rc}')"
        )

    else:

        log(
            f"read_errors: no reason column; "
            f"treating all {len(err)} rows as fatal"
        )

    bad = {
        Path(x).name
        for x in sub[pc].astype(str)
    }

    hit = files["path_abs"].map(
        lambda x: Path(x).name in bad
    )

    frac = (
        float(hit.mean())
        if len(files)
        else 0.0
    )

    if frac > MAX_DROP_FRAC:

        log(
            f"!! read_errors would drop "
            f"{int(hit.sum())} files "
            f"({frac:.0%}) > {MAX_DROP_FRAC:.0%} "
            "-> IGNORED"
        )

        return files

    if hit.any():

        log(
            f"read_errors: excluded "
            f"{int(hit.sum())} files"
        )

        return files[
            ~hit
        ].reset_index(drop=True)

    log(
        "read_errors: no matching files"
    )

    return files


def load_inventory(
    cfg: Cfg,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:

    src = None
    files = None

    for name in INV_CANDIDATES:

        p = cfg.eda / name

        if not p.exists():
            continue

        df = (
            pd.read_csv(p)
            if p.suffix.lower() == ".csv"
            else pd.read_parquet(p)
        )

        if len(df) < MIN_INVENTORY_ROWS:

            log(
                f"skip {p.name}: only {len(df)} rows "
                "(subsample, not a full inventory)"
            )

            continue

        src = p.name
        files = df
        break

    if files is None:

        log(
            f"no usable inventory artifact in "
            f"{cfg.eda} -> rebuilding from filesystem"
        )

        files = scan_raw(cfg)

        assert len(files), (
            f"no Paderborn .mat files under "
            f"{cfg.root.resolve()}"
        )

        files.to_parquet(
            cfg.eda / "inventory.parquet",
            index=False,
        )

        src = "inventory.parquet (rebuilt)"

    log(
        f"inventory <- {src} {files.shape}"
    )

    COL = {
        k: pick(
            files,
            *v,
            required=req,
        )
        for k, v, req in [
            (
                "path",
                (
                    "path",
                    "filepath",
                    "file",
                    "fullpath",
                    "rel_path",
                ),
                True,
            ),
            (
                "bearing",
                (
                    "bearing",
                    "bearing_code",
                    "code",
                    "bearing_id",
                ),
                True,
            ),
            (
                "cond",
                (
                    "cond",
                    "condition",
                    "setting",
                    "op_cond",
                ),
                True,
            ),
            (
                "trial",
                (
                    "trial",
                    "rep",
                    "measurement",
                    "run",
                ),
                False,
            ),
            (
                "component",
                (
                    "component",
                    "damage_component",
                    "fault",
                ),
                False,
            ),
            (
                "origin",
                (
                    "origin",
                    "damage_origin",
                    "kind",
                    "source",
                ),
                False,
            ),
            (
                "label",
                (
                    "label4",
                    "label",
                    "y",
                ),
                False,
            ),
            (
                "mode",
                ("mode",),
                False,
            ),
            (
                "severity",
                ("severity",),
                False,
            ),
            (
                "rpm_set",
                (
                    "rpm",
                    "speed_set",
                    "speed_rpm_nominal",
                ),
                False,
            ),
            (
                "torque_set",
                (
                    "torque",
                    "torque_set",
                    "torque_nm_nominal",
                ),
                False,
            ),
            (
                "force_set",
                (
                    "radial_force",
                    "force_set",
                    "force_n_nominal",
                ),
                False,
            ),
        ]
    }

    log(
        f"resolved columns: {COL}"
    )

    # -------------------------------------------------------------------------
    # labels
    # -------------------------------------------------------------------------

    if (
        COL["label"]
        and pd.api.types.is_integer_dtype(
            files[COL["label"]]
        )
    ):

        files["label"] = (
            files[COL["label"]]
            .astype(int)
        )

        files["label_name"] = (
            files["label"]
            .map(NAMES4)
        )

        assert (
            files["label_name"]
            .notna()
            .all()
        ), (
            "label4 contains values outside 0..3"
        )

    else:

        src_col = (
            COL["label"]
            or COL["component"]
        )

        assert src_col, (
            "inventory has neither label nor component"
        )

        files["label_name"] = (
            files[src_col]
            .map(norm_component)
        )

        files["label"] = (
            files["label_name"]
            .map(LABELS4)
            .astype(int)
        )

    # -------------------------------------------------------------------------
    # standard metadata
    # -------------------------------------------------------------------------

    files["path_abs"] = files[
        COL["path"]
    ].map(
        lambda p:
            str(
                Path(p)
                if Path(p).is_absolute()
                else cfg.root / p
            )
    )

    files["bearing"] = (
        files[COL["bearing"]]
        .astype(str)
        .str.upper()
    )

    files["cond"] = (
        files[COL["cond"]]
        .astype(str)
    )

    files["trial"] = (
        files[COL["trial"]].astype(int)
        if COL["trial"]
        else 0
    )

    files["origin"] = (
        files[COL["origin"]].astype(str)
        if COL["origin"]
        else files["bearing"].map(
            origin_from_code
        )
    )

    for tgt, key in (
        ("torque_set", "torque_set"),
        ("rpm_set", "rpm_set"),
        ("force_set", "force_set"),
    ):

        files[tgt] = (
            pd.to_numeric(
                files[COL[key]],
                errors="coerce",
            )
            if COL[key]
            else np.nan
        )

    files["mode"] = (
        files[COL["mode"]].astype(str)
        if COL["mode"]
        else "unknown"
    )

    files["severity"] = (
        files[COL["severity"]]
        if COL["severity"]
        else np.nan
    )

    files = apply_read_errors(
        cfg,
        files,
    )

    missing = [
        p
        for p in files["path_abs"]
        if not Path(p).exists()
    ]

    assert not missing, (
        f"{len(missing)} inventory paths "
        f"do not exist, e.g. {missing[:3]}"
    )

    assert not files.duplicated(
        subset="path_abs"
    ).any(), (
        "duplicate paths in inventory"
    )

    assert len(files) >= MIN_INVENTORY_ROWS

    files = (
        files
        .sort_values(
            ["bearing", "cond", "trial"]
        )
        .reset_index(drop=True)
    )

    files["file_id"] = np.arange(
        len(files),
        dtype=np.int64,
    )

    if cfg.max_files:

        files = (
            files
            .head(cfg.max_files)
            .copy()
        )

        log(
            f"--max-files: using first "
            f"{len(files)} files"
        )

    log(
        f"OK: {len(files)} files | "
        f"{files.bearing.nunique()} bearings | "
        f"{files.cond.nunique()} conditions | "
        f"unknown-origin="
        f"{int((files.origin == 'unknown').sum())}"
    )

    log(
        "\nlabel x condition:\n"
        + str(
            files.pivot_table(
                index="label_name",
                columns="cond",
                aggfunc="size",
                fill_value=0,
            )
        )
    )

    log(
        "\norigin x label:\n"
        + str(
            files.groupby(
                ["origin", "label_name"]
            ).size()
            .unstack(fill_value=0)
        )
    )

    return files, COL


# =============================================================================
# 4. READER
# =============================================================================

_READER = None
_READER_NAME = "none"


def import_pdb_io(cfg: Cfg):

    candidates = (
        Path.cwd(),
        Path(__file__).resolve().parent,
        cfg.root,
        cfg.root.parent,
        cfg.eda,
    )

    for base in candidates:

        p = Path(base) / "pdb_io.py"

        if not p.exists():
            continue

        spec = (
            importlib.util
            .spec_from_file_location(
                "pdb_io",
                p,
            )
        )

        mod = (
            importlib.util
            .module_from_spec(spec)
        )

        spec.loader.exec_module(mod)

        log(
            f"pdb_io <- {p}"
        )

        return mod

    log(
        "pdb_io.py not found -> "
        "scipy fallback reader"
    )

    return None


def scipy_read(
    path: str,
):

    from scipy.io import loadmat

    m = loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False,
    )

    key = next(
        k for k in m
        if not k.startswith("__")
    )

    entries = np.atleast_1d(
        getattr(m[key], "Y")
    )

    sig = {}
    fs = {}

    for e in entries:

        name = str(
            np.atleast_1d(
                getattr(e, "Name")
            )[0]
        ).strip()

        data = np.asarray(
            getattr(e, "Data"),
            dtype=np.float64,
        ).ravel()

        rate = next(
            (
                float(
                    np.atleast_1d(
                        getattr(e, a)
                    )[0]
                )
                for a in (
                    "sampling_rate",
                    "samplingRate",
                    "Sampling_Rate",
                    "fs",
                    "rate",
                )
                if hasattr(e, a)
            ),
            np.nan,
        )

        sig[name] = data
        fs[name] = rate

    return sig, fs


def normalize_reader_output(obj):

    if (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], dict)
    ):

        data, fs = obj

        return (
            {
                k: np.asarray(
                    v,
                    dtype=np.float64,
                ).ravel()
                for k, v in data.items()
            },
            {
                k: float(v)
                for k, v in (fs or {}).items()
                if np.isscalar(v)
            },
        )

    if isinstance(obj, pd.DataFrame):

        return (
            {
                c: obj[c].to_numpy(
                    np.float64
                )
                for c in obj.columns
            },
            {},
        )

    if isinstance(obj, dict):

        sig = {}
        fs = {}

        for k, v in obj.items():

            if isinstance(v, dict):

                arr = v.get(
                    "data",
                    v.get("Data"),
                )

                if arr is None:
                    continue

                sig[k] = np.asarray(
                    arr,
                    dtype=np.float64,
                ).ravel()

                r = v.get(
                    "fs",
                    v.get(
                        "sampling_rate",
                        v.get("rate"),
                    ),
                )

                if (
                    r is not None
                    and np.isscalar(r)
                ):
                    fs[k] = float(r)

            elif np.isscalar(v):
                continue

            else:

                sig[k] = np.asarray(
                    v,
                    dtype=np.float64,
                ).ravel()

        return sig, fs

    raise TypeError(
        f"unsupported reader output "
        f"type {type(obj)}"
    )


def init_reader(cfg: Cfg):

    global _READER
    global _READER_NAME

    mod = import_pdb_io(cfg)

    if mod is not None:

        for fn in (
            "read_mat",
            "read_file",
            "load_mat",
            "read_channels",
            "load_channels",
            "read_paderborn",
            "read",
        ):

            f = getattr(
                mod,
                fn,
                None,
            )

            if callable(f):

                _READER = f
                _READER_NAME = (
                    f"pdb_io.{fn}"
                )

                log(
                    f"reader = {_READER_NAME}"
                )

                return

        log(
            "pdb_io.py has no known reader "
            "function -> scipy fallback"
        )

    _READER = scipy_read
    _READER_NAME = "scipy.io.loadmat"

    log(
        f"reader = {_READER_NAME}"
    )


def read_channels(
    path: str,
    wanted: Sequence[str],
):

    obj = _READER(path)

    sig, fs = normalize_reader_output(
        obj
    )

    alias = {
        k.strip().lower().replace(" ", "_"): k
        for k in sig
    }

    out_s = {}
    out_f = {}

    for w in wanted:

        key = (
            w
            if w in sig
            else alias.get(
                w.lower()
            )
        )

        if key is None:

            out_s[w] = np.empty(
                0,
                dtype=np.float64,
            )

            out_f[w] = np.nan

        else:

            out_s[w] = np.asarray(
                sig[key],
                dtype=np.float64,
            ).ravel()

            out_f[w] = float(
                fs.get(
                    key,
                    np.nan,
                )
            )

    return out_s, out_f


# =============================================================================
# 5. WINDOWING / TIME ALIGNMENT
# =============================================================================

def windows_of(
    x: np.ndarray,
    n_w: int,
    win: int,
    hop: int,
):

    x = np.ascontiguousarray(x)

    if hop == win:

        return x[
            :n_w * win
        ].reshape(
            n_w,
            win,
        )

    return (
        np.lib.stride_tricks
        .sliding_window_view(
            x,
            win,
        )[::hop][:n_w]
    )


def time_axis(
    name: str,
    sig: Dict[str, np.ndarray],
    fs: Dict[str, float],
    ref: str,
    fs_ref: float,
):

    n = len(sig[name])

    if n == 0:
        return np.empty(0)

    f = fs.get(
        name,
        np.nan,
    )

    if not np.isfinite(f) or f <= 0:

        n_ref = max(
            len(sig[ref]),
            1,
        )

        f = (
            fs_ref
            * n
            / n_ref
        )

    return (
        np.arange(n, dtype=np.float64)
        / f
    )


def window_means(
    x: np.ndarray,
    t: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
):

    if len(x) == 0:

        return np.full(
            len(t0),
            np.nan,
        )

    lo = np.searchsorted(
        t,
        t0,
        side="left",
    )

    hi = np.searchsorted(
        t,
        t1,
        side="left",
    )

    csum = np.concatenate(
        [[0.0], np.cumsum(
            x,
            dtype=np.float64,
        )]
    )

    cnt = np.maximum(
        hi - lo,
        1,
    )

    out = (
        csum[hi]
        - csum[lo]
    ) / cnt

    empty = hi <= lo

    if empty.any():

        out[empty] = x[
            np.clip(
                lo[empty],
                0,
                len(x) - 1,
            )
        ]

    return out


# =============================================================================
# 6. N_KEEP
# =============================================================================

def resolve_n_keep(
    cfg: Cfg,
    files: pd.DataFrame,
):

    def snap(n):

        return (
            max(
                n - cfg.win,
                0,
            )
            // cfg.hop
        ) * cfg.hop + cfg.win

    # -------------------------------------------------------------------------
    # census.parquet
    # -------------------------------------------------------------------------

    p = cfg.eda / "census.parquet"

    if p.exists():

        cen = pd.read_parquet(p)

        col = pick(
            cen,
            *LEN_CANDIDATES,
            required=False,
        )

        if col is not None and len(cen) >= MIN_INVENTORY_ROWS:

            vals = pd.to_numeric(
                cen[col],
                errors="coerce",
            ).dropna()

            if len(vals):

                n_min = int(
                    vals.min()
                )

                if n_min >= cfg.win:

                    return (
                        snap(n_min),
                        f"census.parquet['{col}'] "
                        f"(n={len(cen)}, min={n_min})",
                    )

        log(
            "census.parquet unusable -> "
            "next tier"
        )

    # -------------------------------------------------------------------------
    # eda_summary.json
    # -------------------------------------------------------------------------

    p = cfg.eda / "eda_summary.json"

    if p.exists():

        try:

            js = json.loads(
                p.read_text()
            )

            flat = {}

            def walk(
                d,
                pre="",
            ):

                if not isinstance(
                    d,
                    dict,
                ):
                    return

                for k, v in d.items():

                    if isinstance(
                        v,
                        dict,
                    ):

                        walk(
                            v,
                            f"{pre}{k}.",
                        )

                    elif isinstance(
                        v,
                        (int, float),
                    ):

                        flat[
                            f"{pre}{k}"
                        ] = v

            walk(js)

            for k, v in flat.items():

                kl = k.lower()

                if any(
                    t in kl
                    for t in (
                        "n_keep",
                        "min_len",
                        "min_samples",
                        "n_min",
                        "min_n",
                    )
                ):

                    if int(v) >= cfg.win:

                        return (
                            snap(int(v)),
                            f"eda_summary.json['{k}']={int(v)}",
                        )

        except Exception as e:

            log(
                f"eda_summary.json unreadable: {e!r}"
            )

    # -------------------------------------------------------------------------
    # raw-file probe
    # -------------------------------------------------------------------------

    log(
        "WARNING: no all-file length artifact "
        "-> probing raw files"
    )

    k = min(
        48,
        len(files),
    )

    idx = np.linspace(
        0,
        len(files) - 1,
        k,
    ).astype(int)

    lens = []

    for i in idx:

        try:

            sig, _ = read_channels(
                files.path_abs.iloc[i],
                cfg.channels,
            )

            lens.append(
                min(
                    len(sig[c])
                    for c in cfg.channels
                )
            )

        except Exception:
            pass

    assert lens

    n_min = int(
        min(lens)
    )

    assert n_min >= cfg.win

    return (
        snap(n_min),
        f"probe of {len(lens)} files",
    )


# =============================================================================
# 7. EXPORT
# =============================================================================

def stage_export(
    cfg: Cfg,
):

    init_reader(cfg)

    files, COL = load_inventory(
        cfg
    )

    n_keep, how = resolve_n_keep(
        cfg,
        files,
    )

    n_win_full = (
        1
        + (n_keep - cfg.win)
        // cfg.hop
    )

    C = len(
        cfg.channels
    )

    n_upper = (
        n_win_full
        * len(files)
    )

    gb = (
        n_upper
        * C
        * cfg.win
        * np.dtype(
            cfg.np_dtype
        ).itemsize
        / 1e9
    )

    log(
        f"\nN_KEEP={n_keep:,} via {how}"
    )

    log(
        f"win={cfg.win} hop={cfg.hop} "
        f"-> {n_win_full} windows/file, "
        f"upper bound {n_upper:,} windows"
    )

    log(
        f"channels={list(cfg.channels)} "
        f"dtype={cfg.dtype} -> {gb:.2f} GB"
    )

    assert gb <= cfg.max_gb, (
        f"estimated {gb:.1f} GB > "
        f"--max-gb {cfg.max_gb}"
    )

    all_ch = tuple(
        dict.fromkeys(
            tuple(cfg.channels)
            + CH_SLOW
            + CH_AUX
        )
    )

    X = np.memmap(
        cfg.dat_path,
        mode="w+",
        dtype=cfg.np_dtype,
        shape=(
            n_upper,
            C,
            cfg.win,
        ),
    )

    meta_rows = []
    skipped = []

    cur = 0

    ref = cfg.channels[0]

    from tqdm.auto import tqdm

    for r in tqdm(
        list(files.itertuples()),
        total=len(files),
        desc="export",
    ):

        try:

            sig, fs = read_channels(
                r.path_abs,
                all_ch,
            )

            for c in cfg.channels:

                if len(sig[c]) == 0:

                    raise ValueError(
                        f"missing fast channel '{c}'"
                    )

            n_av = min(
                len(sig[c])
                for c in cfg.channels
            )

            if n_av < cfg.win:

                raise ValueError(
                    f"only {n_av} samples < "
                    f"win {cfg.win}"
                )

            if (
                n_av < n_keep
                and not cfg.allow_short
            ):

                raise ValueError(
                    f"{n_av} < N_KEEP {n_keep}"
                )

            n_f = min(
                n_av,
                n_keep,
            )

            n_w = (
                1
                + (n_f - cfg.win)
                // cfg.hop
            )

            main = np.stack(
                [
                    sig[c][:n_f]
                    for c in cfg.channels
                ]
            )

            if not np.isfinite(
                main
            ).all():

                raise ValueError(
                    "non-finite samples "
                    "in fast channels"
                )

            blk = np.stack(
                [
                    windows_of(
                        main[i],
                        n_w,
                        cfg.win,
                        cfg.hop,
                    )
                    for i in range(C)
                ],
                axis=1,
            )

            fs_ref = fs.get(
                ref,
                np.nan,
            )

            if (
                not np.isfinite(fs_ref)
                or fs_ref <= 0
            ):
                fs_ref = FS_MAIN

            t_ref = (
                np.arange(
                    n_f,
                    dtype=np.float64,
                )
                / fs_ref
            )

            starts = (
                np.arange(n_w)
                * cfg.hop
            )

            t0 = t_ref[
                starts
            ]

            t1 = (
                t0
                + cfg.win / fs_ref
            )

            ops = {}

            for c in CH_SLOW:

                ops[c] = window_means(
                    sig[c],
                    time_axis(
                        c,
                        sig,
                        fs,
                        ref,
                        fs_ref,
                    ),
                    t0,
                    t1,
                )

            aux = sig.get(
                CH_AUX[0],
                np.empty(0),
            )

            temp = (
                float(np.mean(aux))
                if len(aux)
                else np.nan
            )

        except Exception as e:

            skipped.append(
                {
                    "file_id": int(
                        r.file_id
                    ),
                    "path": r.path_abs,
                    "reason": repr(e),
                    "trace": traceback.format_exc(
                        limit=1
                    ),
                }
            )

            continue

        X[
            cur:cur + n_w
        ] = blk.astype(
            cfg.np_dtype,
            copy=False,
        )

        for w in range(n_w):

            meta_rows.append(
                dict(
                    idx=cur + w,
                    file_id=int(r.file_id),
                    window=w,
                    start=int(
                        w * cfg.hop
                    ),
                    n_win_file=n_w,
                    n_samples_used=int(n_f),

                    bearing=r.bearing,
                    cond=r.cond,
                    trial=int(r.trial),
                    origin=r.origin,
                    mode=r.mode,

                    label=int(r.label),
                    label_name=r.label_name,

                    speed_rpm=float(
                        ops["speed"][w]
                    ),
                    torque_nm=float(
                        ops["torque"][w]
                    ),
                    force_n=float(
                        ops["force"][w]
                    ),
                    temp_c=temp,

                    rpm_set=(
                        float(r.rpm_set)
                        if pd.notna(
                            r.rpm_set
                        )
                        else np.nan
                    ),

                    torque_set=(
                        float(r.torque_set)
                        if pd.notna(
                            r.torque_set
                        )
                        else np.nan
                    ),

                    force_set=(
                        float(r.force_set)
                        if pd.notna(
                            r.force_set
                        )
                        else np.nan
                    ),
                )
            )

        cur += n_w

    X.flush()
    del X

    os.truncate(
        cfg.dat_path,
        cur
        * C
        * cfg.win
        * np.dtype(
            cfg.np_dtype
        ).itemsize,
    )

    meta = pd.DataFrame(
        meta_rows
    )

    assert len(meta) == cur

    meta.to_parquet(
        cfg.meta_path,
        index=False,
    )

    files.to_parquet(
        cfg.out / "files_used.parquet",
        index=False,
    )

    if skipped:

        pd.DataFrame(
            skipped
        ).to_csv(
            cfg.out
            / "export_skipped.csv",
            index=False,
        )

    info = dict(
        shape=[
            int(cur),
            C,
            cfg.win,
        ],
        dtype=cfg.dtype,
        channels=list(
            cfg.channels
        ),
        slow_channels=list(
            CH_SLOW
        ),
        aux_channels=list(
            CH_AUX
        ),
        cond_keys=list(
            COND_KEYS
        ),

        fs=FS_MAIN,

        win=cfg.win,
        hop=cfg.hop,

        n_keep=int(n_keep),
        n_keep_source=how,

        n_windows_per_file_full=int(
            n_win_full
        ),

        allow_short=cfg.allow_short,

        n_files_in=int(
            len(files)
        ),
        n_files_exported=int(
            meta.file_id.nunique()
        ),
        n_skipped=len(skipped),

        labels=LABELS4,

        seed=cfg.seed,

        reader=_READER_NAME,
        dat=cfg.dat_path.name,
    )

    cfg.export_meta_path.write_text(
        json.dumps(
            info,
            indent=2,
        )
    )

    log(
        f"\nwrote windows.dat: "
        f"{cur:,} windows, "
        f"{cur * C * cfg.win * np.dtype(cfg.np_dtype).itemsize / 1e9:.2f} GB"
    )

    log(
        f"skipped files: {len(skipped)}"
    )

    log(
        "\nwindows per class:\n"
        + str(
            meta.label_name.value_counts()
        )
    )


# =============================================================================
# 8. SPLITS
# =============================================================================

def split_bearings(
    bearings: Sequence[str],
    rng: np.random.Generator,
    frac=(0.6, 0.2, 0.2),
):

    b = list(
        dict.fromkeys(bearings)
    )

    rng.shuffle(b)

    n = len(b)

    if n >= 3:

        n_tr = max(
            1,
            int(round(
                frac[0] * n
            )),
        )

        n_va = max(
            1,
            int(round(
                frac[1] * n
            )),
        )

        n_tr = min(
            n_tr,
            n - 2,
        )

        n_va = min(
            n_va,
            n - n_tr - 1,
        )

        return (
            b[:n_tr],
            b[n_tr:n_tr + n_va],
            b[n_tr + n_va:],
        )

    if n == 2:

        return (
            b[:1],
            [],
            b[1:],
        )

    return (
        b,
        [],
        [],
    )


def build_splits(
    cfg: Cfg,
    meta: pd.DataFrame,
):

    fmeta = (
        meta
        .drop_duplicates(
            "file_id"
        )
        [
            [
                "file_id",
                "bearing",
                "cond",
                "trial",
                "origin",
                "label_name",
            ]
        ]
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(
        cfg.seed
    )

    ids = lambda mask: sorted(
        int(x)
        for x in fmeta.file_id[mask]
    )

    splits = {}

    # -------------------------------------------------------------------------
    # A. BEARING HOLDOUT
    #
    # Critical property:
    # each bearing is assigned to exactly ONE split globally.
    #
    # We therefore split the complete bearing set, not independently per
    # class. This prevents the same physical bearing from appearing in
    # train/val/test simply because it occurs with multiple labels.
    # -------------------------------------------------------------------------

    bearing_labels = (
        fmeta[
            ["bearing", "label_name"]
        ]
        .drop_duplicates()
        .groupby("bearing")
        ["label_name"]
        .agg(
            lambda x: "|".join(
                sorted(x)
            )
        )
    )

    # Stratify approximately by the most informative label:
    # use a deterministic global bearing split.
    #
    # With Paderborn's 32 bearings this gives a clean 60/20/20 split.
    all_bearings = sorted(
        bearing_labels.index
    )

    rng.shuffle(
        all_bearings
    )

    n = len(
        all_bearings
    )

    n_tr = max(
        1,
        int(round(0.60 * n)),
    )

    n_va = max(
        1,
        int(round(0.20 * n)),
    )

    n_tr = min(
        n_tr,
        n - 2,
    )

    n_va = min(
        n_va,
        n - n_tr - 1,
    )

    b_train = all_bearings[
        :n_tr
    ]

    b_val = all_bearings[
        n_tr:n_tr + n_va
    ]

    b_test = all_bearings[
        n_tr + n_va:
    ]

    splits[
        "bearing_holdout"
    ] = {
        "train": ids(
            fmeta.bearing.isin(
                b_train
            )
        ),
        "val": ids(
            fmeta.bearing.isin(
                b_val
            )
        ),
        "test": ids(
            fmeta.bearing.isin(
                b_test
            )
        ),
        "_bearings": {
            "train": b_train,
            "val": b_val,
            "test": b_test,
        },
    }

    # -------------------------------------------------------------------------
    # B. ARTIFICIAL -> REAL
    # -------------------------------------------------------------------------

    healthy = sorted(
        fmeta.bearing[
            fmeta.origin == "healthy"
        ].unique()
    )

    artificial = sorted(
        fmeta.bearing[
            fmeta.origin == "artificial"
        ].unique()
    )

    real = sorted(
        fmeta.bearing[
            fmeta.origin == "real"
        ].unique()
    )

    rng.shuffle(
        healthy
    )

    rng.shuffle(
        artificial
    )

    # Healthy bearings: train / val
    h_tr, h_va, h_te = split_bearings(
        healthy,
        rng,
    )

    # Artificial damaged bearings:
    # train/val only.
    n_a_val = (
        max(
            1,
            int(round(
                0.15
                * len(artificial)
            )),
        )
        if len(artificial) >= 4
        else 0
    )

    a_val = artificial[
        :n_a_val
    ]

    a_train = artificial[
        n_a_val:
    ]

    # Real damaged bearings are TEST only.
    splits[
        "artificial_to_real"
    ] = {
        "train": ids(
            fmeta.bearing.isin(
                a_train + h_tr
            )
        ),
        "val": ids(
            fmeta.bearing.isin(
                a_val + h_va
            )
        ),
        "test": ids(
            fmeta.bearing.isin(
                real + h_te
            )
        ),
        "_bearings": {
            "train": a_train + h_tr,
            "val": a_val + h_va,
            "test": real + h_te,
        },
    }

    # -------------------------------------------------------------------------
    # C. CROSS-CONDITION
    #
    # Test condition is held out entirely.
    # Same bearings may occur in train/test here because this is a
    # condition-generalization experiment rather than a bearing-generalization
    # experiment.
    # -------------------------------------------------------------------------

    conds = sorted(
        fmeta.cond.unique()
    )

    test_cond = (
        cfg.test_cond
        or (
            "N09_M07_F10"
            if "N09_M07_F10" in conds
            else conds[-1]
        )
    )

    assert test_cond in conds, (
        f"--test-cond {test_cond!r} "
        f"not in {conds}"
    )

    in_test = (
        fmeta.cond
        == test_cond
    )

    # Validation is selected from trials in the remaining conditions.
    trials = sorted(
        fmeta.trial[
            ~in_test
        ].unique()
    )

    n_val_trials = (
        max(
            1,
            len(trials) // 10,
        )
        if len(trials) > 2
        else 0
    )

    val_trials = set(
        trials[
            -n_val_trials:
        ]
    ) if n_val_trials else set()

    splits[
        "cross_condition"
    ] = {
        "train": ids(
            ~in_test
            & ~fmeta.trial.isin(
                val_trials
            )
        ),
        "val": ids(
            ~in_test
            & fmeta.trial.isin(
                val_trials
            )
        ),
        "test": ids(
            in_test
        ),
        "_test_cond": test_cond,
        "_val_trials": sorted(
            val_trials
        ),
    }

    return splits


def load_or_build_splits(
    cfg: Cfg,
    meta: pd.DataFrame,
):

    # IMPORTANT:
    # We deliberately ignore:
    #     cfg.eda / splits.json
    #
    # and also ignore any previous pt_export/splits_export.json.
    #
    # This makes the exporter deterministic and prevents stale artifacts
    # from silently changing the experiment.

    log(
        "splits: deriving fresh splits "
        "from current exported metadata"
    )

    splits = build_splits(
        cfg,
        meta,
    )

    cfg.splits_path.write_text(
        json.dumps(
            splits,
            indent=2,
        )
    )

    log(
        f"splits derived -> "
        f"{cfg.splits_path}"
    )

    return splits


def stage_splits(
    cfg: Cfg,
):

    meta = pd.read_parquet(
        cfg.meta_path
    )

    splits = load_or_build_splits(
        cfg,
        meta,
    )

    rows = []

    for strat, d in splits.items():

        for part in (
            "train",
            "val",
            "test",
        ):

            f = d.get(
                part,
                [],
            )

            m = meta[
                meta.file_id.isin(f)
            ]

            rows.append(
                dict(
                    strategy=strat,
                    split=part,
                    files=len(f),
                    windows=len(m),
                    bearings=m.bearing.nunique(),
                    conds=m.cond.nunique(),
                    classes=m.label_name.nunique(),
                )
            )

    log(
        "\n"
        + pd.DataFrame(rows)
        .to_string(index=False)
    )

    check_leakage(
        meta,
        splits,
    )


def check_leakage(
    meta: pd.DataFrame,
    splits,
):

    for strat, d in splits.items():

        parts = {
            p: set(
                d.get(
                    p,
                    [],
                )
            )
            for p in (
                "train",
                "val",
                "test",
            )
        }

        # File-level leakage is NEVER allowed.
        for a, b in (
            ("train", "val"),
            ("train", "test"),
            ("val", "test"),
        ):

            inter = (
                parts[a]
                & parts[b]
            )

            assert not inter, (
                f"{strat}: file_id overlap "
                f"{a}/{b}: "
                f"{sorted(inter)[:10]}"
            )

        # Bearing-level leakage is forbidden for bearing holdout
        # and artificial-to-real.
        bset = {
            p: set(
                meta.bearing[
                    meta.file_id.isin(ids)
                ]
            )
            for p, ids in parts.items()
        }

        if strat in (
            "bearing_holdout",
            "artificial_to_real",
        ):

            for a, b in (
                ("train", "val"),
                ("train", "test"),
                ("val", "test"),
            ):

                inter = (
                    bset[a]
                    & bset[b]
                )

                assert not inter, (
                    f"{strat}: bearing overlap "
                    f"{a}/{b}: "
                    f"{sorted(inter)}"
                )

        # Cross-condition:
        # bearing overlap is allowed,
        # condition overlap is NOT.
        if strat == "cross_condition":

            ctr = set(
                meta.cond[
                    meta.file_id.isin(
                        parts["train"]
                    )
                ]
            )

            cte = set(
                meta.cond[
                    meta.file_id.isin(
                        parts["test"]
                    )
                ]
            )

            assert not (
                ctr & cte
            ), (
                "cross_condition: "
                "condition overlap "
                f"{sorted(ctr & cte)}"
            )

        if strat == "artificial_to_real":

            o_te = set(
                meta.origin[
                    meta.file_id.isin(
                        parts["test"]
                    )
                ]
            )

            assert (
                "artificial"
                not in o_te
            ), (
                "artificial_to_real: "
                "artificial bearings leaked "
                "into test"
            )

        log(
            f"leakage OK: {strat}"
        )


# =============================================================================
# 9. NORMALIZATION
# =============================================================================

def open_memmap(
    cfg: Cfg,
):

    info = json.loads(
        cfg.export_meta_path.read_text()
    )

    X = np.memmap(
        cfg.dat_path,
        mode="r",
        dtype=np.dtype(
            info["dtype"]
        ),
        shape=tuple(
            info["shape"]
        ),
    )

    return X, info


def stage_norm(
    cfg: Cfg,
):

    meta = pd.read_parquet(
        cfg.meta_path
    )

    splits = load_or_build_splits(
        cfg,
        meta,
    )

    X, info = open_memmap(
        cfg
    )

    C = info["shape"][1]

    for strat, d in splits.items():

        idx = np.asarray(
            meta.index[
                meta.file_id.isin(
                    d.get(
                        "train",
                        [],
                    )
                )
            ],
            dtype=np.int64,
        )

        assert len(idx), (
            f"{strat}: empty train split"
        )

        n = np.zeros(
            C,
            dtype=np.float64,
        )

        s = np.zeros_like(n)
        ss = np.zeros_like(n)

        chunks = np.array_split(
            idx,
            max(
                1,
                len(idx) // 4096,
            ),
        )

        for chunk in chunks:

            if len(chunk) == 0:
                continue

            b = np.asarray(
                X[
                    np.sort(chunk)
                ],
                dtype=np.float64,
            )

            n += (
                b.shape[0]
                * b.shape[2]
            )

            s += b.sum(
                axis=(0, 2)
            )

            ss += (
                b ** 2
            ).sum(
                axis=(0, 2)
            )

        mean = s / n

        std = np.sqrt(
            np.maximum(
                ss / n
                - mean ** 2,
                1e-12,
            )
        )

        cm = meta.loc[
            idx,
            list(COND_KEYS),
        ].to_numpy(
            np.float64
        )

        cmean = np.nanmean(
            cm,
            axis=0,
        )

        cstd = np.nanstd(
            cm,
            axis=0,
        )

        cstd[
            ~np.isfinite(cstd)
            | (cstd < 1e-9)
        ] = 1.0

        np.savez(
            cfg.norm_path(strat),

            mean=mean.astype(
                np.float32
            ),

            std=std.astype(
                np.float32
            ),

            cond_mean=cmean.astype(
                np.float32
            ),

            cond_std=cstd.astype(
                np.float32
            ),

            n_windows=len(idx),

            channels=np.array(
                info["channels"]
            ),

            cond_keys=np.array(
                list(COND_KEYS)
            ),
        )

        log(
            f"{strat}: "
            f"train windows={len(idx)} "
            f"mean={np.round(mean, 5)} "
            f"std={np.round(std, 5)}"
        )

    del X


# =============================================================================
# 10. PYTORCH
# =============================================================================

def require_torch():

    try:

        import torch

        return True

    except Exception as e:

        log(
            f"torch unavailable ({e!r})"
        )

        return False


def build_torch_bits():

    import torch
    from torch import nn
    from torch.utils.data import (
        Dataset,
        DataLoader,
        WeightedRandomSampler,
    )

    class WindowDataset(
        Dataset
    ):

        def __init__(
            self,
            dat,
            shape,
            dtype,
            idx,
            labels,
            cond,
            mean,
            std,
            cond_mean,
            cond_std,
            augment=False,
            shift=0.0625,
            noise=0.01,
            seed=0,
        ):

            self.dat = str(dat)
            self.shape = tuple(shape)
            self.dtype = np.dtype(
                dtype
            )

            self.idx = np.asarray(
                idx,
                dtype=np.int64,
            )

            self.labels = np.asarray(
                labels,
                dtype=np.int64,
            )

            self.cond = np.nan_to_num(
                np.asarray(
                    cond,
                    dtype=np.float32,
                )
            )

            self.mean = np.asarray(
                mean,
                np.float32,
            )[:, None]

            self.std = np.asarray(
                std,
                np.float32,
            )[:, None]

            self.cond_mean = np.asarray(
                cond_mean,
                np.float32,
            )

            self.cond_std = np.asarray(
                cond_std,
                np.float32,
            )

            self.augment = augment
            self.shift = shift
            self.noise = noise
            self.seed = seed

            self._X = None

        def __len__(self):

            return len(
                self.idx
            )

        @property
        def X(self):

            if self._X is None:

                self._X = np.memmap(
                    self.dat,
                    mode="r",
                    dtype=self.dtype,
                    shape=self.shape,
                )

            return self._X

        def __getitem__(
            self,
            i,
        ):

            j = int(
                self.idx[i]
            )

            x = np.asarray(
                self.X[j],
                dtype=np.float32,
            )

            if self.augment:

                rng = np.random.default_rng(
                    (
                        self.seed
                        * 1_000_003
                        + j
                    )
                    & 0x7FFFFFFF
                )

                if self.shift:

                    max_shift = int(
                        self.shift
                        * x.shape[-1]
                    )

                    if max_shift:

                        x = np.roll(
                            x,
                            int(
                                rng.integers(
                                    -max_shift,
                                    max_shift + 1,
                                )
                            ),
                            axis=-1,
                        )

                if self.noise:

                    x = (
                        x
                        + rng.normal(
                            0.0,
                            self.noise,
                            x.shape,
                        ).astype(
                            np.float32
                        )
                        * self.std
                    )

            x = (
                x - self.mean
            ) / self.std

            c = (
                self.cond[i]
                - self.cond_mean
            ) / self.cond_std

            return (
                torch.from_numpy(
                    np.ascontiguousarray(x)
                ),
                torch.from_numpy(
                    np.ascontiguousarray(c)
                ),
                torch.tensor(
                    int(self.labels[i])
                ),
            )

    class CondCNN(
        nn.Module
    ):

        def __init__(
            self,
            in_ch,
            n_cls=4,
            n_cond=4,
            width=32,
        ):

            super().__init__()

            def blk(
                i,
                o,
                k=7,
                s=4,
            ):

                return nn.Sequential(
                    nn.Conv1d(
                        i,
                        o,
                        k,
                        s,
                        k // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(o),
                    nn.ReLU(
                        inplace=True
                    ),
                )

            self.body = nn.Sequential(
                blk(
                    in_ch,
                    width,
                    k=65,
                    s=4,
                ),

                blk(
                    width,
                    width * 2,
                    k=7,
                    s=2,
                ),

                nn.MaxPool1d(2),

                blk(
                    width * 2,
                    width * 4,
                    k=7,
                    s=2,
                ),

                nn.MaxPool1d(2),

                blk(
                    width * 4,
                    width * 8,
                    k=7,
                    s=2,
                ),

                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )

            self.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(
                    width * 8
                    + n_cond,
                    n_cls,
                ),
            )

        def forward(
            self,
            x,
            c,
        ):

            z = self.body(x)

            return self.head(
                torch.cat(
                    [z, c],
                    dim=1,
                )
            )

    return (
        torch,
        nn,
        Dataset,
        DataLoader,
        WeightedRandomSampler,
        WindowDataset,
        CondCNN,
    )


def make_loaders(
    cfg: Cfg,
    strategy: str,
):

    (
        torch,
        nn,
        _,
        DataLoader,
        WeightedRandomSampler,
        WindowDataset,
        CondCNN,
    ) = build_torch_bits()

    meta = pd.read_parquet(
        cfg.meta_path
    )

    splits = load_or_build_splits(
        cfg,
        meta,
    )

    assert strategy in splits, (
        f"unknown strategy {strategy!r}; "
        f"have {list(splits)}"
    )

    info = json.loads(
        cfg.export_meta_path.read_text()
    )

    npz_path = cfg.norm_path(
        strategy
    )

    assert npz_path.exists(), (
        f"missing {npz_path.name}; "
        "run --stage norm first"
    )

    z = np.load(
        npz_path,
        allow_pickle=False,
    )

    loaders = {}
    datasets = {}

    for part in (
        "train",
        "val",
        "test",
    ):

        sel = meta.file_id.isin(
            splits[strategy].get(
                part,
                [],
            )
        )

        idx = np.asarray(
            meta.index[sel],
            dtype=np.int64,
        )

        if not len(idx):

            log(
                f"{strategy}/{part}: empty"
            )

            continue

        ds = WindowDataset(
            cfg.dat_path,
            info["shape"],
            info["dtype"],
            idx,

            meta.loc[
                idx,
                "label",
            ].to_numpy(),

            meta.loc[
                idx,
                list(COND_KEYS),
            ].to_numpy(),

            z["mean"],
            z["std"],
            z["cond_mean"],
            z["cond_std"],

            augment=(
                part == "train"
                and cfg.augment
            ),

            seed=cfg.seed,
        )

        if part == "train":

            y = meta.loc[
                idx,
                "label",
            ].to_numpy()

            cnt = np.bincount(
                y,
                minlength=len(
                    LABELS4
                ),
            ).astype(
                np.float64
            )

            weights = (
                1.0
                / np.maximum(
                    cnt,
                    1,
                )
            )[y]

            sampler = (
                WeightedRandomSampler(
                    torch.as_tensor(
                        weights,
                        dtype=torch.double,
                    ),
                    num_samples=len(idx),
                    replacement=True,
                )
            )

            dl = DataLoader(
                ds,
                batch_size=cfg.batch_size,
                sampler=sampler,
                num_workers=cfg.workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
                persistent_workers=(
                    cfg.workers > 0
                ),
            )

        else:

            dl = DataLoader(
                ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=(
                    cfg.workers > 0
                ),
            )

        loaders[part] = dl
        datasets[part] = ds

        log(
            f"{strategy}/{part}: "
            f"{len(idx):,} windows, "
            f"{meta.file_id[sel].nunique()} files"
        )

    return (
        loaders,
        datasets,
        meta,
        splits,
        info,
    )


# =============================================================================
# 11. SMOKE TEST
# =============================================================================

def stage_smoke(
    cfg: Cfg,
):

    if not require_torch():
        return

    (
        torch,
        nn,
        _,
        _,
        _,
        _,
        CondCNN,
    ) = build_torch_bits()

    (
        loaders,
        datasets,
        meta,
        splits,
        info,
    ) = make_loaders(
        cfg,
        cfg.strategy,
    )

    assert "train" in loaders

    torch.manual_seed(
        cfg.seed
    )

    dev = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = CondCNN(
        in_ch=info["shape"][1],
        n_cls=len(LABELS4),
        n_cond=len(COND_KEYS),
    ).to(dev)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    crit = nn.CrossEntropyLoss(
        label_smoothing=0.05
    )

    log(
        f"device={dev} "
        f"params="
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.4f}M"
    )

    xb, cb, yb = next(
        iter(
            loaders["train"]
        )
    )

    assert xb.shape[1:] == (
        info["shape"][1],
        cfg.win,
    )

    assert torch.isfinite(
        xb
    ).all()

    assert torch.isfinite(
        cb
    ).all()

    log(
        f"batch x={tuple(xb.shape)} "
        f"cond={tuple(cb.shape)} "
        f"y={tuple(yb.shape)} "
        f"x.mean={xb.mean():.3f} "
        f"x.std={xb.std():.3f}"
    )

    model.train()

    losses = []

    iterator = iter(
        loaders["train"]
    )

    for step in range(
        cfg.smoke_steps
    ):

        try:

            xb, cb, yb = next(
                iterator
            )

        except StopIteration:

            iterator = iter(
                loaders["train"]
            )

            xb, cb, yb = next(
                iterator
            )

        xb = xb.to(
            dev,
            non_blocking=True,
        )

        cb = cb.to(
            dev,
            non_blocking=True,
        )

        yb = yb.to(
            dev,
            non_blocking=True,
        )

        opt.zero_grad(
            set_to_none=True
        )

        out = model(
            xb,
            cb,
        )

        loss = crit(
            out,
            yb,
        )

        loss.backward()
        opt.step()

        losses.append(
            float(loss.item())
        )

        pred = out.argmax(
            dim=1
        )

        acc = (
            pred == yb
        ).float().mean().item()

        log(
            f"step {step:3d} "
            f"loss {loss.item():.4f} "
            f"acc {acc:.3f}"
        )

    assert np.isfinite(
        losses
    ).all()

    first = float(
        np.mean(
            losses[:min(5, len(losses))]
        )
    )

    last = float(
        np.mean(
            losses[
                -min(5, len(losses)):
            ]
        )
    )

    log(
        f"loss {first:.4f} -> {last:.4f}"
    )

    if last >= first:

        warnings.warn(
            "loss did not decrease "
            "during smoke test"
        )

    log(
        "SMOKE TEST OK"
    )


# =============================================================================
# 12. VERIFY
# =============================================================================

def stage_verify(
    cfg: Cfg,
):

    info = json.loads(
        cfg.export_meta_path.read_text()
    )

    meta = pd.read_parquet(
        cfg.meta_path
    )

    X, _ = open_memmap(
        cfg
    )

    assert len(meta) == (
        info["shape"][0]
    )

    assert (
        meta.idx.to_numpy()
        == np.arange(
            len(meta)
        )
    ).all()

    expected_bytes = (
        int(
            np.prod(
                info["shape"]
            )
        )
        * np.dtype(
            info["dtype"]
        ).itemsize
    )

    assert (
        cfg.dat_path.stat().st_size
        == expected_bytes
    )

    count = (
        meta
        .groupby("file_id")
        .size()
    )

    assert count.between(
        1,
        info[
            "n_windows_per_file_full"
        ],
    ).all()

    log(
        f"files={len(count)} "
        f"windows/file="
        f"{count.min()}-{count.max()} "
        f"short-of-N_KEEP="
        f"{int((count < info['n_windows_per_file_full']).sum())}"
    )

    for k in COND_KEYS:

        bad = int(
            (
                ~np.isfinite(
                    meta[k]
                )
            ).sum()
        )

        log(
            f"{k}: "
            f"[{np.nanmin(meta[k]):.2f}, "
            f"{np.nanmax(meta[k]):.2f}] "
            f"non-finite={bad}"
        )

        assert bad == 0

    assert meta.speed_rpm.between(
        0,
        5000,
    ).all()

    assert (
        "torque_set"
        in meta.columns
    )

    assert (
        "torque_nm"
        in meta.columns
    )

    rng = np.random.default_rng(
        cfg.seed
    )

    probe = rng.choice(
        len(meta),
        size=min(
            256,
            len(meta),
        ),
        replace=False,
    )

    blk = np.asarray(
        X[
            np.sort(probe)
        ],
        dtype=np.float32,
    )

    assert np.isfinite(
        blk
    ).all()

    assert (
        blk.std(
            axis=(0, 2)
        ) > 0
    ).all()

    log(
        "per-channel std="
        + str(
            np.round(
                blk.std(
                    axis=(0, 2)
                ),
                5,
            )
        )
    )

    log(
        "\nlabel distribution:\n"
        + str(
            meta.label_name.value_counts()
        )
    )

    log(
        "\nlabel x condition:\n"
        + str(
            meta.pivot_table(
                index="label_name",
                columns="cond",
                aggfunc="size",
                fill_value=0,
            )
        )
    )

    splits = load_or_build_splits(
        cfg,
        meta,
    )

    check_leakage(
        meta,
        splits,
    )

    del X

    log(
        "\nVERIFY OK"
    )


# =============================================================================
# 13. CLI
# =============================================================================

def parse_args(
    argv=None,
):

    p = argparse.ArgumentParser(
        description=(
            "Paderborn -> "
            "PyTorch window export"
        )
    )

    p.add_argument(
        "--stage",
        default="all",
        choices=[
            "inventory",
            "export",
            "splits",
            "norm",
            "smoke",
            "verify",
            "all",
        ],
    )

    p.add_argument(
        "--root",
        default="../data_paderborn",
    )

    p.add_argument(
        "--eda",
        default=None,
    )

    p.add_argument(
        "--out",
        default=None,
    )

    # Aligned with notebook
    p.add_argument(
        "--win",
        type=int,
        default=4096,
    )

    p.add_argument(
        "--hop",
        type=int,
        default=4096,
    )

    p.add_argument(
        "--dtype",
        default="float32",
        choices=[
            "float16",
            "float32",
        ],
    )

    p.add_argument(
        "--channels",
        nargs="+",
        default=list(
            CH_MAIN
        ),
    )

    p.add_argument(
        "--no-allow-short",
        dest="allow_short",
        action="store_false",
    )

    p.add_argument(
        "--max-files",
        type=int,
        default=0,
    )

    p.add_argument(
        "--max-gb",
        type=float,
        default=24.0,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    p.add_argument(
        "--test-cond",
        default="",
    )

    p.add_argument(
        "--strategy",
        default="bearing_holdout",
        choices=[
            "bearing_holdout",
            "artificial_to_real",
            "cross_condition",
        ],
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    p.add_argument(
        "--smoke-steps",
        type=int,
        default=10,
    )

    # Kept only for CLI compatibility.
    # It is intentionally ignored internally.
    p.add_argument(
        "--splits-source",
        default="derive",
        choices=[
            "auto",
            "artifact",
            "derive",
        ],
    )

    p.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
    )

    a = p.parse_args(
        argv
    )

    cfg = Cfg(
        root=a.root,
        eda=a.eda,
        out=a.out,

        win=a.win,
        hop=a.hop,
        dtype=a.dtype,

        channels=tuple(
            a.channels
        ),

        allow_short=a.allow_short,

        max_files=a.max_files,
        max_gb=a.max_gb,

        seed=a.seed,

        test_cond=a.test_cond,

        strategy=a.strategy,

        batch_size=a.batch_size,
        workers=a.workers,
        smoke_steps=a.smoke_steps,

        # Always derive.
        splits_source="derive",

        augment=a.augment,
    )

    return cfg, a.stage


# =============================================================================
# 14. MAIN
# =============================================================================

def main(
    argv=None,
):

    cfg, stage = parse_args(
        argv
    )

    log(
        f"root={cfg.root.resolve()}\n"
        f"eda ={cfg.eda.resolve()}\n"
        f"out ={cfg.out.resolve()}"
    )

    log(
        json.dumps(
            {
                k: (
                    str(v)
                    if isinstance(
                        v,
                        Path,
                    )
                    else v
                )
                for k, v
                in asdict(cfg).items()
            },
            indent=2,
            default=str,
        )
    )

    if stage == "all":

        order = [
            "export",
            "splits",
            "norm",
            "smoke",
            "verify",
        ]

    else:

        order = [
            stage
        ]

    for s in order:

        log(
            f"\n{'=' * 78}\n"
            f"== {s.upper()}\n"
            f"{'=' * 78}"
        )

        if s == "inventory":

            init_reader(
                cfg
            )

            load_inventory(
                cfg
            )

        elif s == "export":

            stage_export(
                cfg
            )

        elif s == "splits":

            stage_splits(
                cfg
            )

        elif s == "norm":

            stage_norm(
                cfg
            )

        elif s == "smoke":

            stage_smoke(
                cfg
            )

        elif s == "verify":

            stage_verify(
                cfg
            )

    log(
        "\nDONE"
    )

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )

