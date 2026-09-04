#!/usr/bin/env python3
"""
fh_paderborn.py -- Paderborn (KAt-DataCenter) bearing fault classification.

Leakage-aware splits (incl. bearing-grouped stratified k-fold), per-class
recall/precision/F1 + collapse detection, and an LRP-friendly BatchNorm-free
1D CNN (pure nn.Sequential, ReLU, AvgPool -> works with zennit out of the box).

deps: numpy scipy torch        optional: zennit  (for --mode lrp)

  python fh_paderborn.py --mode inspect --data-root DATA
  python fh_paderborn.py --mode splits  --data-root DATA --split bearing_cv
  python fh_paderborn.py --mode train   --data-root DATA --split bearing_cv
  python fh_paderborn.py --mode lrp     --data-root DATA --ckpt runs/x/fold0.pt
"""
from __future__ import annotations

import argparse, hashlib, json, time, warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

'''
===============================================================================
 RUNBOOK -- run top to bottom. Steps 0-1 cost seconds. L0-L5 are the experiments.
 DATA = ../data_paderborn      (swap for your path)
 Read GOTCHAS at the bottom before changing any flag.
===============================================================================

--- 0. INVENTORY / SANITY (parses filenames + opens ONE .mat) -----------------
python fh_paderborn.py --mode inspect --data-root ../data_paderborn --task 3class --damage-source real
python fh_paderborn.py --mode inspect --data-root ../data_paderborn --task code   --damage-source real
python fh_paderborn.py --mode inspect --data-root ../data_paderborn --task 3class --damage-source all --conditions all
# READ: "[inventory] max usable --folds for bearing_cv = K"  -> never pass --folds > K
#       channel list must contain vibration_1; window shown in ms and shaft revs.

--- 1. SPLIT PREVIEW (asserts the protocol before burning GPU) ----------------
python fh_paderborn.py --mode splits --data-root ../data_paderborn --task 3class --damage-source real --split file
python fh_paderborn.py --mode splits --data-root ../data_paderborn --task 3class --damage-source real --split bearing_cv --folds 5
python fh_paderborn.py --mode splits --data-root ../data_paderborn --task 3class --damage-source all  --split art2real
python fh_paderborn.py --mode splits --data-root ../data_paderborn --task 3class --damage-source real --conditions all --split loco
python fh_paderborn.py --mode splits --data-root ../data_paderborn --task 3class --damage-source real --conditions all --split loco_strict
# READ: "[ok] bearings fully disjoint"          -> bearing / bearing_cv / art2real / loco_strict
#       "[warn] FILE split ... CEILING only"    -> file        (expected, that is the point)
#       "[warn] ... recur train<->test BY DESIGN"-> loco       (expected: condition transfer only)

--- L0  LEAKAGE PROBE: 20-way bearing ID ------------------------------------
# Purpose: prove each bearing has a recognisable fingerprint, so any split that
# reuses bearings is measuring memorisation. High acc here = the L1/L2 gap is real.
python fh_paderborn.py --data-root ../data_paderborn --task code --damage-source real  --split file --window 2048 --stride 2048 --max-windows-per-file 32  --epochs 20 --out runs/L0_fingerprint

--- L1  CEILING: file split, 3 classes --------------------------------------
# Same window/stride/max-windows as L2 on purpose: the L1 -> L2 drop is then
# attributable to the SPLIT alone, not to a change in input length.
python fh_paderborn.py --data-root ../data_paderborn --task 3class --damage-source real  --split file --window 8192 --stride 4096 --max-windows-per-file 24  --epochs 40 --out runs/L1_ceiling

--- L2  HEADLINE: bearing-grouped stratified 5-fold -------------------------
# Unseen bearings. This is the number that goes in the paper/report.
# NOTE: runs/L2_bearing_cv is the OLD broken-selection run -- do not cite it.
python fh_paderborn.py --data-root ../data_paderborn --task 3class --damage-source real  --split bearing_cv --folds 5 --window 8192 --stride 4096 --max-windows-per-file 24  --epochs 40 --class-weights --monitor bal_acc --min-epochs 15 --val-smooth 3  --out runs/L2_v3
# CHECK: no fold prints "[select] ... epoch 1"; every fold reaches ep >= 15 before
#        any "[early stop]"; cvfold1 val now well BELOW its old 0.9785 (fix working).

--- L3  leave-one-condition-out (bearings recur BY DESIGN) -------------------
python fh_paderborn.py --data-root ../data_paderborn --task 3class --damage-source real  --conditions all --split loco --seeds 0 --window 8192 --max-windows-per-file 12  --epochs 40 --out runs/L3_loco

--- L4  artificial -> real damage transfer ----------------------------------
python fh_paderborn.py --data-root ../data_paderborn --task 3class --damage-source all  --split art2real --seeds 0 1 2 --window 8192 --max-windows-per-file 24  --epochs 40 --class-weights --out runs/L4_art2real

--- L5  STRICT: bearing- AND condition-disjoint -----------------------------
# Hardest protocol. 3 bearing draws x 4 held-out conditions = 12 runs.
python fh_paderborn.py --data-root ../data_paderborn --task 3class --damage-source real  --conditions all --split loco_strict --seeds 0 1 2 --window 8192  --max-windows-per-file 12 --epochs 40 --class-weights --out runs/L5_strict

===============================================================================
 L2 FOLLOW-UPS (R-series). Same split as L2, one variable changed each.
===============================================================================

--- R1  L2 + normalisation + regularisation ---------------------------------
# --norm group is the ONLY norm this model accepts (see GOTCHAS). --dropout 0.3
# is already the default; kept explicit so the ablation is self-documenting.
python fh_paderborn.py --data-root ../data_paderborn --task 3class --damage-source real  --split bearing_cv --folds 5 --window 8192 --stride 4096 --max-windows-per-file 24  --epochs 40 --class-weights --monitor bal_acc --min-epochs 15 --val-smooth 3  --norm group --dropout 0.3 --out runs/R1_L2_groupnorm

--- R2  FIXED-BUDGET HEADLINE: no val selection -----------------------------
# 1 val bearing cannot select a model honestly, so: fold val into train, train a
# fixed 25 epochs, report the final epoch. No selection => no optimistic bias.
python fh_paderborn.py --data-root ../data_paderborn --task 3class --damage-source real  --split bearing_cv --folds 5 --window 8192 --stride 4096 --max-windows-per-file 24  --epochs 25 --no-val --class-weights --norm group --out runs/R2_L2_noval


# CHECK: "[no-val] val folded into train -> 12 train bearings" on cvfold0 and 14
#        on the rest. Unequal counts are np.array_split on unequal class sizes,
#        NOT a bug (17 real bearings: 6 healthy / 5 outer / 6 inner).

--- R3  BINARY detection only (the part that transfers) ---------------------
# Use as the positive result: healthy vs damaged generalises to unseen bearings
# even when outer-vs-inner does not.
python fh_paderborn.py --data-root ../data_paderborn --task binary --damage-source real  --split bearing_cv --folds 5 --window 8192 --stride 4096 --max-windows-per-file 24  --epochs 25 --no-val --norm group --out runs/R3_L2_binary

--- R4  augmentation ablation (does noise/polarity buy unseen-bearing acc?) --
python fh_paderborn.py --data-root ../data_paderborn --task 3class --damage-source real  --split bearing_cv --folds 5 --window 8192 --stride 4096 --max-windows-per-file 24  --epochs 40 --class-weights --monitor bal_acc --min-epochs 15 --val-smooth 3  --aug-noise 0.05 --aug-flip --out runs/R4_L2_aug

===============================================================================
 LRP -- run on a checkpoint trained with --norm none
===============================================================================
python fh_paderborn.py --mode lrp --data-root ../data_paderborn --task 3class  --damage-source real --window 8192 --stride 4096 --max-windows-per-file 24  --ckpt runs/L2_v3/cvfold0.pt --lrp-rule epsilon_plus_flat  --lrp-n 128 --out runs/L2_v3/lrp
# CHECK: "relevance sum/logit ratio (conservation check)" should sit near 1.0.

===============================================================================
 GOTCHAS -- each of these has already cost me a run
===============================================================================
1. --norm batch DOES NOT EXIST. Choices are none|group. BatchNorm is omitted on
   purpose: it breaks clean LRP. GroupNorm still needs a zennit canonizer, so any
   checkpoint destined for --mode lrp must be trained with --norm none (default).
2. ONE --out PER RUN. Two runs sharing --out silently overwrite results.json.
3. --no-val folds val windows into train INSIDE train_eval, so report_split still
   prints a "val" row with bearing names. Those bearings ARE trained on. The
   "[no-val] val folded into train -> N" line is the truth; the val row is
   bookkeeping only.
4. Selection = mean of last --val-smooth val scores, and nothing is checkpointed
   before --min-epochs. Never trust a "[select] ... epoch 1" line; if you see one,
   the guard is not active.
5. --folds must be <= the "max usable --folds" printed by --mode inspect.
6. For bearing_cv only seeds[0] is used (it fixes the bearing permutation); the
   folds come from --folds, not from extra --seeds. Extra seeds are ignored here.
   For file / bearing / art2real / loco* every --seeds value is a separate run.
7. Window size is part of the cache key: 2048 (L0) and 8192 (L1-L5) cache to
   different .npz files, so L0 does not invalidate the L2 cache. Changing
   --channel / --conditions / --max-windows-per-file also re-windows from scratch.
8. Compare L1 vs L2 ONLY when window/stride/max-windows match. They do above.
   Keep it that way or the gap becomes uninterpretable.
9. art2real needs --damage-source all; loco/loco_strict need --conditions all.
10. Always read the per-class recall block, not just accuracy. "!! COLLAPSE" or
    "!! classes NEVER recalled" means the headline accuracy is meaningless.

'''

# ============================ 1. DATASET METADATA ============================
FS = 64_000  # Hz, vibration channel

HEALTHY = ["K001", "K002", "K003", "K004", "K005", "K006"]

# artificial damage (EDM / engraver / drilling)
ART = {"KA01": "outer", "KA03": "outer", "KA05": "outer", "KA06": "outer",
       "KA07": "outer", "KA08": "outer", "KA09": "outer",
       "KI01": "inner", "KI03": "inner", "KI05": "inner", "KI07": "inner",
       "KI08": "inner"}

# real damage from accelerated lifetime tests
REAL = {"KA04": "outer", "KA15": "outer", "KA16": "outer", "KA22": "outer",
        "KA30": "outer",
        "KI04": "inner", "KI14": "inner", "KI16": "inner", "KI17": "inner",
        "KI18": "inner", "KI21": "inner",
        "KB23": "both", "KB24": "both", "KB27": "both"}

CONDITIONS = ["N09_M07_F10", "N15_M01_F10", "N15_M07_F04", "N15_M07_F10"]

TASK_CLASSES = {
    "binary": ["healthy", "damaged"],
    "3class": ["healthy", "outer", "inner"],
    "4class": ["healthy", "outer", "inner", "both"],
}


def origin_of(code: str) -> str:
    if code in HEALTHY: return "healthy"
    if code in ART:     return "artificial"
    if code in REAL:    return "real"
    return "unknown"


def damage_of(code: str) -> str:
    if code in HEALTHY: return "none"
    return ART.get(code) or REAL.get(code) or "unknown"


def label_of(code: str, task: str):
    """-> class index, or None if this bearing is not part of the task."""
    dmg = damage_of(code)
    if dmg == "unknown":
        return None
    if task == "binary":
        return 0 if dmg == "none" else 1
    if task == "3class":
        return {"none": 0, "outer": 1, "inner": 2}.get(dmg)   # 'both' -> None
    if task == "4class":
        return {"none": 0, "outer": 1, "inner": 2, "both": 3}[dmg]
    raise ValueError(task)


# ============================== 2. FILE INDEX ================================
@dataclass
class FileRec:
    path: Path
    stem: str
    cond: str
    code: str
    trial: int
    origin: str
    label: int


def build_index(args):
    """Parse filenames only -- no MAT loading. Fast, so --mode splits is instant."""
    root = Path(args.data_root)
    if not root.exists():
        raise SystemExit(f"[data] --data-root not found: {root}")

    want_conds = None if args.conditions == ["all"] else set(args.conditions)
    recs, skipped = [], {}

    for p in sorted(root.rglob("*.mat")):
        parts = p.stem.split("_")
        if len(parts) < 5:
            skipped["bad_name"] = skipped.get("bad_name", 0) + 1
            continue
        cond, code = "_".join(parts[:3]), parts[3]
        try:
            trial = int(parts[4])
        except ValueError:
            skipped["bad_trial"] = skipped.get("bad_trial", 0) + 1
            continue

        if want_conds and cond not in want_conds:
            skipped["cond"] = skipped.get("cond", 0) + 1
            continue

        origin = origin_of(code)
        if origin == "unknown":
            skipped["unknown_code"] = skipped.get("unknown_code", 0) + 1
            continue
        if args.damage_source != "all" and origin not in ("healthy", args.damage_source):
            skipped["damage_source"] = skipped.get("damage_source", 0) + 1
            continue

        if args.task == "code":
            lab = -1  # filled in below once the code vocabulary is known
        else:
            lab = label_of(code, args.task)
            if lab is None:
                skipped["not_in_task"] = skipped.get("not_in_task", 0) + 1
                continue

        recs.append(FileRec(p, p.stem, cond, code, trial, origin, lab))

    if not recs:
        raise SystemExit("[data] no usable .mat files -- check --data-root / "
                         "--task / --damage-source / --conditions")

    if args.task == "code":
        codes = sorted({r.code for r in recs})
        cmap = {c: i for i, c in enumerate(codes)}
        for r in recs:
            r.label = cmap[r.code]
        class_names = codes
    else:
        class_names = TASK_CLASSES[args.task]

    print(f"[data] {len(recs)} files | {len({r.code for r in recs})} bearings | "
          f"{len({r.cond for r in recs})} conditions | {len(class_names)} classes")
    if skipped:
        print(f"[data] skipped: {skipped}")
    return recs, class_names


# ============================== 3. MAT LOADING ===============================
def _extract(obj, channel):
    if hasattr(obj, "_fieldnames"):
        if "Y" in obj._fieldnames:
            for ch in np.atleast_1d(getattr(obj, "Y")):
                if str(getattr(ch, "Name", "")).strip() == channel:
                    return np.asarray(getattr(ch, "Data"), dtype=np.float64).ravel()
        for f in obj._fieldnames:
            r = _extract(getattr(obj, f), channel)
            if r is not None:
                return r
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        for o in obj.ravel():
            r = _extract(o, channel)
            if r is not None:
                return r
    return None


def _channels(obj, out=None):
    out = set() if out is None else out
    if hasattr(obj, "_fieldnames"):
        if "Y" in obj._fieldnames:
            for ch in np.atleast_1d(getattr(obj, "Y")):
                out.add(str(getattr(ch, "Name", "")).strip())
        for f in obj._fieldnames:
            _channels(getattr(obj, f), out)
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        for o in obj.ravel():
            _channels(o, out)
    return out


def load_signal(path: Path, channel: str) -> np.ndarray:
    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        sig = _extract(v, channel)
        if sig is not None:
            return sig.astype(np.float32)
    avail = set()
    for k, v in mat.items():
        if not k.startswith("__"):
            _channels(v, avail)
    raise SystemExit(f"[mat] channel {channel!r} not in {path.name}; available: {sorted(avail)}")


# ============================== 4. WINDOWING =================================
def cache_key(args, n_files):
    h = hashlib.md5(json.dumps({
        "root": str(Path(args.data_root).resolve()), "task": args.task,
        "src": args.damage_source, "cond": sorted(args.conditions),
        "w": args.window, "s": args.stride, "mx": args.max_windows_per_file,
        "ch": args.channel, "n": n_files,
    }, sort_keys=True).encode()).hexdigest()[:12]
    return Path(args.cache_dir) / f"pb_{args.task}_{args.damage_source}_w{args.window}_{h}.npz"


def build_windows(recs, args):
    ck = cache_key(args, len(recs))
    if args.cache and ck.exists():
        z = np.load(ck, allow_pickle=False)
        print(f"[cache] loaded {ck}  X={z['X'].shape}")
        return z["X"], z["y"], z["fidx"]

    W, S, MX = args.window, args.stride, args.max_windows_per_file
    Xs, ys, fs = [], [], []
    t0 = time.time()
    for i, r in enumerate(recs):
        sig = load_signal(r.path, args.channel)
        if sig.size < W:
            warnings.warn(f"{r.stem}: {sig.size} < window {W}, skipped")
            continue
        starts = np.arange(0, sig.size - W + 1, S)
        if MX > 0 and len(starts) > MX:                     # spread over whole record
            starts = starts[np.linspace(0, len(starts) - 1, MX).round().astype(int)]
        for s in starts:
            Xs.append(sig[s:s + W])
            ys.append(r.label)
            fs.append(i)
        if (i + 1) % 100 == 0:
            print(f"  [win] {i+1}/{len(recs)} files, {len(Xs)} windows "
                  f"({time.time()-t0:.0f}s)", flush=True)

    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, np.int64)
    fidx = np.asarray(fs, np.int64)
    print(f"[win] X={X.shape} ({X.nbytes/1e6:.0f} MB) in {time.time()-t0:.0f}s")
    if args.cache:
        ck.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ck, X=X, y=y, fidx=fidx)
        print(f"[cache] saved {ck}")
    return X, y, fidx


# =============================== 5. SPLITS ===================================
# All splits operate at FILE level, then broadcast to windows via fidx.
# Values: "train" / "val" / "test" / "drop".

def _by_class_bearings(recs, mask=None):
    d = {}
    for i, r in enumerate(recs):
        if mask is not None and not mask[i]:
            continue
        d.setdefault(r.label, set()).add(r.code)
    return {k: sorted(v) for k, v in d.items()}


def _pick_per_class(recs, mask, seed, k=1):
    """k bearings per class from masked rows; never takes a class's last bearing."""
    rng, out = np.random.default_rng(seed), set()
    for _, bs in _by_class_bearings(recs, mask).items():
        b = rng.permutation(bs)
        out.update(b[:min(k, max(0, len(b) - 1))])
    return out


def split_file(recs, seed, **_):
    """LEAKY baseline: file-disjoint but the SAME bearings in train/val/test."""
    rng, m = np.random.default_rng(seed), {}
    per = {}
    for r in recs:
        per.setdefault(r.code, []).append(r.stem)
    for code, stems in per.items():
        f = rng.permutation(sorted(stems)); n = len(f)
        nt, nv = max(1, round(.20 * n)), max(1, round(.15 * n))
        for x in f[:nt]:          m[x] = "test"
        for x in f[nt:nt + nv]:   m[x] = "val"
        for x in f[nt + nv:]:     m[x] = "train"
    return np.array([m[r.stem] for r in recs])


def split_bearing(recs, seed, n_test=1, n_val=1, **_):
    """Single bearing-disjoint holdout, stratified by class."""
    rng, m = np.random.default_rng(seed), {}
    for lab, bs in _by_class_bearings(recs).items():
        b, n = rng.permutation(bs), len(bs)
        if n < 3:
            raise SystemExit(f"[split] class {lab} has {n} bearing(s); a bearing-"
                             f"disjoint 3-way split is impossible. Use --task 3class.")
        nt = max(1, min(n_test, n - 2))
        nv = max(1, min(n_val, n - 1 - nt))
        for x in b[:nt]:          m[x] = "test"
        for x in b[nt:nt + nv]:   m[x] = "val"
        for x in b[nt + nv:]:     m[x] = "train"
    return np.array([m[r.code] for r in recs])


def split_bearing_cv(recs, seed, fold=0, folds=5, n_val=1, **_):
    """Bearing-grouped stratified k-fold == StratifiedGroupKFold(groups=bearing),
    exploiting the fact that each bearing carries exactly one label."""
    cls = _by_class_bearings(recs)
    kmax = min(len(v) for v in cls.values())
    if folds > kmax:
        raise SystemExit(f"[split] --folds {folds} > smallest class bearing count "
                         f"({kmax}). Use --folds {kmax} or fewer.")
    m = {}
    for lab, bs in cls.items():
        b = np.random.default_rng(seed).permutation(bs)     # same perm for all folds
        chunks = np.array_split(b, folds)
        test = set(chunks[fold])
        rest = [x for x in b if x not in test]
        val = set(np.random.default_rng(seed * 1000 + fold).permutation(rest)
                  [:min(n_val, len(rest) - 1)])
        for x in b:
            m[x] = "test" if x in test else ("val" if x in val else "train")
    return np.array([m[r.code] for r in recs])


def split_art2real(recs, seed, **_):
    """train = artificial + most healthy | val = held-out artificial + healthy
       test = ALL real damage + held-out healthy   (so test contains every class)"""
    org = np.array([r.origin for r in recs])
    if not ((org == "real").any() and (org == "artificial").any()):
        raise SystemExit("[split] art2real needs both origins -> "
                         "--damage-source all --task 3class")
    rng = np.random.default_rng(seed)
    out = np.full(len(recs), "train", dtype=object)
    out[org == "real"] = "test"

    heal = rng.permutation(sorted({r.code for r in recs if r.origin == "healthy"}))
    nte = max(1, len(heal) // 3)
    h_te, rest = set(heal[:nte]), heal[nte:]
    h_va = set(rest[:max(1, len(rest) // 3)])
    for i, r in enumerate(recs):
        if r.code in h_te: out[i] = "test"
        elif r.code in h_va: out[i] = "val"

    v_art = _pick_per_class(recs, org == "artificial", seed + 1, k=1)
    for i, r in enumerate(recs):
        if r.code in v_art: out[i] = "val"
    return out.astype(str)


def split_loco(recs, seed, fold=0, strict=False, holdout=None, **_):
    """Leave-one-condition-out. strict=True also makes the test bearings unseen."""
    conds = sorted({r.cond for r in recs})
    if len(conds) < 2:
        raise SystemExit("[split] cross-condition needs >=2 conditions (--conditions all)")
    ho = set(holdout) if holdout else {conds[fold % len(conds)]}
    if ho - set(conds):
        raise SystemExit(f"[split] holdout {ho - set(conds)} not in loaded {conds}")

    is_ho = np.array([r.cond in ho for r in recs])
    out = np.full(len(recs), "train", dtype=object)
    out[is_ho] = "test"

    if strict:
        te_b = _pick_per_class(recs, is_ho, seed, k=1)
        for i, r in enumerate(recs):
            if not is_ho[i] and r.code in te_b:   out[i] = "drop"   # never train on them
            elif is_ho[i] and r.code not in te_b: out[i] = "drop"   # test = those only

    vb = _pick_per_class(recs, out == "train", seed + 2, k=1)
    for i, r in enumerate(recs):
        if out[i] == "train" and r.code in vb:
            out[i] = "val"
    return out.astype(str)


DISJOINT = {"bearing", "bearing_cv", "art2real", "loco_strict"}


def build_folds(recs, args):
    """-> list of (name, per-file split array)."""
    s, F = args.split, []
    if s == "file":
        F = [(f"file_s{sd}", split_file(recs, sd)) for sd in args.seeds]
    elif s == "bearing":
        F = [(f"bearing_s{sd}", split_bearing(recs, sd, args.n_test_bearings,
                                              args.n_val_bearings)) for sd in args.seeds]
    elif s == "bearing_cv":
        F = [(f"cvfold{k}", split_bearing_cv(recs, args.seeds[0], k, args.folds,
                                             args.n_val_bearings)) for k in range(args.folds)]
    elif s == "art2real":
        F = [(f"art2real_s{sd}", split_art2real(recs, sd)) for sd in args.seeds]

    elif s in ("loco", "loco_strict"):
        strict = s.endswith("strict")
        conds = sorted({r.cond for r in recs})
        hos = [args.holdout_cond] if args.holdout_cond else [[c] for c in conds]
        F = [(f"{'_'.join(h)}_s{sd}", split_loco(recs, sd, 0, strict, h))
             for sd in args.seeds for h in hos]
    else:
        raise SystemExit(f"[split] unknown strategy {s!r}")
    return F


def report_split(recs, sp, strategy, class_names, name=""):
    print(f"\n[split] {strategy} :: {name}")
    sets, keys = {}, ["train", "val", "test", "drop"]
    hdr = "".join(f"{c[:9]:>11s}" for c in class_names)
    print(f"  {'':7s}{hdr}{'total':>9s}")
    for k in keys:
        m = sp == k
        if not m.any():
            continue
        row = [sum(1 for i, r in enumerate(recs) if m[i] and r.label == c)
               for c in range(len(class_names))]
        print(f"  {k:7s}" + "".join(f"{v:>11d}" for v in row) + f"{sum(row):>9d}")

    for k in ("train", "val", "test"):
        m = sp == k
        if not m.any():
            raise SystemExit(f"[split] {k} split is EMPTY -- protocol invalid")
        b = sorted({r.code for i, r in enumerate(recs) if m[i]})
        sets[k] = set(b)
        miss = [class_names[c] for c in range(len(class_names))
                if not any(m[i] and r.label == c for i, r in enumerate(recs))]
        print(f"  {k:5s} {len(b):2d} bearings {b}")
        if miss:
            raise SystemExit(f"[split] {k} is MISSING classes {miss} -- protocol invalid")

    tr, va, te = sets["train"], sets["val"], sets["test"]
    if strategy in DISJOINT:
        for a, b, n in ((tr, te, "train/test"), (tr, va, "train/val"), (va, te, "val/test")):
            if a & b:
                raise SystemExit(f"[split] {n} bearing OVERLAP: {sorted(a & b)}")
        print("  [ok] bearings fully disjoint across splits")
    elif strategy == "loco":
        print(f"  [warn] {len(tr & te)} bearings recur train<->test BY DESIGN: "
              f"measures CONDITION transfer only, not unseen bearings.")
    else:
        print("  [warn] FILE split: same bearings in train and test -> "
              "in-distribution CEILING only, not generalisation.")
    return sets


# =============================== 6. DATASET ==================================
class WindowDS(Dataset):
    def __init__(self, X, y, idx, train=False, args=None):
        self.X, self.y, self.idx, self.train = X, y, idx, train
        g = (lambda k, d: getattr(args, k, d)) if args is not None else (lambda k, d: d)
        self.gain  = float(g("aug_gain", 0.0))
        self.noise = float(g("aug_noise", 0.0))
        self.flip  = bool(g("aug_flip", False))
        self.aug   = train and (self.gain > 0 or self.noise > 0 or self.flip)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = self.X[j].astype(np.float32, copy=True)
        if self.aug:
            if self.gain > 0:
                x *= np.float32(1.0 + np.random.uniform(-self.gain, self.gain))
            if self.flip and np.random.rand() < 0.5:
                x = -x
            if self.noise > 0:
                x += np.random.normal(0, self.noise * (x.std() + 1e-8),
                                      x.shape).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-8)      # per-window standardisation
        return torch.from_numpy(x[None, :]), int(self.y[j])

# ================================ 7. MODEL ===================================
def build_model(n_classes, args):
    """Flat nn.Sequential, no BatchNorm, ReLU + AvgPool -> zennit-ready."""
    bias = not args.no_bias
    Act = nn.ReLU if args.act == "relu" else nn.GELU
    Pool = nn.AvgPool1d if args.pool == "avg" else nn.MaxPool1d
    L = []

    def block(cin, cout, k, stride, pool):
        L.append(nn.Conv1d(cin, cout, k, stride, padding=k // 2, bias=bias))
        if args.norm == "group":
            L.append(nn.GroupNorm(min(8, cout), cout))
        L.append(Act())
        if pool > 1:
            L.append(Pool(pool))

    c = args.width
    block(1, c, args.first_kernel, args.first_stride, 2)     # wide first kernel (WDCNN)
    for _ in range(args.blocks):
        co = min(c * 2, args.max_width)
        block(c, co, 3, 1, 2)
        c = co
    L += [nn.AdaptiveAvgPool1d(1), nn.Flatten(),
          nn.Dropout(args.dropout), nn.Linear(c, n_classes)]

    m = nn.Sequential(*L)
    for mod in m.modules():
        if isinstance(mod, nn.Conv1d):
            nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            nn.init.zeros_(mod.bias)
    return m


# =============================== 8. METRICS ==================================
def confusion(yt, yp, C):
    cm = np.zeros((C, C), np.int64)
    np.add.at(cm, (yt, yp), 1)
    return cm


def report(cm, names, title=""):
    C = cm.shape[0]
    sup, pred = cm.sum(1), cm.sum(0)
    tp = np.diag(cm).astype(float)
    rec = np.divide(tp, sup, out=np.zeros(C), where=sup > 0)
    pre = np.divide(tp, pred, out=np.zeros(C), where=pred > 0)
    f1 = np.divide(2 * pre * rec, pre + rec, out=np.zeros(C), where=(pre + rec) > 0)
    acc = tp.sum() / max(1, cm.sum())
    bal = float(rec[sup > 0].mean())

    print(f"\n--- {title} ---")
    print(f"{'class':<12}{'support':>9}{'precision':>11}{'recall':>9}{'f1':>9}")
    for c in range(C):
        print(f"{names[c][:12]:<12}{sup[c]:>9d}{pre[c]:>11.4f}{rec[c]:>9.4f}{f1[c]:>9.4f}")
    print(f"{'-'*50}\naccuracy {acc:.4f} | balanced_acc(=macro recall) {bal:.4f} | "
          f"macro_f1 {f1.mean():.4f}")
    frac = pred / max(1, cm.sum())
    print("pred dist: " + " | ".join(f"{names[c][:10]} {frac[c]*100:5.1f}%" for c in range(C)))
    if frac.max() > 0.85:
        print(f"  !! COLLAPSE: {frac.max()*100:.0f}% of predictions are "
              f"'{names[int(frac.argmax())]}' -- accuracy is not meaningful")
    if bal < 1.3 / C:
        print(f"  !! balanced_acc {bal:.3f} ~ chance ({1/C:.3f}) -- model is not learning")
    if (rec[sup > 0] == 0).any():
        z = [names[c] for c in range(C) if sup[c] > 0 and rec[c] == 0]
        print(f"  !! classes NEVER recalled: {z}")
    print(f"confusion matrix (rows=true, cols=pred):\n{cm}")
    return dict(acc=float(acc), bal_acc=bal, macro_f1=float(f1.mean()),
                recall=rec.tolist(), precision=pre.tolist(), f1=f1.tolist(),
                support=sup.tolist(), cm=cm.tolist())


def agg_by_group(prob, yt, grp):
    """Mean-softmax aggregation within each group (file or bearing)."""
    u = np.unique(grp)
    P = np.stack([prob[grp == g].mean(0) for g in u])
    Y = np.array([yt[grp == g][0] for g in u])
    return Y, P.argmax(1)


# ============================== 9. TRAIN / EVAL ==============================
def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def run_epoch(model, dl, crit, dev, opt=None, sched=None, scaler=None):
    train = opt is not None
    model.train(train)
    tot = n = corr = 0
    for x, y in dl:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(dev.type, enabled=scaler is not None):
                out = model(x)
                loss = crit(out, y)
            if train:
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                if sched is not None:
                    sched.step()                     # <-- PER BATCH (OneCycle)
        tot += loss.item() * y.size(0); n += y.size(0)
        corr += (out.argmax(1) == y).sum().item()
    return tot / n, corr / n


@torch.no_grad()
def predict(model, dl, dev, C):
    model.eval()
    P, Y = [], []
    for x, y in dl:
        P.append(torch.softmax(model(x.to(dev)).float(), 1).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(Y), np.concatenate(P)


def eval_val(model, loader, dev, C, monitor):
    y, p = predict(model, loader, dev, C)
    cm = confusion(y, p.argmax(1), C)
    sup = cm.sum(1)
    rec = np.divide(np.diag(cm), sup, out=np.zeros(C), where=sup > 0)
    acc = float(np.trace(cm) / max(1, cm.sum()))
    bal = float(rec[sup > 0].mean()) if np.any(sup > 0) else 0.0
    return (bal if monitor == "bal_acc" else acc), acc, bal


def train_eval(X, y, recs, fidx, sp_win, class_names, args, tag, outdir):
    dev = torch.device(args.device)
    C = len(class_names)

    itr = np.where(sp_win == "train")[0]
    iva = np.where(sp_win == "val")[0]
    ite = np.where(sp_win == "test")[0]

    if args.no_val and len(iva):
        itr = np.concatenate([itr, iva]); iva = iva[:0]
        nb = len({recs[i].code for i in fidx[itr]})
        print(f"[no-val] val folded into train -> {nb} train bearings")


    print(f"[data] windows train={len(itr)} val={len(iva)} test={len(ite)}")

    mk = lambda idx, tr: DataLoader(
        WindowDS(X, y, idx, tr, args), batch_size=args.batch_size, shuffle=tr,
        num_workers=args.workers, pin_memory=(dev.type == "cuda"), drop_last=tr)
    dtr, dva, dte = mk(itr, True), mk(iva, False), mk(ite, False)

    model = build_model(C, args).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    print(f"[model] {npar/1e3:.0f}k params | norm={args.norm} act={args.act} pool={args.pool}")

    w = None
    if args.class_weights:
        cnt = np.bincount(y[itr], minlength=C).astype(np.float64)
        w = torch.tensor(cnt.sum() / (C * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
        print(f"[loss] class weights {w.cpu().numpy().round(3)}")
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=max(1, len(dtr)),
        pct_start=0.25)
    scaler = None
    if args.amp and dev.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()



    n_val = len(dva.dataset) if dva is not None else 0
    use_val = n_val > 0 and not args.no_val
    if not use_val:
        print(f"[select] no validation ({n_val} windows) -> "
              f"final-epoch model at fixed budget {args.epochs}")

    hist, best, best_ep, best_state, bad = [], -np.inf, args.epochs, None, 0

    for ep in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, dtr, crit, dev, opt, sched, scaler)

        if not use_val:
            print(f"ep{ep:3d} loss {tl:.4f} acc {ta:.4f} | "
                  f"lr {sched.get_last_lr()[0]:.2e}")
            continue

        mon, vacc, vbal = eval_val(model, dva, dev, C, args.monitor)
        hist.append(mon)
        sm = float(np.mean(hist[-args.val_smooth:]))
        print(f"ep{ep:3d} loss {tl:.4f} acc {ta:.4f} | val_acc {vacc:.4f} "
              f"val_bal {vbal:.4f} | sm{len(hist[-args.val_smooth:])} {sm:.4f} "
              f"| lr {sched.get_last_lr()[0]:.2e}")

        if ep < args.min_epochs:
            continue
        if sm > best + 1e-5:
            best, best_ep, bad = sm, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[early stop] no smoothed {args.monitor} gain for "
                      f"{bad} epochs")
                break





    if use_val:
        if best_state is None:
            print(f"[warn] {tag}: never reached --min-epochs {args.min_epochs}; "
                  f"using final epoch")
            best_ep = args.epochs
        else:
            model.load_state_dict(best_state)
        print(f"[select] {tag}: epoch {best_ep} | smoothed {args.monitor} "
              f"{best if np.isfinite(best) else float('nan'):.4f}")

    torch.save({"state": model.state_dict(), "class_names": class_names,
                "args": vars(args), "epoch": int(best_ep),
                "val": float(best) if np.isfinite(best) else None,
                "selected_on": args.monitor if use_val else "final_epoch"},
               outdir / f"{tag}.pt")

    yt, pt = predict(model, dte, dev, C)
    cm_w = confusion(yt, pt.argmax(1), C)
    res = {"window": report(cm_w, class_names, f"{tag} | WINDOW level")}

    fi = fidx[ite]
    Yf, Pf = agg_by_group(pt, yt, fi)
    res["file"] = report(confusion(Yf, Pf, C), class_names, f"{tag} | FILE level (mean softmax)")

    bi = np.array([recs[i].code for i in fi])
    Yb, Pb = agg_by_group(pt, yt, bi)


    res["best_val"] = float(best) if np.isfinite(best) else None
    res["best_epoch"] = int(best_ep)
    res["selected_on"] = args.monitor if use_val else "final_epoch"
    res["bearing"] = report(confusion(Yb, Pb, C), class_names,f"{tag} | BEARING level")


    return res, cm_w


# ================================= 10. MODES =================================
def mode_inspect(args):
    recs, names = build_index(args)
    per_code, per_cond = {}, {}
    for r in recs:
        per_code[r.code] = per_code.get(r.code, 0) + 1
        per_cond[r.cond] = per_cond.get(r.cond, 0) + 1
    print("\n[inventory] files per bearing")
    for c in sorted(per_code):
        print(f"  {c}  {per_code[c]:4d}  origin={origin_of(c):<10s} damage={damage_of(c)}")
    print("\n[inventory] files per condition")
    for c in sorted(per_cond):
        print(f"  {c}  {per_cond[c]:4d}")
    print("\n[inventory] bearings per class")
    for lab, bs in sorted(_by_class_bearings(recs).items()):
        print(f"  {names[lab]:<10s} {len(bs):2d}  {bs}")
    kmax = min(len(v) for v in _by_class_bearings(recs).values())
    print(f"\n[inventory] max usable --folds for bearing_cv = {kmax}")

    p = recs[0].path
    mat = sio.loadmat(str(p), squeeze_me=True, struct_as_record=False)
    avail = set()
    for k, v in mat.items():
        if not k.startswith("__"):
            _channels(v, avail)
    sig = load_signal(p, args.channel)
    print(f"\n[mat] {p.name}\n  channels: {sorted(avail)}"
          f"\n  {args.channel}: n={sig.size} ({sig.size/FS:.2f}s) "
          f"mean={sig.mean():.4f} std={sig.std():.4f}")
    print(f"  window {args.window} = {args.window/FS*1000:.1f} ms "
          f"= {args.window/FS*1500/60:.2f} shaft revs @1500rpm")


def mode_splits(args):
    recs, names = build_index(args)
    for nm, sp in build_folds(recs, args):
        report_split(recs, sp, args.split, names, nm)


def mode_train(args):
    recs, names = build_index(args)
    folds = build_folds(recs, args)
    X, y, fidx = build_windows(recs, args)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    cms, accs, bals, all_res = [], [], [], {}
    for nm, sp in folds:
        report_split(recs, sp, args.split, names, nm)
        set_seed(args.seeds[0])
        res, cm = train_eval(X, y, recs, fidx, sp[fidx], names, args, nm, outdir)
        cms.append(cm)
        accs.append(res["window"]["acc"])
        bals.append(res["window"]["bal_acc"])
        all_res[nm] = res

    if len(folds) > 1:
        print(f"\n{'='*64}\n== {args.split}: {len(folds)} folds ==")
        print(f"window acc      {np.mean(accs):.4f} +/- {np.std(accs):.4f}  "
              f"[{min(accs):.4f}, {max(accs):.4f}]")
        print(f"balanced acc    {np.mean(bals):.4f} +/- {np.std(bals):.4f}  "
              f"[{min(bals):.4f}, {max(bals):.4f}]")
        pooled = report(sum(cms), names, f"POOLED over {len(folds)} folds")
        all_res["pooled"] = pooled
        all_res["summary"] = dict(acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                                  bal_mean=float(np.mean(bals)), bal_std=float(np.std(bals)),
                                  per_fold_acc=accs, per_fold_bal=bals)
    (outdir / "results.json").write_text(json.dumps(all_res, indent=2))
    print(f"\n[out] {outdir/'results.json'}")


def mode_lrp(args):
    try:
        from zennit.attribution import Gradient
        from zennit.composites import EpsilonAlpha2Beta1Flat, EpsilonPlusFlat
    except ImportError:
        raise SystemExit("[lrp] pip install zennit")


    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    names, cka = ck["class_names"], ck.get("args", {})
    ARCH = ("norm", "act", "pool", "no_bias", "width", "max_width",
            "blocks", "first_kernel", "first_stride", "dropout")
    margs = argparse.Namespace(**{**vars(args),
                                  **{k: cka[k] for k in ARCH if k in cka}})
    if cka.get("window", args.window) != args.window:
        raise SystemExit(f"[lrp] --window {args.window} != trained "
                         f"{cka['window']}; must match")
    dev = torch.device(args.device)
    model = build_model(len(names), margs).to(dev)
    model.load_state_dict(ck["state"]); model.eval()

    recs, _ = build_index(args)
    X, y, fidx = build_windows(recs, args)
    sel = np.random.default_rng(0).choice(len(X), min(args.lrp_n, len(X)), replace=False)
    xb = torch.stack([torch.from_numpy(((X[i] - X[i].mean()) / (X[i].std() + 1e-8))[None])
                      for i in sel]).to(dev)

    comp = {"epsilon_plus_flat": EpsilonPlusFlat,
            "epsilon_a2b1_flat": EpsilonAlpha2Beta1Flat}[args.lrp_rule]()
    with Gradient(model=model, composite=comp) as att:
        with torch.no_grad():
            pred = model(xb).argmax(1)
        onehot = torch.eye(len(names), device=dev)[pred]
        out, rel = att(xb, onehot)

    p = Path(args.out); p.mkdir(parents=True, exist_ok=True)
    np.savez(p / "lrp.npz", x=xb.cpu().numpy(), relevance=rel.detach().cpu().numpy(),
             y_true=y[sel], y_pred=pred.cpu().numpy(), class_names=np.array(names))
    print(f"[lrp] {args.lrp_rule} on {len(sel)} windows -> {p/'lrp.npz'}")
    print(f"[lrp] relevance sum/logit ratio (conservation check): "
          f"{(rel.sum((1,2)) / out.gather(1, pred[:,None]).squeeze(1)).mean().item():.3f}")


# =================================== CLI =====================================
def get_args():
    p = argparse.ArgumentParser("Paderborn bearing CNN1D")
    p.add_argument("--mode", default="train", choices=["inspect", "splits", "train", "lrp"])
    # data
    p.add_argument("--data-root", required=True)
    p.add_argument("--task", default="3class", choices=["binary", "3class", "4class", "code"])
    p.add_argument("--damage-source", default="real", choices=["real", "artificial", "all"])
    p.add_argument("--conditions", nargs="+", default=["N15_M07_F10"],
                   help=f"'all' or any of {CONDITIONS}")
    p.add_argument("--channel", default="vibration_1")
    p.add_argument("--window", type=int, default=8192)
    p.add_argument("--stride", type=int, default=4096)
    p.add_argument("--max-windows-per-file", type=int, default=24)
    p.add_argument("--cache", type=int, default=1)
    p.add_argument("--cache-dir", default="cache")
    # split
    p.add_argument("--split", default="bearing_cv",
                   choices=["file", "bearing", "bearing_cv", "art2real", "loco", "loco_strict"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--n-test-bearings", type=int, default=1)
    p.add_argument("--n-val-bearings", type=int, default=1)
    p.add_argument("--holdout-cond", nargs="+", default=None)
    # model (LRP-friendly defaults)
    p.add_argument("--norm", default="none", choices=["none", "group"])
    p.add_argument("--act", default="relu", choices=["relu", "gelu"])
    p.add_argument("--pool", default="avg", choices=["avg", "max"])
    p.add_argument("--no-bias", action="store_true")
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--max-width", type=int, default=128)
    p.add_argument("--blocks", type=int, default=5)
    p.add_argument("--first-kernel", type=int, default=64)
    p.add_argument("--first-stride", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.3)
    # train
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--monitor", default="bal_acc", choices=["bal_acc", "acc"])
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="runs/exp")

    p.add_argument("--aug-gain",  type=float, default=0.0)
    p.add_argument("--aug-noise", type=float, default=0.0)
    p.add_argument("--aug-flip",  action="store_true")

    p.add_argument("--min-epochs", type=int, default=15,
                   help="no checkpointing/early-stop before this epoch")
    p.add_argument("--val-smooth", type=int, default=3,
                   help="select on mean of last N val scores")
    p.add_argument("--no-val", action="store_true",
                   help="fold val bearings into train; keep final-epoch model")

    # lrp
    p.add_argument("--ckpt", default=None)
    p.add_argument("--lrp-n", type=int, default=64)
    p.add_argument("--lrp-rule", default="epsilon_plus_flat",
                   choices=["epsilon_plus_flat", "epsilon_a2b1_flat"])
    a = p.parse_args()
    if a.norm != "none":
        print("[warn] --norm group requires a zennit canonizer for correct LRP")
    return a


if __name__ == "__main__":
    a = get_args()
    set_seed(a.seeds[0])
    {"inspect": mode_inspect, "splits": mode_splits,
     "train": mode_train, "lrp": mode_lrp}[a.mode](a)