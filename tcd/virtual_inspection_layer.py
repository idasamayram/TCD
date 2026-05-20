"""
Virtual Inspection Layer (VIL) for frequency-domain relevance in TCD.

This module implements the real-valued IDFT "virtual layer" described by
Vielhaben for DFT-LRP.  The core idea is to represent the complex IDFT as a
real-valued linear layer so that standard LRP-z rules can be applied without
complex arithmetic.

Mathematical summary (real-valued IDFT layer):
    x_n = (1/N) * sum_k [a_k cos(2πkn/N) - b_k sin(2πkn/N)]
    where z_k = a_k + i b_k is the DFT of x.

Define weight matrices:
    W_re[n, k] = (1/N) * cos(2πkn/N)
    W_im[n, k] = -(1/N) * sin(2πkn/N)

LRP-z redistribution for the VIL (per frequency bin k):
    R_a[k] = sum_n (a_k * W_re[n, k] / (x_n + ε)) * R_x[n]
    R_b[k] = sum_n (b_k * W_im[n, k] / (x_n + ε)) * R_x[n]
    R_z[k] = R_a[k] + R_b[k]

This matches the CNC implementation logic while keeping all operations real.
"""
from __future__ import annotations

from typing import Dict, Tuple, List, Iterable

import numpy as np


def stabilized_divisor(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return ``signal`` with small values replaced by signed ``eps``."""
    signal = np.asarray(signal, dtype=np.float64)
    sign = np.sign(signal)
    sign[sign == 0] = 1.0
    return np.where(np.abs(signal) < eps, eps * sign, signal)


def _dft_weights(n_time: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return real-valued IDFT weights W_re and W_im (shape: N x N)."""
    k_vals, n_vals = np.mgrid[0:n_time, 0:n_time]
    theta = 2.0 * np.pi * k_vals * n_vals / n_time
    w_re = (1.0 / n_time) * np.cos(theta)
    w_im = -(1.0 / n_time) * np.sin(theta)
    return w_re, w_im


def vil_idft_frequency_relevance(
    signal: np.ndarray,
    relevance: np.ndarray,
    sample_rate: float = 400.0,
    eps: float = 1e-6,
    one_sided: bool = True,
    renormalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Compute frequency-domain relevance using the VIL IDFT formulation.

    Args:
        signal: 1-D input signal of shape (T,).
        relevance: 1-D time-domain relevance of shape (T,).
        sample_rate: Sampling rate in Hz.
        eps: Stabilizer for division by small values.
        one_sided: If True, return non-negative frequencies only.
        renormalize: If True, scale frequency relevance to match time relevance sum.

    Returns:
        (freqs, freq_relevance, diagnostics)
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    relevance = np.asarray(relevance, dtype=np.float64).reshape(-1)
    if signal.shape != relevance.shape:
        raise ValueError(
            f"signal and relevance must have the same shape, got "
            f"{signal.shape} and {relevance.shape}"
        )

    n_time = signal.shape[0]
    w_re, w_im = _dft_weights(n_time)

    spectrum = np.fft.fft(signal)
    a = spectrum.real
    b = spectrum.imag

    ratio = relevance / stabilized_divisor(signal, eps=eps)

    # sum_n W[n,k] * (R_n / x_n)
    sum_re = w_re.T @ ratio
    sum_im = w_im.T @ ratio

    r_a = a * sum_re
    r_b = b * sum_im
    r_z = r_a + r_b

    time_sum = float(np.sum(relevance))
    freq_sum_full = float(np.sum(r_z))
    conservation_error = abs(time_sum - freq_sum_full) / (abs(time_sum) + eps)

    if renormalize and abs(freq_sum_full) > eps:
        r_z = r_z * (time_sum / freq_sum_full)
        freq_sum_full = float(np.sum(r_z))

    freqs_full = np.fft.fftfreq(n_time, d=1.0 / sample_rate)

    diagnostics = {
        "time_relevance_sum": time_sum,
        "frequency_relevance_sum_full": freq_sum_full,
        "conservation_error": float(conservation_error),
        "signal_energy": float(np.sum(signal ** 2)),
        "relevance_l1": float(np.sum(np.abs(relevance))),
    }

    if not one_sided:
        return freqs_full, r_z, diagnostics

    mask = freqs_full >= 0
    return freqs_full[mask], r_z[mask], diagnostics


# --- STDFT helpers (mirroring CNC defaults and window logic) ---

def rectangle_window(m: int, width: int, signal_length: int, shift: int) -> np.ndarray:
    """
    Rectangle window used in CNC.

    Note: ``shift`` is interpreted as the *fraction* of the window width
    by which the window is shifted (CNC convention).
    """
    w_nm = np.zeros(signal_length)
    w_nm[m:m + width] = 1.0 / np.sqrt(shift)
    return w_nm


def halfsine_window(m: int, width: int, signal_length: int, shift: int | None = None) -> np.ndarray:
    """Half-sine window used in CNC."""
    w_nm = np.zeros(signal_length)
    w_nm[m:m + width] = np.sin(np.pi / width * (np.arange(width) + 0.5))
    return w_nm


WINDOWS = {"rectangle": rectangle_window, "halfsine": halfsine_window}


def create_window_mask(
    shift: int,
    width: int,
    signal_length: int,
    window_function,
) -> np.ndarray:
    """
    CNC-compatible window mask generator.

    CNC convention: ``shift`` is a fraction of ``width`` (step = width // shift).
    """
    step = max(1, width // shift)
    ms = np.arange(0, signal_length - width + 1, step)
    w_mn = [window_function(m, width, signal_length, shift)[np.newaxis] for m in ms]
    w_mn = np.concatenate(w_mn, axis=0)
    return w_mn.transpose((1, 0))  # (N, n_windows)


def vil_stdft_frequency_relevance(
    signal: np.ndarray,
    relevance: np.ndarray,
    sample_rate: float = 400.0,
    eps: float = 1e-6,
    one_sided: bool = True,
    renormalize: bool = False,
    window_width: int = 128,
    window_shift: int | None = None,
    window_shape: str = "rectangle",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Compute STDFT-style frequency relevance using VIL per time window.

    This mirrors the CNC defaults:
        window_width = 128
        window_shift = window_width // 2 (if None)
        window_shape = "rectangle"

    Returns:
        freqs: Frequency bins (one-sided if requested)
        relevance_tf: Array of shape (n_windows, n_freqs)
        diagnostics: Aggregate diagnostics (mean conservation error, etc.)
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    relevance = np.asarray(relevance, dtype=np.float64).reshape(-1)
    if signal.shape != relevance.shape:
        raise ValueError(
            f"signal and relevance must have the same shape, got "
            f"{signal.shape} and {relevance.shape}"
        )

    if window_shift is None:
        # CNC default (see compute_enhanced_dft_lrp): 50% overlap parameter
        window_shift = window_width // 2

    if window_shape not in WINDOWS:
        raise ValueError(
            f"window_shape must be one of {list(WINDOWS.keys())}, got {window_shape}"
        )

    window_fn = WINDOWS[window_shape]
    w_mn = create_window_mask(window_shift, window_width, signal.shape[0], window_fn)

    n_windows = w_mn.shape[1]
    all_relevances: List[np.ndarray] = []
    diag_errors: List[float] = []
    diag_time_sums: List[float] = []
    diag_freq_sums: List[float] = []

    freqs_ref = None

    for w_idx in range(n_windows):
        window = w_mn[:, w_idx]
        signal_w = signal * window
        relevance_w = relevance * window

        freqs, freq_rel, diagnostics = vil_idft_frequency_relevance(
            signal=signal_w,
            relevance=relevance_w,
            sample_rate=sample_rate,
            eps=eps,
            one_sided=one_sided,
            renormalize=renormalize,
        )
        freqs_ref = freqs
        all_relevances.append(freq_rel)
        diag_errors.append(diagnostics["conservation_error"])
        diag_time_sums.append(diagnostics["time_relevance_sum"])
        diag_freq_sums.append(diagnostics["frequency_relevance_sum_full"])

    relevance_tf = np.stack(all_relevances, axis=0) if all_relevances else np.zeros((0, 0))

    diagnostics = {
        "time_relevance_sum": float(np.sum(diag_time_sums)) if diag_time_sums else 0.0,
        "frequency_relevance_sum_full": float(np.sum(diag_freq_sums)) if diag_freq_sums else 0.0,
        "conservation_error": float(np.mean(diag_errors)) if diag_errors else 0.0,
        "n_windows": int(n_windows),
    }

    if freqs_ref is None:
        freqs_ref = np.fft.fftfreq(signal.shape[0], d=1.0 / sample_rate)
        if one_sided:
            freqs_ref = freqs_ref[freqs_ref >= 0]

    return freqs_ref, relevance_tf, diagnostics
