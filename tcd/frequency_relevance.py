"""
Frequency-domain relevance utilities for TCD.

This module implements a small, dependency-light DFT-LRP style transform for
post-hoc prototype analysis.  It is intended to be used with input-level
relevance heatmaps saved by ``scripts/run_analysis.py`` and raw CNC signals.
"""

from typing import Dict, Iterable, List, Tuple

import numpy as np


def stabilized_divisor(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return ``signal`` with small values replaced by signed ``eps``."""
    sign = np.sign(signal)
    sign[sign == 0] = 1.0
    return np.where(np.abs(signal) < eps, eps * sign, signal)


def dft_lrp_frequency_relevance(
    signal: np.ndarray,
    relevance: np.ndarray,
    sample_rate: float = 400.0,
    eps: float = 1e-6,
    one_sided: bool = True,
    renormalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Map time-domain relevance to DFT-bin relevance.

    This follows the DFT-LRP rule used in the CNC thesis code in a compact
    NumPy form.  For a signal ``x`` and time relevance ``R_t`` we compute

    ``R_k = |X_k| * sum_n cos(2*pi*k*n/N + angle(X_k)) * R_t[n] / x[n]``.

    Args:
        signal: 1-D input signal of shape ``(T,)``.
        relevance: 1-D time-domain relevance of shape ``(T,)``.
        sample_rate: Sampling rate in Hz.
        eps: Stabilizer for divisions by values close to zero.
        one_sided: If ``True``, return non-negative frequencies only.
        renormalize: If ``True``, scale frequency relevance so that its sum
            matches the time-relevance sum when numerically possible.

    Returns:
        Tuple ``(freqs, freq_relevance, diagnostics)``.
    """
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    relevance = np.asarray(relevance, dtype=np.float64).reshape(-1)
    if signal.shape != relevance.shape:
        raise ValueError(
            f"signal and relevance must have the same shape, got "
            f"{signal.shape} and {relevance.shape}"
        )

    n_time = signal.shape[0]
    ratio = relevance / stabilized_divisor(signal, eps=eps)

    spectrum = np.fft.fft(signal)
    amplitudes = np.abs(spectrum)
    phases = np.angle(spectrum)

    k = np.arange(n_time, dtype=np.float64)[:, None]
    n = np.arange(n_time, dtype=np.float64)[None, :]
    basis = np.cos((2.0 * np.pi * k * n / n_time) + phases[:, None])
    freq_relevance_full = amplitudes * (basis @ ratio)

    time_sum = float(np.sum(relevance))
    freq_sum_full = float(np.sum(freq_relevance_full))
    conservation_error = abs(time_sum - freq_sum_full) / (abs(time_sum) + eps)

    if renormalize and abs(freq_sum_full) > eps:
        freq_relevance_full = freq_relevance_full * (time_sum / freq_sum_full)
        freq_sum_full = float(np.sum(freq_relevance_full))

    freqs_full = np.fft.fftfreq(n_time, d=1.0 / sample_rate)

    diagnostics = {
        "time_relevance_sum": time_sum,
        "frequency_relevance_sum_full": freq_sum_full,
        "conservation_error": float(conservation_error),
        "signal_energy": float(np.sum(signal ** 2)),
        "relevance_l1": float(np.sum(np.abs(relevance))),
    }

    if not one_sided:
        return freqs_full, freq_relevance_full, diagnostics

    mask = freqs_full >= 0
    return freqs_full[mask], freq_relevance_full[mask], diagnostics


def band_relevance(
    freqs: np.ndarray,
    relevance: np.ndarray,
    bands: Iterable[Tuple[str, float, float]],
    use_absolute: bool = True,
) -> Dict[str, float]:
    """Aggregate frequency relevance inside named frequency bands."""
    freqs = np.asarray(freqs)
    values = np.abs(relevance) if use_absolute else relevance
    result: Dict[str, float] = {}
    total = float(np.sum(np.abs(values)) + 1e-12)
    for name, low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        band_sum = float(np.sum(values[mask])) if np.any(mask) else 0.0
        result[f"band_{name}"] = band_sum
        result[f"band_{name}_ratio"] = band_sum / total
    return result


DEFAULT_CNC_BANDS: List[Tuple[str, float, float]] = [
    ("0_10", 0.0, 10.0),
    ("10_50", 10.0, 50.0),
    ("50_100", 50.0, 100.0),
    ("100_200", 100.0, 200.0),
]
