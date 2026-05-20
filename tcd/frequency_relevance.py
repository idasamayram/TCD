"""Frequency-domain relevance utilities for TCD."""

from typing import Dict, Iterable, List, Tuple

import numpy as np


def stabilized_divisor(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return ``signal`` with small values replaced by signed ``eps``."""
    sign = np.sign(signal)
    sign[sign == 0] = 1.0
    return np.where(np.abs(signal) < eps, eps * sign, signal)


def create_fourier_weights(
    signal_length: int,
    inverse: bool = False,
    symmetry: bool = False,
    real: bool = False,
) -> np.ndarray:
    """Numpy port of CNC ``utils/dft_utils.py:create_fourier_weights``."""
    k_vals, n_vals = np.mgrid[0:signal_length, 0:signal_length]
    theta_vals = 2 * np.pi * k_vals * n_vals / signal_length
    sign = 1.0 if inverse else -1.0
    norm = 1 / np.sqrt(signal_length)

    if symmetry:
        nyquist_k = signal_length // 2
        if inverse:
            w_0 = np.ones(signal_length)[np.newaxis, :]
            w_nyquist = np.ones(signal_length)[np.newaxis, :]
            if real:
                return norm * np.vstack([
                    w_0,
                    2 * np.cos(theta_vals[1:nyquist_k]),
                    w_nyquist,
                    -2 * np.sin(theta_vals[1:nyquist_k]),
                ])
            return norm * np.vstack([w_0, 2 * np.exp(sign * 1j * theta_vals[1:nyquist_k]), w_nyquist])
        if real:
            return norm * np.hstack([
                np.cos(theta_vals[:, :nyquist_k + 1]),
                -np.sin(theta_vals[:, 1:nyquist_k]),
            ])
        return norm * np.exp(sign * 1j * theta_vals[:, :nyquist_k + 1])

    if real:
        if inverse:
            return norm * np.vstack([np.cos(theta_vals), -np.sin(theta_vals)])
        return norm * np.hstack([np.cos(theta_vals), -np.sin(theta_vals)])

    return norm * np.exp(sign * 1j * theta_vals)




def rectangle_window(m: int, width: int, signal_length: int, shift: int) -> np.ndarray:
    """Rectangle window used by STDFT weight construction."""
    w_nm = np.zeros(signal_length)
    w_nm[m:m + width] = 1 / np.sqrt(shift)
    return w_nm


def halfsine_window(m: int, width: int, signal_length: int, shift: int = 1) -> np.ndarray:
    """Half-sine window used by STDFT weight construction."""
    del shift
    w_nm = np.zeros(signal_length)
    w_nm[m:m + width] = np.sin(np.pi / width * (np.arange(width) + 0.5))
    return w_nm


WINDOWS = {"rectangle": rectangle_window, "halfsine": halfsine_window}


def create_window_mask(shift: int, width: int, signal_length: int, window_function) -> np.ndarray:
    """Create STDFT window mask matrix ``W_mn`` as in CNC dft_utils."""
    ms = np.arange(0, signal_length - width + 1, width // shift)
    w_mn = [window_function(m, width, signal_length, shift)[np.newaxis] for m in ms]
    w_mn = np.concatenate(w_mn, axis=0)
    return w_mn.transpose((1, 0))


def create_short_time_fourier_weights(
    signal_length: int,
    shift: int,
    window_width: int,
    window_shape: str,
    inverse: bool = False,
    real: bool = False,
    symmetry: bool = False,
) -> np.ndarray:
    """Numpy port of CNC ``create_short_time_fourier_weights``."""
    if window_shape not in ("rectangle", "halfsine", "hann"):
        raise ValueError("Available window shapes: rectangle, halfsine, hann")
    if window_shape == "hann":
        raise NotImplementedError("hann window is declared in CNC but not implemented there either")

    window_function = WINDOWS[window_shape]
    w_mn = create_window_mask(shift, window_width, signal_length, window_function)
    dft_kn = create_fourier_weights(signal_length, inverse=inverse, symmetry=symmetry, real=real)

    dtype = np.complex64 if not real else np.float16
    if inverse:
        w = w_mn.sum(axis=1)
        dft_kn_m = np.zeros((w_mn.shape[1] * dft_kn.shape[0], dft_kn.shape[1]), dtype=dtype)
        for i in range(0, dft_kn_m.shape[0], dft_kn.shape[0]):
            dft_kn_m[i:i + dft_kn.shape[0]] = dft_kn
        return dft_kn_m / w.astype(dtype)

    stdft_mkn = np.zeros((dft_kn.shape[0], w_mn.shape[1] * dft_kn.shape[1]), dtype=dtype)
    for m, k in enumerate(range(0, w_mn.shape[1] * dft_kn.shape[1], dft_kn.shape[1])):
        stdft_mkn[:, k:k + dft_kn.shape[1]] = dft_kn * w_mn[:, m][:, np.newaxis]
    return stdft_mkn

def _dft_frequency_relevance_core(
    signal: np.ndarray,
    score: np.ndarray,
    sample_rate: float = 400.0,
    eps: float = 1e-6,
    one_sided: bool = True,
    renormalize: bool = False,
    method_name: str = "dft",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Shared DFT mapping from a time-domain score to frequency-bin relevance."""
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if signal.shape != score.shape:
        raise ValueError(f"signal and score must have the same shape, got {signal.shape} and {score.shape}")

    n_time = signal.shape[0]
    ratio = score / stabilized_divisor(signal, eps=eps)

    # CNC-style normalized DFT basis with explicit cosine projection.
    spectrum = np.fft.fft(signal) / np.sqrt(n_time)
    amplitudes = np.abs(spectrum)
    phases = np.angle(spectrum)

    k = np.arange(n_time, dtype=np.float64)[:, None]
    n = np.arange(n_time, dtype=np.float64)[None, :]
    basis = np.cos((2.0 * np.pi * k * n / n_time) + phases[:, None])
    freq_relevance_full = amplitudes * (basis @ ratio)

    time_sum = float(np.sum(score))
    freq_sum_full = float(np.sum(freq_relevance_full))
    conservation_error = abs(time_sum - freq_sum_full) / (abs(time_sum) + eps)

    if renormalize and abs(freq_sum_full) > eps:
        freq_relevance_full = freq_relevance_full * (time_sum / freq_sum_full)
        freq_sum_full = float(np.sum(freq_relevance_full))

    freqs_full = np.fft.fftfreq(n_time, d=1.0 / sample_rate)
    diagnostics = {
        "time_score_sum": time_sum,
        "frequency_relevance_sum_full": freq_sum_full,
        "conservation_error": float(conservation_error),
        "signal_energy": float(np.sum(signal ** 2)),
        "score_l1": float(np.sum(np.abs(score))),
        "method": method_name,
    }

    if one_sided:
        mask = freqs_full >= 0
        return freqs_full[mask], freq_relevance_full[mask], diagnostics
    return freqs_full, freq_relevance_full, diagnostics


def dft_lrp_frequency_relevance(signal: np.ndarray, relevance: np.ndarray, **kwargs):
    freqs, freq_rel, diagnostics = _dft_frequency_relevance_core(
        signal=signal,
        score=relevance,
        method_name="dft_lrp",
        **kwargs,
    )
    diagnostics["time_relevance_sum"] = diagnostics.pop("time_score_sum")
    diagnostics["relevance_l1"] = diagnostics.pop("score_l1")
    return freqs, freq_rel, diagnostics


def dft_crp_frequency_relevance(signal: np.ndarray, concept_relevance: np.ndarray, **kwargs):
    freqs, freq_rel, diagnostics = _dft_frequency_relevance_core(
        signal=signal,
        score=concept_relevance,
        method_name="dft_crp",
        **kwargs,
    )
    diagnostics["time_concept_relevance_sum"] = diagnostics.pop("time_score_sum")
    diagnostics["concept_relevance_l1"] = diagnostics.pop("score_l1")
    return freqs, freq_rel, diagnostics


def dft_pcx_frequency_relevance(signal: np.ndarray, prototype_contribution: np.ndarray, **kwargs):
    freqs, freq_rel, diagnostics = _dft_frequency_relevance_core(
        signal=signal,
        score=prototype_contribution,
        method_name="dft_pcx",
        **kwargs,
    )
    diagnostics["time_prototype_contribution_sum"] = diagnostics.pop("time_score_sum")
    diagnostics["prototype_contribution_l1"] = diagnostics.pop("score_l1")
    return freqs, freq_rel, diagnostics


def band_relevance(
    freqs: np.ndarray,
    relevance: np.ndarray,
    bands: Iterable[Tuple[str, float, float]],
    use_absolute: bool = True,
) -> Dict[str, float]:
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
