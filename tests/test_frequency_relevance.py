import numpy as np

from tcd.frequency_relevance import (
    DEFAULT_CNC_BANDS,
    band_relevance,
    dft_lrp_frequency_relevance,
)


def test_dft_lrp_frequency_relevance_shapes_and_diagnostics():
    rng = np.random.RandomState(0)
    signal = rng.randn(128) + 0.1
    relevance = rng.randn(128) * 0.01

    freqs, freq_rel, diagnostics = dft_lrp_frequency_relevance(
        signal, relevance, sample_rate=400.0, eps=1e-6
    )

    assert freqs.ndim == 1
    assert freq_rel.shape == freqs.shape
    assert len(freqs) == 64
    assert np.all(freqs >= 0)
    assert np.all(np.isfinite(freq_rel))
    assert np.isfinite(diagnostics["conservation_error"])
    assert diagnostics["relevance_l1"] > 0


def test_band_relevance_returns_band_sums_and_ratios():
    freqs = np.array([1.0, 20.0, 75.0, 150.0])
    relevance = np.array([1.0, 2.0, 3.0, 4.0])

    result = band_relevance(freqs, relevance, DEFAULT_CNC_BANDS)

    assert result["band_0_10"] == 1.0
    assert result["band_10_50"] == 2.0
    assert result["band_50_100"] == 3.0
    assert result["band_100_200"] == 4.0
    assert np.isclose(
        sum(result[f"band_{name}_ratio"] for name, _, _ in DEFAULT_CNC_BANDS), 1.0
    )
