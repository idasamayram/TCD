import numpy as np

from tcd.frequency_relevance import (
    DEFAULT_CNC_BANDS,
    band_relevance,
    dft_crp_frequency_relevance,
    dft_lrp_frequency_relevance,
    dft_pcx_frequency_relevance,
    create_short_time_fourier_weights,
    create_window_mask,
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


def test_dft_crp_and_dft_pcx_shapes_and_method_tags():
    rng = np.random.RandomState(1)
    signal = rng.randn(64) + 0.2
    concept_rel = rng.randn(64) * 0.05
    proto_contrib = rng.randn(64) * 0.03

    f1, r1, d1 = dft_crp_frequency_relevance(signal, concept_rel, sample_rate=400.0)
    f2, r2, d2 = dft_pcx_frequency_relevance(signal, proto_contrib, sample_rate=400.0)

    assert f1.shape == r1.shape
    assert f2.shape == r2.shape
    assert np.all(np.isfinite(r1))
    assert np.all(np.isfinite(r2))
    assert d1["method"] == "dft_crp"
    assert d2["method"] == "dft_pcx"
    assert "time_concept_relevance_sum" in d1
    assert "time_prototype_contribution_sum" in d2


def test_create_fourier_weights_symmetry_real_shape():
    from tcd.frequency_relevance import create_fourier_weights

    w = create_fourier_weights(signal_length=128, inverse=False, symmetry=True, real=True)
    assert w.shape == (128, 128)


def test_create_window_mask_shape():
    wm = create_window_mask(shift=2, width=16, signal_length=64, window_function=lambda m,w,s,sh: __import__("numpy").ones(s))
    assert wm.shape[0] == 64

def test_create_short_time_fourier_weights_shapes():
    w = create_short_time_fourier_weights(
        signal_length=64,
        shift=2,
        window_width=16,
        window_shape="rectangle",
        inverse=False,
        real=True,
        symmetry=True,
    )
    assert w.ndim == 2
    assert w.shape[0] == 64
