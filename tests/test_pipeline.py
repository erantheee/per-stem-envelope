import numpy as np
import pytest

from stemenvelope.bands import BandMap
from stemenvelope.evidence import Evidence
from stemenvelope.pipeline import StemEnvelopeProcessor

SR = 44100


def _stems(n_stems=4, n=11025):
    t = np.arange(n) / SR
    freqs = (110.0, 440.0, 1500.0, 5000.0)
    return np.stack([0.2 * np.sin(2 * np.pi * f * t) for f in freqs[:n_stems]])


def _processor(n_stems=4, **kwargs):
    band_map = BandMap(n_fft=512)
    return StemEnvelopeProcessor(n_stems=n_stems, band_map=band_map, n_fft=512, hop=128,
                                 **kwargs)


def test_static_evidence_runs_end_to_end():
    processor = _processor()
    stems = _stems()
    result = processor.process(stems, Evidence(level_dbfs=-54.0, event="keyboard"))
    n_frames = result.gains_db.shape[1]
    assert result.mix.shape == (stems.shape[1],)
    assert result.gains_db.shape == (4, n_frames, 24)
    assert result.band_db.shape == (4, n_frames, 24)
    assert np.all(np.isfinite(result.gains_db))
    assert np.all(np.isfinite(result.mix))
    assert abs(result.gains_db).max() > 0.0


def test_missing_evidence_fails_open_to_no_change():
    processor = _processor()
    stems = _stems()
    result = processor.process(stems, Evidence(level_dbfs=None))
    assert np.allclose(result.gains_db, 0.0)
    assert np.allclose(result.curve_db, 0.0)
    assert result.ticks and result.ticks[0][1] is None


def test_callable_evidence_changes_over_time():
    processor = _processor(decision_interval_s=0.05)
    stems = _stems()

    def timeline(t):
        if t < 0.1:
            return Evidence(level_dbfs=-66.0)
        return Evidence(level_dbfs=-50.0, event="keyboard")

    result = processor.process(stems, timeline)
    assert len(result.ticks) >= 3
    assert result.ticks[0][1] is not None
    # the later curve must be hotter than the quiet one
    assert abs(result.curve_db[-1]).max() > abs(result.curve_db[0]).max()
    assert result.peak_gain_db.max() > 0.0


def test_only_stems_with_content_get_gain():
    processor = _processor()
    stems = _stems()
    result = processor.process(stems, Evidence(level_dbfs=-50.0))
    # stem 0 is a 110 Hz tone; its top band (5 kHz) must stay untouched
    top_band = processor.band_map.band_index(5000.0)
    assert np.allclose(result.gains_db[0, :, top_band], 0.0, atol=1e-9)


def test_stem_count_mismatch_raises():
    processor = _processor(n_stems=4)
    with pytest.raises(ValueError):
        processor.process(_stems(n_stems=2), Evidence(level_dbfs=-54.0))


def test_band_map_fft_mismatch_raises():
    with pytest.raises(ValueError):
        StemEnvelopeProcessor(n_stems=4, band_map=BandMap(n_fft=1024), n_fft=512)
