"""Noise-aware per-stem dynamic envelope.

Public surface, smallest useful set::

    from stemenvelope import BandMap, Evidence, StemEnvelopeProcessor

    processor = StemEnvelopeProcessor(n_stems=4)
    result = processor.process(stems, Evidence(level_dbfs=-56.0, event="keyboard"))
    result.mix, result.gains_db

Layers, bottom-up:

* :mod:`stemenvelope.bands`     — FFT bins <-> 24 log bands
* :mod:`stemenvelope.evidence`  — validated ambient snapshot (the I/O boundary)
* :mod:`stemenvelope.recipe`    — evidence -> one bounded per-band curve
* :mod:`stemenvelope.decision`  — environment label -> per-stem attenuation
* :mod:`stemenvelope.envelope`  — content gating + per-band attack/release
* :mod:`stemenvelope.dsp`       — STFT, gain application, stem summing
* :mod:`stemenvelope.pipeline`  — the above, wired together
"""
from .bands import DEFAULT_CENTERS_HZ, BandMap
from .decision import (DEFAULT_MODIFIERS, ENVIRONMENTS, EnvironmentTracker,
                       NOISE_BASE_FACTOR, NOISE_CEIL_DBFS, NOISE_FLOOR_DBFS,
                       apply_environment, noise_exposure)
from .dsp import (DEFAULT_HOP, DEFAULT_N_FFT, apply_band_gains, frame_count,
                  istft, render_stems, stem_band_db, stft)
from .envelope import PerStemEnvelope
from .evidence import Evidence
from .pipeline import RenderResult, StemEnvelopeProcessor
from .recipe import DEFAULT_CONTENT_CURVE_DB, DEFAULT_MASK_K, TuningRecipe
from .smoothing import slew

__version__ = "0.1.0"

__all__ = [
    "BandMap", "DEFAULT_CENTERS_HZ",
    "Evidence",
    "TuningRecipe", "DEFAULT_CONTENT_CURVE_DB", "DEFAULT_MASK_K",
    "EnvironmentTracker", "ENVIRONMENTS", "DEFAULT_MODIFIERS", "apply_environment",
    "noise_exposure", "NOISE_FLOOR_DBFS", "NOISE_CEIL_DBFS", "NOISE_BASE_FACTOR",
    "PerStemEnvelope",
    "stft", "istft", "apply_band_gains", "stem_band_db", "render_stems",
    "frame_count", "DEFAULT_N_FFT", "DEFAULT_HOP",
    "StemEnvelopeProcessor", "RenderResult",
    "slew", "__version__",
]
