"""Offline pipeline: stems + ambient evidence -> processed mix.

Everything is load bearing here except the plumbing; the plumbing exists so a
caller does not have to re-derive the tick rate, the frame alignment and the
fail-open handling.

Two timing rules make the behaviour understandable:

* the **decision** runs at a fixed human rate (default 1 s).  A human cannot hear
  the difference between a 0.8 s and a 1.2 s decision period, and running it per
  STFT frame only makes the gains jitter.
* the **envelope** runs per STFT frame (default 11.6 ms at 44.1 kHz / hop 512), so
  the gain moves smoothly between decisions instead of stepping.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bands import BandMap
from .dsp import (DEFAULT_HOP, DEFAULT_N_FFT, render_stems, stem_band_db, stft)
from .envelope import PerStemEnvelope
from .recipe import TuningRecipe


@dataclass
class RenderResult:
    """What the pipeline produced, including the state you need to inspect it."""

    mix: np.ndarray
    gains_db: np.ndarray                      # [n_stems, frames, bands]
    band_db: np.ndarray                       # [n_stems, frames, bands] content
    curve_db: np.ndarray                      # [frames, bands] applied recipe curve
    n_fft: int = DEFAULT_N_FFT
    hop: int = DEFAULT_HOP
    sample_rate: float = 44100.0
    ticks: list = field(default_factory=list)  # (time_s, curve_or_None) per decision

    @property
    def peak_gain_db(self) -> np.ndarray:
        return np.max(np.abs(self.gains_db), axis=(1, 2))


class StemEnvelopeProcessor:
    """End-to-end processor.  Construct once, reuse across chunks/files."""

    def __init__(self, n_stems: int, band_map: BandMap = None,
                 recipe: TuningRecipe = None, envelope: PerStemEnvelope = None,
                 n_fft: int = DEFAULT_N_FFT, hop: int = DEFAULT_HOP,
                 decision_interval_s: float = 1.0) -> None:
        self.band_map = band_map or BandMap(n_fft=n_fft)
        self.recipe = recipe or TuningRecipe()
        if self.recipe.n_bands != self.band_map.n_bands:
            raise ValueError("recipe and band map disagree on band count")
        self.envelope = envelope or PerStemEnvelope(n_stems, self.band_map.n_bands)
        if self.envelope.n_stems != int(n_stems):
            raise ValueError("envelope and processor disagree on stem count")
        if int(n_fft) != self.band_map.n_fft:
            raise ValueError("band_map was built for a different n_fft")
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.decision_interval_s = max(float(decision_interval_s), 1e-3)

    def process(self, stems, evidence_at, sample_rate: float = None) -> RenderResult:
        """Run the pipeline.

        Parameters
        ----------
        stems:
            ``[n_stems, n_samples]`` audio; stems may be synthesised, produced by
            a separator, or real multitrack files.
        evidence_at:
            An ``Evidence`` (constant for the whole file) or a callable
            ``t_seconds -> Evidence``.  Return ``None`` or invalid evidence to
            exercise the fail-open path.
        """
        audio = np.atleast_2d(np.asarray(stems, dtype=np.float64))
        n_stems, n_samples = audio.shape
        if n_stems != self.envelope.n_stems:
            raise ValueError(
                f"envelope is for {self.envelope.n_stems} stems, got {n_stems}")
        sr = float(sample_rate) if sample_rate else float(self.band_map.sample_rate)

        spectra = [stft(audio[i], self.n_fft, self.hop) for i in range(n_stems)]
        frames = spectra[0].shape[0]
        band_db = stem_band_db(audio, self.band_map, self.n_fft, self.hop)
        dt = self.hop / sr

        gains = np.zeros((n_stems, frames, self.band_map.n_bands), dtype=np.float64)
        curve_track = np.zeros((frames, self.band_map.n_bands), dtype=np.float64)
        ticks = []
        curve = np.zeros(self.band_map.n_bands, dtype=np.float64)
        active = False
        next_decision = 0.0
        for frame in range(frames):
            now = frame * dt
            if now >= next_decision - 1e-9:
                evidence = evidence_at(now) if callable(evidence_at) else evidence_at
                fresh = self.recipe.band_curve(evidence)
                if fresh is None:
                    curve = np.zeros(self.band_map.n_bands, dtype=np.float64)
                    active = False
                    ticks.append((now, None))
                else:
                    curve = fresh
                    active = True
                    ticks.append((now, fresh.copy()))
                next_decision = now + self.decision_interval_s
            gains[:, frame, :] = self.envelope.step(band_db[:, frame, :], curve, dt,
                                                    active=active)
            curve_track[frame] = curve

        mix = render_stems(audio, gains, self.band_map, self.n_fft, self.hop,
                           spectra=spectra)
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 0.995:
            mix = mix * (0.995 / peak)
        return RenderResult(mix=mix, gains_db=gains, band_db=band_db,
                            curve_db=curve_track, n_fft=self.n_fft, hop=self.hop,
                            sample_rate=sr, ticks=ticks)
