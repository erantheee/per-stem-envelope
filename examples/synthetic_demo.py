#!/usr/bin/env python3
"""Synthetic end-to-end demo: four generated stems, three ambient scenarios.

Everything here is generated with numpy sin/noise, so the example is
reproducible on any machine and redistributes no third-party audio.  That is a
licensing decision as much as a convenience one — see DATA_POLICY.md.

    python examples/synthetic_demo.py
    python examples/synthetic_demo.py --seconds 12 --out build
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stemenvelope import (BandMap, Evidence, RenderResult, StemEnvelopeProcessor)  # noqa: E402

SR = 44100
STEM_NAMES = ("bass", "drums", "vocals", "pad")

#: (start_s, end_s, level_dbfs, event, label)
SCENARIOS = (
    (0.0, 3.0, -66.0, None, "quiet room"),
    (3.0, 5.5, -54.0, "keyboard", "keyboard nearby"),
    (5.5, 8.0, -48.0, None, "street noise"),
)


def build_stems(seconds: float, sr: int = SR) -> np.ndarray:
    """Four crude but spectrally separated stems (see the docstring table)."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(7)

    # bass: 110 Hz + 165 Hz, slow tremolo
    bass = 0.22 * (np.sin(2 * np.pi * 110 * t) + 0.4 * np.sin(2 * np.pi * 165 * t))
    bass *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t)

    # drums: 60 Hz kick every 0.5 s + 7 kHz noise hats on the off-beat
    drums = np.zeros(n)
    for start in np.arange(0.0, seconds, 0.5):
        i = int(start * sr)
        m = min(n - i, int(0.12 * sr))
        env = np.exp(-np.linspace(0, 6, m))
        drums[i:i + m] += 0.12 * env * np.sin(2 * np.pi * 60 * np.arange(m) / sr)
    noise = rng.standard_normal(n)
    hats = np.diff(noise, prepend=0.0)          # first difference = crude high-pass
    hats = 0.60 * hats / max(1e-9, np.abs(hats).max())
    gate = np.zeros(n)
    for start in np.arange(0.25, seconds, 0.5):
        i = int(start * sr)
        m = min(n - i, int(0.05 * sr))
        gate[i:i + m] = np.exp(-np.linspace(0, 4, m))
    drums += hats * gate

    # vocals: three formants around 700 / 1200 / 2600 Hz, 4 Hz syllable AM
    syllable = 0.5 + 0.5 * np.clip(np.sin(2 * np.pi * 4 * t), 0, 1)
    vocals = np.zeros(n)
    for hz, amp in ((700.0, 0.10), (1200.0, 0.07), (2600.0, 0.04)):
        vocals += amp * np.sin(2 * np.pi * hz * t)
    vocals *= syllable

    # pad: a static chord an octave up, wide band, never dominant
    pad = sum(0.05 * np.sin(2 * np.pi * f * t) for f in (440.0, 554.0, 659.0))

    return np.stack([bass, drums, vocals, pad])


def evidence_timeline(seconds: float):  # noqa: ARG001 - keeps the call symmetric
    """Return ``callable(t) -> Evidence`` for the scenario table."""
    def at(t: float) -> Evidence:
        for start, end, level, event, _label in SCENARIOS:
            if start <= t < end:
                return Evidence(level_dbfs=level, event=event)
        level, event = SCENARIOS[-1][2], SCENARIOS[-1][3]
        return Evidence(level_dbfs=level, event=event)

    return at


def write_wav(path: Path, signal, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(signal, dtype=np.float64), -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes((pcm * 32767.0).astype("<i2").tobytes())


def band_report(result: RenderResult, band_map: BandMap, names=STEM_NAMES) -> str:
    """Which bands each stem actually received gain in (the gating evidence)."""
    lines = ["", "per-stem gain (peak |dB|) and the bands that moved (> 0.1 dB):"]
    peaks = result.peak_gain_db
    for index, name in enumerate(names):
        active = np.where(np.abs(result.gains_db[index]).max(axis=0) > 0.1)[0]
        bands = ",".join(f"{band_map.centers_hz[b]:.0f}Hz" for b in active) or "-"
        lines.append(f"  {name:<7} {peaks[index]:5.2f}   [{bands}]")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--out", type=Path, default=Path("build"))
    args = parser.parse_args()

    stems = build_stems(args.seconds)
    band_map = BandMap(n_fft=2048, sample_rate=SR)
    processor = StemEnvelopeProcessor(n_stems=len(STEM_NAMES), band_map=band_map,
                                      n_fft=2048, hop=512, decision_interval_s=1.0)
    result = processor.process(stems, evidence_timeline(args.seconds), sample_rate=SR)

    print(f"stems: {len(STEM_NAMES)}  rate: {SR} Hz  duration: {args.seconds:.1f} s")
    print("scenario timeline:")
    for start, end, level, event, label in SCENARIOS:
        if start >= args.seconds:
            break
        print(f"  {start:4.1f}-{min(end, args.seconds):4.1f}s  "
              f"level={level:6.1f} dBFS  event={str(event):<9} {label}")
    print(f"\ndecision ticks: {len(result.ticks)} "
          f"(fail-open ticks: {sum(1 for _, c in result.ticks if c is None)})")
    print(band_report(result, band_map))

    before = stems.sum(axis=0)
    write_wav(args.out / "demo_before.wav", before)
    write_wav(args.out / "demo_after.wav", result.mix)
    print(f"\nwrote {args.out / 'demo_before.wav'}  (sum of stems)")
    print(f"wrote {args.out / 'demo_after.wav'}   (per-stem gated envelope)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
