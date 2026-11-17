# per-stem-envelope

Per-stem, per-band dynamic envelope for separated audio.

When the room gets noisy, adjust **each stem** in **the bands that stem actually
occupies** — smoothly, bounded, and never when you don't know what is going on.

Pure numpy. No models, no audio I/O, no transport. MIT.

```
                        ambient evidence                stems (any separator)
                               │                                 │
                     ┌─────────▼─────────┐          ┌────────────▼────────────┐
                     │ exposure/dominance│          │ per-stem band levels    │
                     │ (0..1, calibrated)│          │ (24 log bands, per frame)│
                     └─────────┬─────────┘          └────────────┬────────────┘
                               │                                 │
                     ┌─────────▼─────────┐                       │
                     │  tuning recipe    │── one bounded curve ──┤
                     └───────────────────┘                       │
                                                     ┌───────────▼───────────┐
                                                     │ content gate × curve  │
                                                     │ independent slew      │
                                                     └───────────┬───────────┘
                                                                 │
                                                        gain [stem, band]
```

## The problem it solves

A separator hands you 4-6 stems. Ambient noise changes. You want the vocal to
come forward, or the music bed to step back — but:

* a **global** EQ/compressor cannot tell the vocal from the cymbal;
* a **per-stem scalar** gain cannot tell the vocal's 1.2 kHz formant from its
  8 kHz sibilance;
* applying an EQ curve **to every stem** is not "per stem" at all — it lifts the
  bands where a stem has nothing, which is exactly where separation residue
  lives, and residue lifted by 3 dB is audible as mush;
* anything that reacts per STFT frame pumps.

So: compute the curve once from ambient evidence, then gate it per stem by that
stem's own content, smooth every (stem, band) cell independently, and stay inside
hard bounds.

## Install

```bash
pip install -e ".[dev]"
pytest -q                     # 55 tests
python examples/synthetic_demo.py
```

The demo generates four stems with numpy (bass / drums / vocals / pad), walks
through three ambient scenes, and writes `build/demo_before.wav` and
`build/demo_after.wav`. No third-party audio is downloaded or required — see
[DATA_POLICY.md](DATA_POLICY.md).

```
$ python examples/synthetic_demo.py
stems: 4  rate: 44100 Hz  duration: 8.0 s
scenario timeline:
   0.0- 3.0s  level= -66.0 dBFS  event=None      quiet room
   3.0- 5.5s  level= -54.0 dBFS  event=keyboard  keyboard nearby
   5.5- 8.0s  level= -48.0 dBFS  event=None      street noise

per-stem gain (peak |dB|) and the bands that moved (> 0.1 dB):
  bass     1.39   [71Hz,141Hz]
  drums    0.58   [71Hz,141Hz,7020Hz,8553Hz,10677Hz,13638Hz]
  vocals   1.60   [696Hz,1171Hz,2503Hz]
  pad      1.29   [452Hz,567Hz,696Hz]
```

Read that table as the whole point: four stems, four different band sets, from
one shared curve.

## API

```python
import numpy as np
from stemenvelope import BandMap, Evidence, StemEnvelopeProcessor

stems = np.stack([bass, drums, vocals, pad])        # [4, samples], float

processor = StemEnvelopeProcessor(
    n_stems=4,
    band_map=BandMap(n_fft=2048, sample_rate=44100),
    n_fft=2048, hop=512,
    decision_interval_s=1.0,
)

# evidence: a constant snapshot, or a callable t -> Evidence
def ambient(t):
    return Evidence(level_dbfs=-54.0, event="keyboard")

result = processor.process(stems, ambient, sample_rate=44100)
result.mix          # [samples] processed mix
result.gains_db     # [stems, frames, bands] what was actually applied
result.peak_gain_db # [stems] telemetry
```

Lower-level pieces, if you already have your own STFT pipeline:

| Module | What it is |
|---|---|
| `bands.BandMap` | 24 log bands ↔ FFT bins (any band count, any FFT size) |
| `evidence.Evidence` | validated ambient snapshot — the only I/O boundary |
| `recipe.TuningRecipe` | evidence → one bounded per-band curve |
| `decision.EnvironmentTracker` | label hysteresis (`hold_windows`, `fast_windows`) |
| `decision.apply_environment` | label → per-stem attenuation table |
| `envelope.PerStemEnvelope` | content gate + per-band slew → `[stems, bands]` dB |
| `dsp` | Hann STFT/ISTFT, gain application, stem summing |

## The four knobs that matter

| Knob | Default | What it really says |
|---|---|---|
| `presence_window_db` | 24 dB | How deep into a stem's own spectrum the envelope may reach. 24 dB = dominant content only; a drum stem's cymbals can sit 60 dB below the kick and stay untouched **by design**. |
| `tau_rise_s` / `tau_fall_s` | 0.65 / 0.85 s | How fast the gain may move. Slow enough not to chase hats, fast enough to track a door closing. |
| `limit_db` (recipe) | 4 dB | Ceiling of the curve before gating. |
| `max_boost_db` / `max_cut_db` | +3 / −12 dB | Hard bounds on the applied gain, independent of the curve. |

## Design notes

**Attenuation-only in the environment layer.** Boosting estimated stems
re-amplifies separation leakage; cutting a competitor is always safer than
lifting the target when both come from the same imperfect separator. The default
`DEFAULT_MODIFIERS` table is therefore mostly empty, with offsets only for
masking-type events (keyboard, a nearby talker).

**Decision rate ≠ frame rate.** The recipe runs at ~1 Hz (a human cannot hear the
difference between a 0.8 s and a 1.2 s decision period, and per-frame decisions
only make the gains jitter). The envelope runs per STFT frame (11.6 ms at
44.1 kHz / hop 512). One second of decision period plus 0.65–0.85 s of slew
settles in roughly 2 s — that is the intended feel, not a bug.

**An absolute silence floor, not only a relative gate.** A digitally silent stem
has a flat spectrum, so relative gating alone (`peak − 24 dB`) gives it
`presence = 1` in every band and the full curve. `silence_floor_db = -80 dB` of
band power rules that out; it is far above numerical zero (−180 dB) and far below
anything audible, so the value is comfortable, not delicate.

**Fail-open everywhere.** Missing level, non-finite numbers, unknown labels,
stale evidence → zero gain, identity. A lost evidence stream must return to
transparency, never to a latched cut.

**Cost.** Per frame: `n_stems × 24` cells, one multiply, one exp per cell. The
STFT and the separator are the expensive parts. The control loop is noise.

## Where this came from, what was removed

Extracted from an internal edge product (6-stem BandSplitNet on an RK3588-class
board, driven by a DMIC evidence stream). This repository keeps only the layer
that is separable from the hardware:

| Kept | Deliberately left behind |
|---|---|
| band map, recipe, gating, slew, STFT application | the separator model and its RKNN conversion chain |
| evidence *contract* (validated snapshot) | DMIC capture, HTTP/file transport, node policy state machine |
| attenuation-only environment table | the style/content classifier |

Consequence: this package contains **no separator**. Feed it stems from Demucs,
Open-Unmix, a commercial API, or real multitrack files.

## Running on small hardware

The control loop is cheap; the separator is not. On an MCU you can run the
envelope (a few hundred flops per frame, no allocation if you preallocate) but you
cannot run a 6-stem neural separator at interactive latency — budget a
system-side separator, or a non-neural band split. Numbers in this repository are
from a desktop CPU; nothing here claims a microcontroller profile.

## Data and licensing

* **This repository redistributes no third-party audio and no model weights.**
  The demo is generated by `examples/synthetic_demo.py`.
* MUSDB18-HQ (and MUSDB18) are **non-commercial, educational use only** and
  contain MedleyDB (CC BY-NC-SA) material. Nothing here is derived from them, and
  they must not be added. See [DATA_POLICY.md](DATA_POLICY.md) for the rules and
  for the legally usable alternatives (Slakh2100 / BabySlakh are CC BY 4.0).
* Demucs **weights** are CC-BY-NC 4.0 even though the code is MIT. If you
  separate audio with them, do not commit the stems or the weights.
* All dBFS anchors in `recipe.TuningRecipe` are calibrations from one microphone
  array. The *shape* is portable; the *anchors* are not. Recalibrate.

## Tests

```
tests/test_bands.py       band map, interpolation, power mapping
tests/test_dsp.py         STFT roundtrip, gain application, stem summing
tests/test_recipe.py      exposure, masking, fail-open, bounds
tests/test_decision.py    hysteresis, fail-open labels, table scaling
tests/test_envelope.py    content gating, silence floor, slew, bounds
tests/test_pipeline.py    end-to-end, fail-open, stem-count guards
tests/test_data_policy.py fails the build if audio/weights get committed
```

## License

MIT — see [LICENSE](LICENSE). No warranty; no claim of fitness for a product.
