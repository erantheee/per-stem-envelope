# Khadas 方案（Edge2 / RK3588 RKNPU）

Edge2 上跑**完整链路**：分轨（NPU）+ 环境判断 + 逐轨包络 + STFT 施加。

| 层 | 组件 | 位置 |
|---|---|---|
| 分轨 | BandSplitNet 六轨，554k 参数 | `weights/six_stem_band_split.pt`（研究用途，见 `weights/LICENSE-WEIGHTS.md`）、`weights/six_stem_rk-6s.onnx` |
| 板端证据 | DMIC 24 段 + 电平 + event + policy_state | `board_adapter.py::evidence_from_board` / `read_evidence_file` |
| 包络 | `PerStemEnvelope` + `TuningRecipe` | `src/stemenvelope/` |
| 施加 | STFT 逐轨增益 + residual 不动 | `src/stemenvelope/dsp.py` |
| 遥测 | `/tmp/tuning-latest.json` | `board_adapter.py::telemetry_from_result` |
| 档位 | `/tmp/tuning-control.json`（`depth` / `masking_dose` / `max_positive_db`） | `control_payload` / `clamp_control` |

## 接口契约（与现有 worker 一致）

```python
from board_adapter import (evidence_from_board, read_evidence_file,
                          control_payload, clamp_control, telemetry_from_result)

evidence = read_evidence_file("/tmp/dmic-latest.json", max_age_s=2.5)  # 过期 = 无证据
result = processor.process(stems, evidence, sample_rate=44100)
telemetry = telemetry_from_result(result, band_map, modes=control_payload("basic"))
```

* 证据过期、字段缺失、`level_dbfs = NaN` → 全部 fail-open（不衰减、不抬升）。
* `control_payload` 的三个数与 worker 侧夹取范围完全一致（depth 1–4、dose 0.5–2、
  max_positive 1–6 dB）；`clamp_control` 会把越界值夹回来。
* `telemetry_from_result` 输出的是面板需要的键：`band_db / stem_db / stem_peak_db / control`。

## 与私有仓库的关系

`khadas-ows-node` 里的 `worker.py` 仍然是音频通路；这里只提供**适配层**，不复制 worker。
接法是把 `_s3_output` 里的内容特征换成 `stemenvelope.dsp.stem_band_db`，把 `BoardTuner`
换成 `PerStemEnvelope`——约 50 行，尚未落到那个仓库（保持其只读）。

## 模型

`weights/six_stem_band_split.pt` 与 `six_stem_rk-6s.onnx` 都**不可商用**：
训练数据是 MUSDB18-HQ 混音 + htdemucs_6s 伪标签。24 MB 的 fp16 `.rknn` 没有入库——
它是设备相关的、可从 ONNX 在板端重新转换。

要可商用的分轨权重只有三条路：自采多轨重训 / 商业 API / 非神经频段切分。
