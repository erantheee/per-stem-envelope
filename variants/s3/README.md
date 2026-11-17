# S3 方案（ESP32-S3）

S3 上跑的是**判断 + 执行**，不跑分轨。

| 层 | 在 S3 上跑什么 | 现状 |
|---|---|---|
| 情境判断 | `ContextNet v0.2-s3`（92-32-16-9，3,657 参数，3,600 MACs/次，14.6 kB f32），10 Hz | 权重在 `weights/contextnet_v0.2-s3.{pt,json}`；numpy 参考实现在 `python/contextnet.py`，与原始 torch 模型有黄金对拍 |
| 类别→环境 | `python/env_map.py`：9 类 → `quiet/speech/babble/street/office/wind/keyboard` | 映射是策略选择，写在文件头注释里 |
| 包络执行 | `src/stemenvelope_s3.c`：presence 门控 + 逐段 slew + 硬边界，零分配 | C99，与 `src/stemenvelope/envelope.py` 逐行对齐，`make check` 对拍 12 步误差 ≤ 1e-6 dB |
| 分轨 | **不跑** | S3 跑不动 6 轨 55 万参数的分轨模型（板上 int8 实测单窗 3.4 s，质量崩塌）；要么放系统侧，要么做非神经的频段切分 |

## 为什么分轨不在 S3

不是"优化得不够"，是量级问题：六轨 BandSplitNet 有 554k 参数、要 4 s 窗的 STFT，int8 量化
在 RKNPU 上尚且 3.4 s/窗且 mask 失真 0.178（不可用）。ESP32-S3 是 240 MHz + 8 MB PSRAM，
差两个数量级。所以 S3 方案是"**别人分好轨、我调包络**"，或者"不分轨，按频段调混音"。

## 构建与对拍

```bash
make -C variants/s3 check
# c/python parity: 12 steps, worst error 0.000001 dB (tol 0.000200)
# bounds ok / arg checks ok
```

`tests/golden.h` 由 `tools/make_goldens.py` 从 Python 参考实现生成；C 侧漂移会被
`tests/test_variants_s3.py` 抓到。

## 固件接入面

```c
#include "stemenvelope_s3.h"

static float gains[6 * 24];
se_envelope_t env;
se_init(&env, 6, 24, gains);            /* 0.65/0.85 s, 24 dB 窗, -80 dB 静音门, +3/-12 dB */

/* 每个控制节拍（建议 10–20 ms 一次，或每个 band 帧一次） */
se_step(&env, stem_band_db, curve_db, dt_s, evidence_ok);
/* env.gains 就是 6×24 的 dB 增益，交给你的 filterbank */
```

`evidence_ok = 0` 时增益向 0 dB 衰减（fail-open），不是冻结。

## 注意事项

- `stem_band_db` 是**每条轨自己的**频段电平（dB）。如果你的系统里没有分轨，就把整条混音
  当一个 stem 传（`n_stems = 1`），这时门控退化为"该频段有没有内容"。
- `curve_db` 由 `TuningRecipe` 产出（24 段）。S3 上如果要省算力，可以直接用常量曲线，
  只让 exposure 缩放——但那样就不是"环境响应"了。
- 24 段的对数频带映射由你的 filterbank 决定；`BandMap` 只在 Python 侧用。
