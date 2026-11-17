# per-stem-envelope（中文）

分轨之后，按**每条轨自己在哪些频段有内容**，逐频段动态调整包络。

房间变吵时，把"某条轨"在某些频段上推一点/压一点——而不是把一条 EQ 曲线糊在整个混音上。
纯 numpy，无模型、无音频 I/O、无传输层。MIT。

## 它解决什么

分轨器给你 4–6 条轨，环境噪声在变，你想让 vocal 靠前、音乐床靠后。但：

- **全局** EQ/压缩分不清 vocal 和镲片；
- **整轨一个标量增益**分不清 vocal 的 1.2 kHz 共振和 8 kHz 齿音；
- 把同一条 EQ 曲线**贴到每条轨**上并不叫"分轨调节"——它会把该轨**没有内容**的频段一起抬起来，
  而那正好是分离残留（residue）所在，抬 3 dB 就是一片糊；
- 逐 STFT 帧做决策一定抽吸（pumping）。

所以：曲线只算一次（来自环境证据），然后**按每条轨自己的内容做门控**，
每个（轨, 频段）单元独立平滑，并且始终在硬边界内。

## 安装与运行

```bash
pip install -e ".[dev]"
pytest -q                     # 55 个测试
python examples/synthetic_demo.py
```

demo 用 numpy 生成四条轨（bass / drums / vocals / pad），走三个环境场景，输出
`build/demo_before.wav` 与 `build/demo_after.wav`。不需要任何第三方音频。

```
per-stem gain (peak |dB|) and the bands that moved (> 0.1 dB):
  bass     1.39   [71Hz,141Hz]
  drums    0.58   [71Hz,141Hz,7020Hz,8553Hz,10677Hz,13638Hz]
  vocals   1.60   [696Hz,1171Hz,2503Hz]
  pad      1.29   [452Hz,567Hz,696Hz]
```

这张表就是全部卖点：四条轨、四组不同频段、来自同一条曲线。

## 用法

```python
import numpy as np
from stemenvelope import BandMap, Evidence, StemEnvelopeProcessor

processor = StemEnvelopeProcessor(
    n_stems=4, band_map=BandMap(n_fft=2048, sample_rate=44100),
    n_fft=2048, hop=512, decision_interval_s=1.0)

result = processor.process(
    np.stack([bass, drums, vocals, pad]),           # [4, samples]
    lambda t: Evidence(level_dbfs=-54.0, event="keyboard"),
    sample_rate=44100)

result.mix, result.gains_db, result.peak_gain_db
```

已有自己的 STFT 管线的，直接用底层件：

| 模块 | 作用 |
|---|---|
| `bands.BandMap` | 24 段对数带 ↔ FFT bin（带数、FFT 长度可配） |
| `evidence.Evidence` | 校验过的环境快照——唯一的 I/O 边界 |
| `recipe.TuningRecipe` | 证据 → 一条有界的 24 段曲线 |
| `decision.EnvironmentTracker` | 环境标签迟滞（`hold_windows` / `fast_windows`） |
| `decision.apply_environment` | 标签 → 逐轨衰减表 |
| `envelope.PerStemEnvelope` | 内容门控 + 逐段 slew → `[轨, 段]` dB |
| `dsp` | Hann STFT/ISTFT、增益施加、逐轨求和 |

## 四个真正重要的旋钮

| 旋钮 | 默认 | 它实际在说什么 |
|---|---|---|
| `presence_window_db` | 24 dB | 允许进入该轨自身频谱多深。24 dB = 只动主内容；鼓轨的镲片可能比底鼓低 60 dB，**按设计不动** |
| `tau_rise_s` / `tau_fall_s` | 0.65 / 0.85 s | 增益能多快移动。慢到不追镲片，快到能跟上门声 |
| `limit_db`（recipe） | 4 dB | 曲线本身的上限 |
| `max_boost_db` / `max_cut_db` | +3 / −12 dB | 施加增益的硬边界 |

## 设计取舍

**环境层只做衰减。** 抬升估计出来的轨 = 抬升分离泄漏；在同一个不完美分轨器下，
压竞争对手永远比抬目标安全。所以默认 `DEFAULT_MODIFIERS` 基本是空的，
只对掩蔽类事件（键盘、近场人声）给偏移。

**决策率 ≠ 帧率。** 曲线 1 Hz 更新（人耳分不出 0.8 s 和 1.2 s 的决策周期，
逐帧决策只会抖），包络逐 STFT 帧更新（44.1k / hop 512 = 11.6 ms）。
1 s 决策 + 0.65–0.85 s slew ≈ 2 s 到位——这是设计意图，不是缺陷。

**必须有绝对静音门。** 数字静音的轨频谱是平的，只做相对门控（`peak − 24 dB`）
会让它每个频段 `presence = 1`，吃满整条曲线。`silence_floor_db = -80 dB` 排掉这种
情况；它远高于数值零（−180 dB）、远低于任何可闻内容。

**全程 fail-open。** 缺电平、非有限值、未知标签、证据过期 → 增益归零（直通）。
证据流断掉必须回到透明，绝不能停在一个已 latch 的衰减上。

**开销。** 每帧 `轨数 × 24` 个单元，一次乘法、一次 exp。STFT 和分轨器才是成本大头。

## 从哪里来、抽掉了什么

抽自一个端侧产品（RK3588 级板子上的六轨 BandSplitNet + DMIC 证据流）。
仓库只保留与硬件无关的那一层：

| 保留 | 有意留下 |
|---|---|
| 频带映射、配方、门控、slew、STFT 施加 | 分轨模型及其 RKNN 转换链 |
| 证据**契约**（校验过的快照） | DMIC 采集、HTTP/文件传输、node 策略状态机 |
| 只衰减的环境表 | 风格/内容分类器 |

因此本包**不含分轨器**：喂 Demucs、Open-Unmix、商业 API 的输出，或真实多轨文件都行。

## 小硬件上能不能跑

控制环路便宜，分轨器不便宜。MCU 上可以跑包络（每帧几百 flops，预分配后零分配），
但跑不动六轨神经分轨的交互延迟——要么把分轨放到系统侧，要么换成非神经的频段切分。
本仓库的数字来自桌面 CPU，**不声称任何微控制器性能指标**。

## 数据与许可（重点）

- **本仓库不再分发任何第三方音频、不含任何模型权重。** demo 由
  `examples/synthetic_demo.py` 生成。
- MUSDB18-HQ 是**仅教育用途、非商用**，且含 MedleyDB（CC BY-NC-SA）素材。
  这里没有任何东西来自它，也不允许加进来。规则与合法替代素材见
  [DATA_POLICY.md](DATA_POLICY.md)：Slakh2100 / BabySlakh 是 CC BY 4.0。
- Demucs 的**代码**是 MIT，但**预训练权重是 CC-BY-NC 4.0**；用它分离出的 stem
  不要提交。
- `recipe` 里所有 dBFS 锚点都是某一支麦克风阵列上的标定值。**形状**可移植，
  **锚点**不可移植，换硬件必须重标。

## 许可

MIT，见 [LICENSE](LICENSE)。无担保，不声称适用于任何具体产品。
