# 权重许可（WEIGHTS LICENCE）

**代码是 MIT，权重不是。** 这两件事必须分开读。

| 目录/文件 | 许可 | 可否商用 | 训练数据 |
|---|---|---|---|
| `src/`、`variants/`、`tests/`、`examples/` | MIT（见 `LICENSE`） | 可以 | — |
| `weights/contextnet_v0.2-s3.pt` / `.json` | MIT | 可以 | **自采** DMIC 录音（`记录/房间本底`、`记录/键盘风声`），无第三方数据集 |
| `weights/six_stem_band_split.pt` | **CC BY-NC 4.0（研究用途）** | **不可以** | MUSDB18-HQ 混音 + htdemucs_6s 伪标签（两者都是非商用） |
| `weights/six_stem_rk-6s.onnx` | **CC BY-NC 4.0（研究用途）** | **不可以** | 同上（由 `six_stem_band_split.pt` 导出） |

## 为什么分轨权重是 NC

* MUSDB18-HQ 的原文许可：*"provided for educational purposes only … should not be
  used for any commercial purpose"*（[Zenodo](https://zenodo.org/records/3338373)），其中 46 首来自
  MedleyDB（CC BY-NC-SA 4.0）。
* 教师模型 htdemucs_6s 的**权重**是 CC-BY-NC 4.0（代码才是 MIT）。
* 用非商用数据训练的权重属于衍生使用，不因为"是我自己训练的"就解除限制。
  这也是为什么它不能继承本仓库的 MIT。

## "不可商用"具体意味着

允许：

* 研究、评测、内部技术验证（不进入产品）；
* 复现论文/博客里的数值；
* 拿它当教师模型去蒸馏**你自己有权使用的数据**训练的模型。

不允许：

* 装进出货的产品或 OTA 包；
* 用公司客户数据微调后商用；
* 对外声称该权重可商用，或把它标成 MIT/Apache。

## 分轨权重的替代路径

要一份**可商用**的分轨权重，只有三条路：自采多轨重训、用商业分轨 API、
不用神经网络而做频段切分。可商用的公开多轨素材推荐 **Slakh2100 / BabySlakh**
（CC BY 4.0，署名即可），见 [../DATA_POLICY.md](../DATA_POLICY.md)。

## 熵/哈希

每个文件的 sha256、参数量、输入形状写在 `MANIFEST.json`。CI 会校验哈希：权重被
替换或篡改会直接失败。
