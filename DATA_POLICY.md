# 数据与许可策略（DATA_POLICY）

这个文件是给"要把它发到公司内部"这件事写的操作手册。核心结论：

> **公开仓库里不放任何第三方音频、不放任何模型权重、不引用任何非商用数据集的评测数字。**
> 算法本身与 MUSDB 无关，所以这不是妥协，是把两件事分开。

## 1. 四条硬规则

| # | 规则 | 为什么 |
|---|---|---|
| 1 | 不提交 `.wav/.flac/.mp3/.ogg` 等音频 | 仓库是公开的，等于再分发；LFS 也一样 |
| 2 | 不提交 `.onnx/.rknn/.pt/.pth` 权重 | 权重训练数据往往是非商用许可，且体积与许可都不受你的控制 |
| 3 | README / 文档里不出现非商用数据集的评测数字 | "SDR 2.64 on MUSDB" 这类数字本身就是衍生使用 |
| 4 | 演示素材必须可再生 | 合成信号（`examples/synthetic_demo.py`）或 CC0/CC-BY 素材，附署名 |

规则 1/2 由 `tests/test_data_policy.py` 在 CI 里强制执行：一旦有人提交音频或
权重，测试直接失败。

## 2. 为什么 MUSDB18-HQ 不能用

* MUSDB18 / MUSDB18-HQ 的许可原文：*"provided for educational purposes only …
  should not be used for any commercial purpose without the express permission of
  the copyright holders"*（[Zenodo](https://zenodo.org/records/3338373)）。
* 其中 46 首来自 MedleyDB，**CC BY-NC-SA 4.0**；100 首来自 DSD100 / Mixing
  Secrets 多轨库。
* 派生物（用 MUSDB 训练的权重、MUSDB 上的评测表）同样落在非商用范围。参见
  [AudioShake 关于"代码 MIT ≠ 权重可商用"的分析](https://audioshake.ai/post/can-demucs-be-used-commercially-musdb18-model-weights-licensing)。
* **NC 的含义包含"公司内部使用"**：NC 限制的是使用目的，不是分发渠道。所以
  "我只是发给同事看"不构成豁免。

## 3. 合法的替代素材

| 素材 | 许可 | 用法 |
|---|---|---|
| **合成信号**（本仓库默认） | 你自己的代码 | 无任何限制；可复现、可进 CI |
| **Slakh2100 / BabySlakh** | CC BY 4.0（可商用，需署名） | 真实多轨（合成器渲染的 MIDI），适合做客观对比。**下载脚本，不要提交音频** |
| **CC0 音效/循环**（freesound CC0 等） | CC0 | 自己拼多轨，可提交极小样本（几十 KB 以内）并附来源 |
| **CC-BY 音乐 + 运行时分离** | CC BY x.0 | 用 Demucs 在本地分离，只提交脚本与结果描述，不提交音频 |

Slakh 许可依据：[Zenodo Slakh2100 / BabySlakh](https://zenodo.org/records/4603870)（Creative Commons Attribution 4.0 International）。

## 4. 如果同事要看"真实效果"

推荐做法（合规）：

1. 在本机用 Demucs 分离一首 CC-BY 音乐，或直接读 Slakh 的多轨；
2. 跑本仓库的处理，把 `gains_db` 与听感差异写进内部文档；
3. 仓库里只提交**脚本 + 一行复现命令**，例如
   `python examples/real_audio.py --stems /path/to/stems`。

不推荐：把分离后的 stem 一起提交、把 MUSDB 评测表贴进 README、把微调过的权重
传上来当 demo。

## 5. 模型权重

本仓库不含任何权重，也不需要。如果你以后要加一个可选的分离器：

* Demucs 代码 MIT，但**预训练权重是 CC-BY-NC 4.0**；
* 任何基于 MUSDB 训练的权重都不能随公开仓库分发；
* 想商用就走：自采多轨训练 / 商业分轨 API / 无神经网络的分频段方案。

## 6. 提交前检查清单

- [ ] `git status` 里没有 `*.wav/*.onnx/*.pt`
- [ ] `pytest -q` 通过（含 `test_data_policy`）
- [ ] README 里没有非商用数据集的数字
- [ ] 新增的演示素材要么是合成信号，要么有明确的 CC0/CC-BY 署名
- [ ] 若引用了外部数据集，在 `NOTICE` 里写清来源与许可
