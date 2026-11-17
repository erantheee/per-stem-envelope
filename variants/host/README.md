# 上位机方案

上位机（Mac / PC）只做两件事：**让人类选档位**、**显示板端实际在做什么**。
它不能阻塞音频通路，所以遥测走文件或短超时 HTTP，解析全部防御性。

```
上位机                          板端
  │  /tmp/tuning-control.json   →   depth / masking_dose / max_positive_db
  │  /tmp/tuning-latest.json    ←   band_db / stem_db / stem_peak_db / control
```

## 用法

```python
from tuning_bridge import TuningBridge

bridge = TuningBridge()                 # 默认两个 /tmp 路径
bridge.set_mode("basic")                # off / basic / aggressive
print(bridge.summary())
# {'mode': 'PER_STEM_ENVELOPE', 'stale': False, 'peak_db': 1.6,
#  'stems': {'0': [...24 个数], ...}, 'control': {'depth': 1.3, ...}}
```

读 HTTP 遥测（非阻塞，0.4 s 超时）：

```python
TuningBridge(telemetry_url="http://172.16.20.18:8787/v1/tuning")
```

## 行为约定

* `set_mode` 原子写（先写 `.writing` 再 `os.replace`），板端永远读不到半个文件；
* `telemetry()` 超过 3 s 未更新返回 `None`，`summary()` 标 `stale: True`；
* JSON 坏、文件缺、URL 不通 → 返回 `None`，不抛异常——上位机不允许把板端的故障变成
  自己的崩溃；
* 档位表 `MODES` 与 `variants/khadas/board_adapter.py::MODE_CONTROLS` 一致；
  `off` 是真直通（depth 1.0 + max_positive 1.0），不是"关闭包络但保留静态曲线"。

## 与旧面板的关系

旧面板把档位写进 `/tmp/tuning-control.json` 后需要重启 worker 才生效（热加载未做）。
本模块只负责写文件；热加载属于板端 worker 的事，不在这一层假装实现。
