# 环境搭建

整个项目只依赖两样外部东西：**ffmpeg** 和 **一个 whisper 模型**。两样都在 Windows 上踩过坑，这里把绕法写清楚。

## 0. 前置条件

| 项 | 要求 | 说明 |
|---|---|---|
| Python | 3.9+ | 3.13 实测通过 |
| ffmpeg | 6.0+，带 `libass` | 必须要 `ass` 滤镜才能烧双语字幕 |
| 磁盘 | ≥ 5 GB | 模型 1.5 GB + 中间产物 |
| GPU | 不需要 | 纯 CPU 可跑，`medium` 模型约 1.3~1.5x 实时 |

## 1. 装 ffmpeg

**别用 `winget install ffmpeg`**——那版常常不带 `libass`，字幕烧不上去，而且报告是 `No such filter: ass`，很难第一时间想到是打包问题。

下载 BtbN 的 GPL 静态包（自带 libass）：

```
https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
```

解压后把 `bin/` 整个目录放到项目的 `tools/` 下，最终结构：

```
tools/
└── ffmpeg-master-latest-win64-gpl/
    └── bin/
        ├── ffmpeg.exe
        └── ffprobe.exe
```

程序会自动按这个顺序找：`FFMPEG_BIN` 环境变量 → `tools/` 下的常见路径 → `PATH`。

## 2. Python 环境

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e .                # 开发模式安装，改代码不用重装
```

验证：

```bash
lectureforge --version
lectureforge doctor
```

`doctor` 会逐项检查 ffmpeg 路径、`ass`/`afftdn`/`loudnorm` 滤镜、`libx264` 编码器、Python 依赖、模型是否本地化，最后给一句结论。

## 3. 拿模型（三个坑）

`medium` 模型约 1.5 GB。默认让它自己下，但国内 + Windows 的组合会遇到三个问题。

### 坑 1：huggingface 新版走 xet 协议，镜像站返回 401

现象：下载卡在 0%，日志里出现 `cas-server.xethub.hf.co` 和 401。

原因：`huggingface_hub` 新版本默认启用 xet 传输，走的是另一套域名，国内镜像站不支持鉴权。

代码里已经默认处理了：

```python
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
```

### 坑 2：Windows 建不了符号链接，快照目录是空的

现象：模型下完了，但 `snapshots/<hash>/` 里没有文件，只有 `blobs/` 里有。

原因：huggingface_hub 用符号链接做快照管理，Windows 上创建符号链接需要管理员权限或开发者模式。

**解法（推荐）：手工把 blobs 重组成一个普通目录。**

```python
import os, shutil
base = os.path.expanduser("~/.cache/huggingface/hub/models--Systran--faster-whisper-medium/blobs")
dst = "models/faster-whisper-medium"
os.makedirs(dst, exist_ok=True)
for n in os.listdir(base):
    p = os.path.join(base, n)
    size = os.path.getsize(p)
    head = open(p, "rb").read(1)
    if size > 500_000_000:
        name = "model.bin"
    elif head == b"{":
        name = "tokenizer.json" if size > 100_000 else "config.json"
    else:
        name = "vocabulary.txt"
    shutil.copy2(p, os.path.join(dst, name))
```

重组后的目录结构（faster-whisper 认这个）：

```
models/faster-whisper-medium/
├── config.json
├── model.bin            ← 约 1.5 GB
├── tokenizer.json
└── vocabulary.txt
```

程序发现 `models/faster-whisper-medium/` 存在就直接用，不再联网。

### 坑 3：PyAV 版本不兼容，转写直接 TypeError

现象：

```
TypeError: open() got an unexpected keyword argument 'metadata_errors'
```

原因：`faster-whisper` 1.2.x 内部调 `av.open(..., metadata_errors="ignore")`，这个参数在 `av` 19 已经被移除。锁 `av<19` 能解决，但会和别的包打架。

**本项目的解法和它们无关**：音频由 ffmpeg 解码成 float32 numpy 数组再喂给模型，完全不经过 PyAV。所以 av 装哪个版本都不影响。

## 4. 常见故障速查

| 现象 | 原因 | 处理 |
|---|---|---|
| `No such filter: ass` | ffmpeg 不带 libass | 换 BtbN GPL 静态包 |
| 字幕烧上去是方块 | 系统缺字体 | 装微软雅黑，或改 `subtitle.style.font_zh` |
| 转写出来整段繁体 | whisper 中文老问题 | 已内置 opencc 转换，无需干预 |
| 首次运行卡十分钟没动静 | 在下载模型 | 看 `~/.cache/huggingface` 体积是否在涨 |
| `CUDA out of memory` | 配置写了 cuda | 本项目固定 CPU 推理，改 `asr.compute_type: int8` |
| 英文翻译全空 | 没配 API key | 设 `LLM_API_KEY` 环境变量 |

## 5. 目录约定

```
lecture-forge/
├── src/lectureforge/     代码
├── docs/                 文档
├── examples/             术语表样例
├── tools/                ffmpeg（gitignore，自行下载）
├── models/               whisper 模型（gitignore）
├── input/                放原始录像（gitignore）
├── work/                 中间产物（gitignore）
└── output/               成片与字幕（gitignore）
```
