# -*- coding: utf-8 -*-
"""ffmpeg 封装：定位可执行文件、探测媒体信息、解码、渲染。

有一件事在这里刻意做了：**音频不用 PyAV 解码**。
faster-whisper 1.2.x 内部调 `av.open(..., metadata_errors=...)`，
该参数在 av 19 已被移除，直接 TypeError。自己用 ffmpeg 出 float32，
既绕开版本雷，也顺手拿到原始波形给卡壳判定做能量分析。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Sequence

_CACHE: Dict[str, str] = {}

# 项目自带 tools/ 目录下常见的位置（下载静态包解压后的结构）
_BUNDLED = (
    "tools/ffmpeg-master-latest-win64-gpl/bin",
    "tools/ffmpeg/bin",
    "tools/bin",
)


def _project_root() -> str:
    # src/lectureforge/ffmpeg.py → 上三级
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find(name: str) -> Optional[str]:
    """按 环境变量 → 项目自带 → PATH 的顺序找"""
    env_key = f"{name.upper()}_BIN"
    p = os.environ.get(env_key)
    if p and os.path.exists(p):
        return p
    root = _project_root()
    for rel in _BUNDLED:
        cand = os.path.join(root, rel, f"{name}.exe" if os.name == "nt" else name)
        if os.path.exists(cand):
            return cand
    found = shutil.which(name)
    return found


def ffmpeg() -> str:
    if "ffmpeg" not in _CACHE:
        p = _find("ffmpeg")
        if not p:
            raise RuntimeError(
                "找不到 ffmpeg。三种解法任选：\n"
                "  1. 把静态包解压到 tools/ 下（见 docs/setup.md）\n"
                "  2. 设置环境变量 FFMPEG_BIN 指向 ffmpeg 可执行文件\n"
                "  3. 把 ffmpeg 加进 PATH"
            )
        _CACHE["ffmpeg"] = p
    return _CACHE["ffmpeg"]


def ffprobe() -> Optional[str]:
    if "ffprobe" not in _CACHE:
        _CACHE["ffprobe"] = _find("ffprobe") or ""
    return _CACHE["ffprobe"] or None


def run(args: Sequence[str], cwd: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [ffmpeg(), "-hide_banner", "-nostdin", *args]
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "ignore")[-1200:]
        raise RuntimeError(f"ffmpeg 执行失败 ({proc.returncode}):\n{msg}")
    return proc


def probe(path: str) -> Dict:
    """返回 {duration, width, height, fps, vcodec, acodec, streams}"""
    pb = ffprobe()
    if not pb:
        # 没有 ffprobe 时用 ffmpeg -i 的 stderr 兜底解析时长
        proc = subprocess.run([ffmpeg(), "-i", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {"duration": _parse_duration(proc.stderr.decode("utf-8", "ignore")), "raw": True}

    proc = subprocess.run(
        [pb, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe 读取失败: {proc.stderr.decode('utf-8', 'ignore')[:400]}")
    info = json.loads(proc.stdout.decode("utf-8"))
    out: Dict = {
        "duration": float(info.get("format", {}).get("duration", 0.0)),
        "streams": [],
    }
    for s in info.get("streams", []):
        item = {
            "index": s.get("index"),
            "type": s.get("codec_type"),
            "codec": s.get("codec_name"),
        }
        if s.get("codec_type") == "video":
            item["width"] = s.get("width")
            item["height"] = s.get("height")
            out["width"] = s.get("width")
            out["height"] = s.get("height")
            out["fps"] = _parse_fps(s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1")
            out["vcodec"] = s.get("codec_name")
        elif s.get("codec_type") == "audio":
            item["channels"] = s.get("channels")
            item["sample_rate"] = s.get("sample_rate")
            out.setdefault("acodec", s.get("codec_name"))
        out["streams"].append(item)
    out["audio_streams"] = [s for s in out["streams"] if s["type"] == "audio"]
    return out


def _parse_fps(expr: str) -> float:
    try:
        num, den = expr.split("/")
        return round(float(num) / float(den or 1), 3)
    except Exception:
        return 0.0


def _parse_duration(text: str) -> float:
    import re
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", text)
    if not m:
        return 0.0
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def decode_audio(path: str, sr: int = 16000) -> "object":
    """解码成 16k 单声道 float32 numpy 数组"""
    import numpy as np

    proc = run([
        "-v", "error", "-i", path,
        "-f", "f32le", "-ac", "1", "-ar", str(sr),
        "-",
    ])
    return np.frombuffer(proc.stdout, dtype=np.float32)


def extract_audio(path: str, out: str, sr: int = 16000, track: str = "mic", index: int = 0) -> str:
    """抽音轨。track=mic 时优先第一条音频流（录课通常麦克风是 0 号）"""
    info = probe(path)
    streams = info.get("audio_streams") or []
    if not streams:
        raise RuntimeError(f"文件里没有音轨: {path}")
    if track == "index":
        idx = streams[min(index, len(streams) - 1)]["index"]
        args = ["-i", path, "-map", f"0:{idx}"]
    elif track == "mix":
        # 全部音轨混音（麦克风 + 系统声）
        args = ["-i", path]
        for s in streams:
            args += ["-map", f"0:{s['index']}"]
        n = len(streams)
        args += ["-filter_complex", f"amix=inputs={n}:duration=longest:normalize=0"]
    else:
        idx = streams[0]["index"]
        args = ["-i", path, "-map", f"0:{idx}"]

    run([*args, "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", "-y", out])
    return out


def burn_subtitles(
    video: str,
    ass: str,
    out: str,
    vcodec: str = "libx264",
    preset: str = "veryfast",
    crf: int = 20,
    acodec: str = "aac",
    abr: str = "192k",
    workdir: Optional[str] = None,
) -> str:
    """把 ASS 硬烧进画面。ass 路径相对 workdir 给，避免 Windows 盘符冒号被滤镜误解析"""
    ass_name = os.path.basename(ass)
    if workdir:
        shutil.copy2(ass, os.path.join(workdir, ass_name))
    run([
        "-i", os.path.abspath(video),
        "-vf", f"ass={ass_name}",
        "-c:v", vcodec, "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", acodec, "-b:a", abr,
        "-y", os.path.abspath(out),
    ], cwd=workdir)
    return out


def filter_available(name: str) -> bool:
    proc = subprocess.run([ffmpeg(), "-hide_banner", "-filters"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return f" {name} " in proc.stdout.decode("utf-8", "ignore")


def encoder_available(name: str) -> bool:
    proc = subprocess.run([ffmpeg(), "-hide_banner", "-encoders"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return name in proc.stdout.decode("utf-8", "ignore")
