# -*- coding: utf-8 -*-
"""语音转写：faster-whisper + 词级时间戳"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from . import ffmpeg as ff
from .subs import Word


def resolve_model(model: str, models_dir: Optional[str] = None) -> str:
    """
    模型定位优先级：显式本地目录 → 项目 models/ 下的同名目录 → 交给 faster-whisper 自己下载。

    为什么要支持本地目录：Windows 上 huggingface_hub 建符号链接需要管理员权限，
    快照目录经常建不起来。预先手工放好目录最稳（见 docs/setup.md）。
    """
    if os.path.isdir(model):
        return model
    if models_dir and os.path.isdir(models_dir):
        for name in (f"faster-whisper-{model}", f"whisper-{model}", model):
            cand = os.path.join(models_dir, name)
            if os.path.isdir(cand):
                return cand
    return model


def transcribe(
    audio_path: str,
    model: str = "medium",
    language: str = "zh",
    compute_type: str = "int8",
    cpu_threads: int = 8,
    beam_size: int = 2,
    initial_prompt: str = "",
    vad_filter: bool = True,
    models_dir: Optional[str] = None,
    out_json: Optional[str] = None,
) -> List[Word]:
    # 国内直连 HuggingFace 很慢，默认走镜像
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # 必须禁用 xet：新版 huggingface_hub 默认走 cas-server.xethub.hf.co，
    # 镜像站不支持会返回 401，导致模型下载卡死
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from faster_whisper import WhisperModel

    model_ref = resolve_model(model, models_dir)
    m = WhisperModel(model_ref, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)

    audio = ff.decode_audio(audio_path, sr=16000)

    segments, info = m.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=vad_filter,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=120),
        initial_prompt=initial_prompt or None,
        condition_on_previous_text=False,   # 长音频关掉，防上下文跑偏
    )

    words: List[Word] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            words.append(Word(start=w.start, end=w.end, text=w.word,
                              prob=getattr(w, "probability", 1.0)))

    if out_json:
        dump_words(words, out_json, language=getattr(info, "language", language),
                   duration=getattr(info, "duration", 0.0))
    return words


def dump_words(words: List[Word], path: str, language: str = "zh", duration: float = 0.0) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "language": language,
                "duration": duration,
                "words": [{"s": w.start, "e": w.end, "t": w.text, "p": w.prob} for w in words],
            },
            f, ensure_ascii=False, indent=1,
        )


def load_words(path: str) -> List[Word]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return [Word(start=w["s"], end=w["e"], text=w["t"], prob=w.get("p", 1.0)) for w in d["words"]]
