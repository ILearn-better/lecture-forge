# -*- coding: utf-8 -*-
"""渲染：把 ASS 字幕烧进画面。

字幕一律用 ass 滤镜硬烧，不走外挂字幕流——
外挂字幕在多数国产播放器/平台上默认不显示，学员不会去手动开。
"""
from __future__ import annotations

import os
import shutil
from typing import Optional, Sequence

from . import ffmpeg as ff
from .subs import Cue, write_bilingual_ass, write_srt


def burn(
    video: str,
    ass: str,
    out: str,
    vcodec: str = "libx264",
    preset: str = "veryfast",
    crf: int = 20,
    acodec: str = "aac",
    abr: str = "192k",
) -> str:
    workdir = os.path.dirname(os.path.abspath(ass))
    return ff.burn_subtitles(
        video, ass, out, vcodec=vcodec, preset=preset,
        crf=crf, acodec=acodec, abr=abr, workdir=workdir,
    )


def export_subs(
    cues: Sequence[Cue],
    outdir: str,
    stem: str = "lecture",
    width: int = 1920,
    height: int = 1080,
    font_zh: str = "Microsoft YaHei",
    font_en: str = "Segoe UI",
    size_zh: int = 56,
    size_en: int = 38,
) -> dict:
    """同时输出 ASS（双语烧录用）和 SRT（投平台/二次编辑用）"""
    os.makedirs(outdir, exist_ok=True)
    ass = os.path.join(outdir, f"{stem}.ass")
    srt_zh = os.path.join(outdir, f"{stem}.zh.srt")
    srt_bi = os.path.join(outdir, f"{stem}.bilingual.srt")
    write_bilingual_ass(cues, ass, width=width, height=height,
                        font_zh=font_zh, font_en=font_en,
                        sz_zh=size_zh, sz_en=size_en)
    write_srt(cues, srt_zh, lang="zh")
    write_srt(cues, srt_bi, bilingual=True)
    return {"ass": ass, "srt_zh": srt_zh, "srt_bilingual": srt_bi}
