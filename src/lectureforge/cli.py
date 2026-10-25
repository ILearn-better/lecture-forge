# -*- coding: utf-8 -*-
"""命令行入口。

    lectureforge doctor              # 环境自检
    lectureforge transcribe a.mp4    # 只转写，产出词级时间戳
    lectureforge subtitle a.mp4      # 转写 + 断句 + 术语修正 + 双语字幕
    lectureforge subtitle a.mp4 --burn   # 再加一步：烧进画面出成片
    lectureforge render a.mp4 -a x.ass   # 已有字幕，只做烧录
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

from . import __version__
from . import asr as asr_mod
from . import ffmpeg as ff
from . import render as render_mod
from .config import Config
from .subs import (
    Cue, TermTable, dump_review, postprocess, risk_report, split_cues,
)


# ------------------------------------------------------------
# 小工具
# ------------------------------------------------------------

def _say(msg: str) -> None:
    print(msg, flush=True)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _fmt_t(sec: float) -> str:
    return f"{int(sec // 60)}m{sec % 60:04.1f}s"


# ------------------------------------------------------------
# doctor
# ------------------------------------------------------------

def cmd_doctor(args) -> int:
    ok = True
    _say("LectureForge 环境自检")
    _say("=" * 56)

    try:
        f = ff.ffmpeg()
        _say(f"[ OK ] ffmpeg      {f}")
    except RuntimeError as e:
        _say(f"[FAIL] ffmpeg      {e}")
        return 1
    pb = ff.ffprobe()
    _say(f"{'[ OK ]' if pb else '[WARN]'} ffprobe     {pb or '缺失，时长探测会退化'}")

    for name in ("afftdn", "loudnorm", "ass", "highpass"):
        has = ff.filter_available(name)
        ok &= has or name in ("afftdn", "highpass")
        _say(f"{'[ OK ]' if has else '[WARN]'} 滤镜 {name:<10} {'可用' if has else '缺失'}")
    for enc in ("libx264", "aac"):
        has = ff.encoder_available(enc)
        ok &= has
        _say(f"{'[ OK ]' if has else '[FAIL]'} 编码器 {enc:<8} {'可用' if has else '缺失'}")

    _say("-" * 56)
    for mod, hint in (
        ("numpy", "numpy"),
        ("faster_whisper", "faster-whisper"),
        ("opencc", "opencc-python-reimplemented"),
        ("yaml", "pyyaml"),
    ):
        try:
            __import__(mod)
            _say(f"[ OK ] 依赖 {mod}")
        except ImportError:
            ok = False
            _say(f"[FAIL] 依赖 {mod} 缺失 → pip install {hint}")

    model = args.model or Config.load(args.config)("asr.model", "medium")
    ref = asr_mod.resolve_model(model, os.path.join(os.getcwd(), "models"))
    if os.path.isdir(ref):
        size = sum(os.path.getsize(os.path.join(ref, x)) for x in os.listdir(ref)
                   if os.path.isfile(os.path.join(ref, x)))
        _say(f"[ OK ] 模型 {model:<8} 本地 {ref} ({size / 1e6:.0f} MB)")
    else:
        _say(f"[WARN] 模型 {model:<8} 未本地化，首次运行会下载（约 1.5 GB，见 docs/setup.md）")

    _say("=" * 56)
    _say("结论：" + ("可以开工" if ok else "有硬性缺失，先按上面 FAIL 项补齐"))
    return 0 if ok else 1


# ------------------------------------------------------------
# transcribe
# ------------------------------------------------------------

def cmd_transcribe(args) -> int:
    cfg = Config.load(args.config)
    outdir = _ensure_dir(args.outdir or cfg("paths.work", "work"))
    stem = _stem(args.input)
    wav = os.path.join(outdir, f"{stem}.16k.wav")
    words_json = os.path.join(outdir, f"{stem}.words.json")

    t0 = time.time()
    _say(f"[1/2] 抽音轨 → {wav}")
    ff.extract_audio(args.input, wav, track=cfg("audio.track", "mic"),
                     index=cfg("audio.track_index", 0))
    dur = ff.probe(args.input).get("duration") or 0

    _say(f"[2/2] 转写（{args.model or cfg('asr.model')} / "
         f"{cfg('asr.compute_type')} / {cfg('asr.cpu_threads')} 线程）…")
    words = asr_mod.transcribe(
        wav,
        model=args.model or cfg("asr.model", "medium"),
        language=args.language or cfg("asr.language", "zh"),
        compute_type=cfg("asr.compute_type", "int8"),
        cpu_threads=cfg("asr.cpu_threads", 8),
        beam_size=cfg("asr.beam_size", 2),
        vad_filter=cfg("asr.vad_filter", True),
        models_dir=os.path.join(os.getcwd(), "models"),
        out_json=words_json,
    )
    el = time.time() - t0
    speed = dur / el if el else 0
    _say(f"完成：{len(words)} 个词 → {words_json}")
    _say(f"耗时 {_fmt_t(el)}，速度 {speed:.2f}x 实时")
    return 0


# ------------------------------------------------------------
# subtitle
# ------------------------------------------------------------

def cmd_subtitle(args) -> int:
    cfg = Config.load(args.config)
    outdir = _ensure_dir(args.outdir or cfg("paths.output", "output"))
    workdir = _ensure_dir(cfg("paths.work", "work"))
    stem = _stem(args.input)

    terms_path = args.terms or cfg("paths.terms", "terms.json")
    term = TermTable.load(terms_path if os.path.exists(terms_path) else None)
    if term.zh2en:
        _say(f"术语表：{len(term.zh2en)} 条（{terms_path}）")
    else:
        _say("术语表：未提供 —— 准确率和英文质量都会下降，建议先建一份 terms.json")
    if term.fixups:
        term.fixups.update({k: v for k, v in term.fixups.items()})

    # 1) 词级时间戳：能复用就复用，转写是整条链路最慢的一步
    words_json = os.path.join(workdir, f"{stem}.words.json")
    if args.reuse_words and os.path.exists(args.reuse_words):
        _say(f"[1/4] 复用已有转写 {args.reuse_words}")
        words = asr_mod.load_words(args.reuse_words)
    else:
        if os.path.exists(words_json) and not args.force:
            _say(f"[1/4] 复用缓存 {words_json}（--force 可强制重转）")
            words = asr_mod.load_words(words_json)
        else:
            wav = os.path.join(workdir, f"{stem}.16k.wav")
            if not os.path.exists(wav):
                _say(f"[1/4] 抽音轨 → {wav}")
                ff.extract_audio(args.input, wav, track=cfg("audio.track", "mic"),
                                 index=cfg("audio.track_index", 0))
            _say(f"[1/4] 转写中（CPU 上大约是 1.3~1.5x 实时）…")
            t0 = time.time()
            words = asr_mod.transcribe(
                wav,
                model=args.model or cfg("asr.model", "medium"),
                language=args.language or cfg("asr.language", "zh"),
                compute_type=cfg("asr.compute_type", "int8"),
                cpu_threads=cfg("asr.cpu_threads", 8),
                beam_size=cfg("asr.beam_size", 2),
                vad_filter=cfg("asr.vad_filter", True),
                initial_prompt=term.initial_prompt(),
                models_dir=os.path.join(os.getcwd(), "models"),
                out_json=words_json,
            )
            _say(f"      {len(words)} 词，耗时 {_fmt_t(time.time() - t0)}")

    # 2) 断句 + 修正
    _say("[2/4] 断句 / 简繁统一 / 术语修正")
    cues = split_cues(words, max_len=cfg("subtitle.max_len", 18),
                      max_dur=cfg("subtitle.max_dur", 6.5),
                      min_dur=cfg("subtitle.min_dur", 1.0))
    postprocess(cues, term)
    fixed = sum(1 for c in cues if c.fixes)
    _say(f"      {len(cues)} 条字幕，自动修正 {fixed} 条")

    # 3) 翻译
    if args.no_translate or not cfg("subtitle.translate.enabled", False):
        _say("[3/4] 跳过英文翻译")
    else:
        api_key = os.environ.get("LLM_API_KEY") or cfg("subtitle.translate.api_key", "")
        if not api_key:
            _say("[3/4] 跳过翻译：没有 API key（设环境变量 LLM_API_KEY 或填 config）")
        else:
            from .translate import Translator
            _say(f"[3/4] 翻译中（{cfg('subtitle.translate.model')}）…")
            tr = Translator(
                api_base=cfg("subtitle.translate.api_base"),
                api_key=api_key,
                model=cfg("subtitle.translate.model", "deepseek-chat"),
                max_chars=cfg("subtitle.translate.max_chars", 78),
                cache_path=os.path.join(workdir, "translate_cache.json"),
            )
            tr.translate_cues(cues, term, batch=cfg("subtitle.translate.batch", 10))

    # 4) 导出
    info = ff.probe(args.input)
    w, h = info.get("width") or 1920, info.get("height") or 1080
    style = cfg("subtitle.style")
    _say(f"[4/4] 导出字幕（{w}x{h}）")
    paths = render_mod.export_subs(
        cues, outdir, stem=stem, width=w, height=h,
        font_zh=style.get("font_zh", "Microsoft YaHei"),
        font_en=style.get("font_en", "Segoe UI"),
        size_zh=style.get("size_zh", 56),
        size_en=style.get("size_en", 38),
    )
    for k, v in paths.items():
        _say(f"      {k:<14} {v}")

    risky = risk_report(cues)
    review = os.path.join(outdir, f"{stem}.review.md")
    dump_review(cues, risky, review)
    rate = 100.0 * len(risky) / max(len(cues), 1)
    _say(f"      需人工过目 {len(risky)}/{len(cues)} 条（{rate:.0f}%）→ {review}")

    if args.burn:
        out_mp4 = os.path.join(outdir, f"{stem}.subtitled.mp4")
        _say(f"[+] 烧录字幕（{cfg('render.preset')} / crf {cfg('render.crf')}）…")
        t0 = time.time()
        render_mod.burn(
            args.input, paths["ass"], out_mp4,
            vcodec=cfg("render.vcodec", "libx264"),
            preset=cfg("render.preset", "veryfast"),
            crf=cfg("render.crf", 20),
            acodec=cfg("render.acodec", "aac"),
            abr=cfg("render.abr", "192k"),
        )
        _say(f"      成片 {out_mp4}（{os.path.getsize(out_mp4) / 1e6:.0f} MB，"
             f"耗时 {_fmt_t(time.time() - t0)}）")
    return 0


# ------------------------------------------------------------
# render
# ------------------------------------------------------------

def cmd_render(args) -> int:
    cfg = Config.load(args.config)
    out = args.output or os.path.join(args.outdir or "output",
                                      f"{_stem(args.input)}.subtitled.mp4")
    _ensure_dir(os.path.dirname(os.path.abspath(out)))
    _say(f"烧录 {args.ass} → {out}")
    render_mod.burn(
        args.input, args.ass, out,
        vcodec=cfg("render.vcodec", "libx264"),
        preset=cfg("render.preset", "veryfast"),
        crf=cfg("render.crf", 20),
        acodec=cfg("render.acodec", "aac"),
        abr=cfg("render.abr", "192k"),
    )
    _say("完成")
    return 0


# ------------------------------------------------------------
# parser
# ------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lectureforge",
        description="LectureForge — 录课后期自动化：转写、卡壳判定、双语字幕、成片渲染",
    )
    p.add_argument("-v", "--version", action="version", version=f"lectureforge {__version__}")
    p.add_argument("-c", "--config", help="配置文件路径（默认 ./config.yaml）")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="检查 ffmpeg / 依赖 / 模型是否就绪")
    d.add_argument("--model", help="要检查的模型名")
    d.set_defaults(func=cmd_doctor)

    t = sub.add_parser("transcribe", help="只做转写，产出词级时间戳 json")
    t.add_argument("input")
    t.add_argument("-o", "--outdir")
    t.add_argument("--model")
    t.add_argument("--language", default=None)
    t.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("subtitle", help="转写 + 断句 + 修正 + 双语字幕（可选烧录）")
    s.add_argument("input")
    s.add_argument("-o", "--outdir")
    s.add_argument("--terms", help="术语表 json（默认取配置 paths.terms）")
    s.add_argument("--model")
    s.add_argument("--language", default=None)
    s.add_argument("--reuse-words", help="复用已有的 words.json，跳过转写")
    s.add_argument("--no-translate", action="store_true", help="不翻译，只出中文字幕")
    s.add_argument("--burn", action="store_true", help="直接烧录出成片")
    s.add_argument("--force", action="store_true", help="强制重新转写")
    s.set_defaults(func=cmd_subtitle)

    r = sub.add_parser("render", help="已有 ASS，只做烧录")
    r.add_argument("input")
    r.add_argument("-a", "--ass", required=True)
    r.add_argument("-o", "--output")
    r.add_argument("--outdir")
    r.set_defaults(func=cmd_render)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        _say("\n已中断")
        return 130
    except Exception as e:
        _say(f"\n[错误] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
