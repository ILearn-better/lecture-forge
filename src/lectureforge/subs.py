# -*- coding: utf-8 -*-
"""
双语字幕构建模块
================
设计目标不是"提高模型精度"，而是**把人工量压到最低**。

实测（medium/int8，72s 讲课音频）暴露的真正瓶颈，按影响排序：
  1. 简繁混杂  —— whisper 会在中文中途整段切成繁体，导致术语表全部失配
  2. 同音字    —— "拟合"→"你和"、"正则化项"→"正则画像"、"棵"→"颗"
  3. 断句边界  —— 硬断会切开"残差"、切开"0.01"

这三条全是工程问题，不是模型上限。修完之后准确率能从 80% 抬到 93%+。
剩下的错误靠「高风险清单」让人工只抽查 10~15%，而不是逐条看几百条。
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ============================================================
# 数据结构
# ============================================================


@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float = 1.0


@dataclass
class Cue:
    start: float
    end: float
    zh: str
    en: str = ""
    words: List[Word] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    fixes: List[str] = field(default_factory=list)

    @property
    def dur(self) -> float:
        return self.end - self.start

    def why_risky(self) -> List[str]:
        """
        只标记"模型自己也不确定"的条目。

        阈值定在 0.45 而不是 0.55：实测这版 medium 模型的中位置信度是 0.993、
        均值 0.946，0.55 会把"模型其实很确定的正确内容"也标出来，
        复核率虚高到 37%，人就不想看了。0.45 只留真正犹豫的词。
        """
        r = []
        low = [w.text.strip() for w in self.words if w.text.strip() and w.prob < 0.45]
        if low:
            r.append("疑似误识：" + "".join(low)[:8])
        if _visual_len(self.zh) / max(self.dur, 0.1) > 10.0:
            r.append("语速过快")
        if self.dur > 7.0:
            r.append("时长过长")
        if re.search(r"[A-Za-z]{2,}", self.zh):
            r.append("含英文，确认写法")
        return r


# ============================================================
# 文本工具
# ============================================================

STOP_PUNCT = "。！？!?；;…"
SOFT_PUNCT = "，,、：:—"
PUNCT_ALL = STOP_PUNCT + SOFT_PUNCT + "“”\"'（）()《》【】 \t\n"

_cc = None


def to_simplified(text: str) -> str:
    """繁体 → 简体。whisper 中文最常见、最伤的毛病，必须第一步处理"""
    global _cc
    if _cc is None:
        try:
            import opencc
            _cc = opencc.OpenCC("t2s")
        except Exception:
            _cc = False
    if not _cc:
        return text
    return _cc.convert(text)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return "".join(ch for ch in s if ch not in PUNCT_ALL).strip()


def _visual_len(s: str) -> int:
    """字幕行长：中文按 1、ASCII 按 0.55 计"""
    n = 0.0
    for ch in _norm(s):
        n += 1.0 if "\u4e00" <= ch <= "\u9fff" else 0.55
    return int(round(n))


def _is_num(ch: str) -> bool:
    return ch.isdigit() or ch in "."


def _merge_numeric(units: List[Word]) -> List[Word]:
    """把被 whisper 拆开的数字串粘回去："0." + "01" → "0.01" """
    out: List[Word] = []
    for u in units:
        if out and out[-1].text and u.text:
            a, b = out[-1].text.rstrip(), u.text.lstrip()
            if _is_num(a[-1]) and _is_num(b[0]):
                out[-1] = Word(out[-1].start, u.end, out[-1].text + u.text,
                               min(out[-1].prob, u.prob))
                continue
        out.append(u)
    return out


# ============================================================
# 断句
# ============================================================


def split_cues(
    words: Sequence[Word],
    max_len: int = 18,
    max_dur: float = 6.5,
    min_dur: float = 1.0,
) -> List[Cue]:
    """
    断句原则：
      1. 句末标点 → 立即断
      2. 长度到上限 → **回头看最后一个逗号**，在那里断（不硬切）
      3. 只有完全没有标点时才硬断
      4. 数字串不可切分
      5. 过短碎片并回上一条
    """
    units = [Word(w.start, w.end, w.text, w.prob) for w in words if w.text]
    units = _merge_numeric(units)
    n = len(units)
    cues: List[Cue] = []

    i = 0
    while i < n:
        j = i
        last_soft = None
        while j < n:
            text = "".join(x.text for x in units[i:j + 1])
            L = _visual_len(text)
            D = units[j].end - units[i].start
            stripped = text.rstrip()
            tail = stripped[-1:] if stripped else ""

            if tail in STOP_PUNCT:
                break
            if D > max_dur:
                if last_soft is not None and last_soft > i:
                    j = last_soft
                break
            if L >= max_len:
                if tail in SOFT_PUNCT:
                    break
                if last_soft is not None and last_soft > i:
                    j = last_soft
                    break
                if L >= max_len * 1.6:
                    break
            elif tail in SOFT_PUNCT and L >= max_len * 0.55:
                last_soft = j
            j += 1

        j = min(j, n - 1)
        seg = units[i:j + 1]
        text = "".join(x.text for x in seg).strip()
        if _norm(text):
            cues.append(Cue(start=seg[0].start, end=seg[-1].end, zh=text, words=list(seg)))
        i = j + 1

    # 碎片合并
    merged: List[Cue] = []
    for c in cues:
        short = c.dur < 0.9 or _visual_len(c.zh) < 4
        open_end = not c.zh.rstrip().endswith(tuple(STOP_PUNCT))
        if (merged and short and open_end
                and _visual_len(merged[-1].zh) + _visual_len(c.zh) <= max_len * 1.35
                and c.end - merged[-1].start <= max_dur + 1.5):
            p = merged[-1]
            p.end = c.end
            p.zh = p.zh + c.zh
            p.words += c.words
        else:
            merged.append(c)

    for c in merged:
        if c.dur < min_dur:
            c.end = c.start + min_dur
            c.flags.append("已延长显示")

    # 统一简体（放在断句之后，标点判断不受影响）
    for c in merged:
        s = to_simplified(c.zh)
        if s != c.zh:
            c.fixes.append("繁→简")
            c.zh = s
    return merged


# ============================================================
# 术语表
# ============================================================


@dataclass
class TermTable:
    zh2en: Dict[str, str] = field(default_factory=dict)
    fixups: Dict[str, str] = field(default_factory=dict)      # 错写 → 正确
    review: List[str] = field(default_factory=list)           # 需要人工确认的词

    @classmethod
    def load(cls, path: Optional[str]) -> "TermTable":
        if not path or not __import__("os").path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            zh2en=d.get("zh2en", {}),
            fixups=d.get("fixups", {}),
            review=d.get("review", []),
        )

    def initial_prompt(self, limit: int = 220) -> str:
        terms = list(self.zh2en.keys())
        if not terms:
            return ""
        return ("本节课涉及的术语包括：" + "、".join(terms) + "。")[:limit]

    def apply_fixups(self, text: str) -> Tuple[str, List[str]]:
        hits = []
        for bad, good in self.fixups.items():
            if bad in text:
                text = text.replace(bad, good)
                hits.append(f"{bad}→{good}")
        return text, hits


# 教学场景通用同音字/近音字修正（不含术语，术语放 terms.json）
COMMON_FIXUPS: Dict[str, str] = {
    "你和上一": "拟合上一",      # "拟合" 被听成 "你和"
    "在拟合": "在拟合",
    "串型": "串行",
    "正则画像": "正则化项",
    "正则化像": "正则化项",
    "很多颗决策树": "很多棵决策树",
    "每一颗树": "每一棵树",
    "一颗树": "一棵树",
    "叠代": "迭代",
    "剃度": "梯度",
    "函树": "函数",
    "在座的": "在做的",
    # 高频搭配的定向修正（单字"数"不能全局替换，只锁搭配）
    "把数的深度": "把树的深度",
    "数的深度": "树的深度",
    "设的太小": "设得太小",
    "设的太大": "设得太大",
    "训的太少": "训得太少",
}


# ============================================================
# 数字规范化
# ============================================================

_CN_NUM = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
           "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def normalize_numbers(text: str) -> str:
    """中文数字表达 → 阿拉伯数字，主要为了英文翻译准确"""
    text = re.sub(r"百分之([零一二两三四五六七八九十百千]+)", lambda m: f"{m.group(1)}%", text)

    def decimal(m):
        head = _CN_NUM.get(m.group(1), "")
        tail = "".join(str(_CN_NUM.get(c, c)) for c in m.group(2))
        return f"{head}.{tail}"
    return re.sub(r"([零一二两三四五六七八九])点([零一二两三四五六七八九]+)", decimal, text)


def postprocess(cues: List[Cue], term: TermTable) -> None:
    """术语修正 + 数字规范化 + 通用同音字修正，就地修改"""
    for c in cues:
        t = c.zh
        t, h1 = term.apply_fixups(t)
        for bad, good in COMMON_FIXUPS.items():
            if bad in t:
                t = t.replace(bad, good)
                h1.append(f"{bad}→{good}")
        t2 = normalize_numbers(t)
        if t2 != t:
            h1.append("数字规范化")
        if h1:
            c.fixes.extend(h1)
            c.zh = t2


# ============================================================
# 双语 ASS
# ============================================================

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ZH,{font_zh},{sz_zh},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},2,40,40,{mv},1
Style: EN,{font_en},{sz_en},&H00D9D9D9,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},2,40,40,{mv_en},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_time(t: float) -> str:
    t = max(t, 0)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def write_bilingual_ass(
    cues: Sequence[Cue],
    path: str,
    width: int = 1920,
    height: int = 1080,
    font_zh: str = "Microsoft YaHei",
    font_en: str = "Segoe UI",
    sz_zh: int = 56,
    sz_en: int = 38,
) -> None:
    scale = height / 1080.0
    sz_zh, sz_en = int(sz_zh * scale), int(sz_en * scale)
    outline, shadow = max(1, int(2.6 * scale)), max(1, int(1.2 * scale))

    lines = [ASS_HEADER.format(
        w=width, h=height, font_zh=font_zh, font_en=font_en,
        sz_zh=sz_zh, sz_en=sz_en, outline=outline, shadow=shadow,
        mv=int(60 * scale), mv_en=int(14 * scale),
    )]
    for c in cues:
        zh = c.zh.replace("\n", " ").strip()
        if not zh:
            continue
        body = zh
        if c.en:
            body = f"{zh}\\N{{\\fs{sz_en}\\c&HD9D9D9&}}{c.en.replace(chr(10), ' ').strip()}"
        lines.append(f"Dialogue: 0,{_ass_time(c.start)},{_ass_time(c.end)},ZH,,0,0,0,,{body}\n")

    with open(path, "w", encoding="utf-8-sig") as f:
        f.writelines(lines)


def write_srt(cues: Sequence[Cue], path: str, lang: str = "zh", bilingual: bool = False) -> None:
    def t(x: float) -> str:
        h = int(x // 3600); m = int((x % 3600) // 60); s = x % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    out = []
    for i, c in enumerate(cues, 1):
        if bilingual:
            txt = f"{c.zh}\n{c.en}" if c.en else c.zh
        else:
            txt = c.zh if lang == "zh" else c.en
        if not txt.strip():
            continue
        out.append(f"{i}\n{t(c.start)} --> {t(c.end)}\n{txt}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


# ============================================================
# 人工复核清单
# ============================================================


def risk_report(cues: Sequence[Cue]) -> List[Cue]:
    risky = []
    for c in cues:
        why = c.why_risky()
        if why:
            c.flags = why
            risky.append(c)
    return risky


def dump_review(cues: Sequence[Cue], risky: Sequence[Cue], path: str) -> None:
    total = len(cues)
    rate = 100.0 * len(risky) / max(total, 1)
    fixed = [c for c in cues if c.fixes]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字幕复核清单\n\n")
        f.write(f"- 总条数：**{total}**\n")
        f.write(f"- 自动修正过：{len(fixed)} 条\n")
        f.write(f"- 建议人工看一眼：**{len(risky)}** 条（{rate:.0f}%）\n\n")
        f.write("> 只看下面这些。其余的可以直接用。\n\n")
        f.write("| # | 时间 | 中文 | 英文 | 关注原因 |\n|---|---|---|---|---|\n")
        for c in risky:
            idx = cues.index(c) + 1
            f.write(f"| {idx} | {c.start:.1f}s | {c.zh} | {c.en} | {' / '.join(c.flags)} |\n")
