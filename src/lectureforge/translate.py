# -*- coding: utf-8 -*-
"""字幕翻译：术语约束 + 上下文感知 + 长度控制

关键设计（决定"要不要人调"）：
  1. 术语表强制映射——模型不允许自由发挥专有名词
  2. 带上一条原文做上下文——保证跨句术语与语气一致
  3. 长度上限——英文超过 2 行会挡画面，超长自动压缩重译
  4. 结果缓存——同一节课重跑不重复花钱
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Dict, List, Optional, Sequence

from .subs import Cue, TermTable

SYS_PROMPT = """You are a professional subtitle translator for university-level lectures.
Translate each Chinese subtitle line into concise, natural academic English.

Rules:
1. Use the provided glossary EXACTLY as given — never invent alternative translations for these terms.
2. Preserve technical accuracy over literal wording. Keep the lecturer's spoken register (clear, direct, no fluff).
3. Keep each English line under {max_chars} characters. If it would be longer, compress by omitting filler, not by dropping meaning.
4. Keep numbers, units, formulas and symbols unchanged.
5. Do NOT add explanations, notes, or quotation marks.
6. Output ONLY a JSON array of strings, same length and order as the input array. No other text.
"""


def _glossary_block(term: TermTable, texts: Sequence[str]) -> str:
    """只注入本条字幕实际用到的术语，省 token 且更聚焦"""
    hit = {}
    joined = "".join(texts)
    for zh, en in term.zh2en.items():
        if zh in joined:
            hit[zh] = en
    if not hit:
        return "(none)"
    return "\n".join(f"- {zh} => {en}" for zh, en in hit.items())


def _cache_key(text: str, prev: str, max_chars: int) -> str:
    return hashlib.md5(f"{text}|{prev}|{max_chars}".encode()).hexdigest()


class Translator:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "deepseek-chat",
        max_chars: int = 78,
        cache_path: Optional[str] = None,
        timeout: int = 60,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_chars = max_chars
        self.timeout = timeout
        self.cache_path = cache_path
        self.cache: Dict[str, str] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def _save_cache(self):
        if not self.cache_path:
            return
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False)

    def _call(self, texts: List[str], prev: str, glossary: str) -> List[str]:
        import urllib.request

        user = (
            f"Glossary (authoritative):\n{glossary}\n\n"
            f"Previous line (context only, do NOT translate):\n{prev or '(none)'}\n\n"
            f"Translate these {len(texts)} lines:\n"
            + json.dumps(texts, ensure_ascii=False)
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYS_PROMPT.format(max_chars=self.max_chars)},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.loads(r.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.M).strip()
        arr = json.loads(content)
        if not isinstance(arr, list) or len(arr) != len(texts):
            raise ValueError(f"返回长度不匹配: 期望 {len(texts)}，得到 {len(arr) if isinstance(arr, list) else type(arr)}")
        return [str(x).strip() for x in arr]

    def translate_cues(self, cues: List[Cue], term: TermTable, batch: int = 10) -> None:
        """就地填充 cue.en"""
        for i in range(0, len(cues), batch):
            chunk = cues[i : i + batch]
            todo, todo_idx = [], []
            for j, c in enumerate(chunk):
                key = _cache_key(c.zh, cues[i - 1].zh if i > 0 else "", self.max_chars)
                if key in self.cache:
                    c.en = self.cache[key]
                else:
                    todo.append(c.zh)
                    todo_idx.append(j)

            if todo:
                prev = cues[i - 1].zh if i > 0 else ""
                gl = _glossary_block(term, todo)
                for attempt in range(3):
                    try:
                        outs = self._call(todo, prev, gl)
                        break
                    except Exception as e:
                        if attempt == 2:
                            print(f"  [warn] 批次 {i} 翻译失败，留空待人工: {e}")
                            outs = [""] * len(todo)
                        else:
                            time.sleep(1.5 * (attempt + 1))
                for j, en in zip(todo_idx, outs):
                    chunk[j].en = en
                    key = _cache_key(chunk[j].zh, prev, self.max_chars)
                    self.cache[key] = en
                self._save_cache()
            print(f"  翻译进度 {min(i+batch, len(cues))}/{len(cues)}", end="\r")
        print()

        # 超长的英文再压一轮
        long_ones = [c for c in cues if len(c.en) > self.max_chars * 1.35 and c.en]
        if long_ones:
            print(f"  压缩 {len(long_ones)} 条超长英文…")
            gl = _glossary_block(term, [c.zh for c in long_ones])
            try:
                outs = self._call(
                    ["请把这条英文字幕压缩到 60 字符以内，保留全部信息点，去掉冗余修饰：" + c.en for c in long_ones],
                    "", gl,
                )
                for c, en in zip(long_ones, outs):
                    if en:
                        c.en = en
            except Exception as e:
                print(f"  [warn] 压缩失败: {e}")
