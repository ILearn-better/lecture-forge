# -*- coding: utf-8 -*-
"""配置读取：config.yaml → 带默认值的字典包装。

刻意不引入 pydantic —— 这是一条单人维护的工具链，
配置项少、改动频繁，用 dataclass 反而每次加参数都要改两处。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "paths": {
        "work": "work",
        "output": "output",
        "assets": "assets",
        "terms": "terms.json",
    },
    "audio": {
        "track": "mic",
        "track_index": 0,
        "denoise": {
            "enabled": True,
            "highpass_hz": 80,
            "afftdn_nr": 12,
            "afftdn_nf": -32,
            "rnnoise": False,
        },
        "loudnorm": {"enabled": True, "i": -16, "tp": -1.5, "lra": 11},
    },
    "asr": {
        "model": "medium",
        "compute_type": "int8",
        "language": "zh",
        "beam_size": 2,
        "cpu_threads": 8,
        "vad_filter": True,
    },
    "subtitle": {
        "max_len": 18,
        "max_dur": 6.5,
        "min_dur": 1.0,
        "style": {
            "font_zh": "Microsoft YaHei",
            "font_en": "Segoe UI",
            "size_zh": 56,
            "size_en": 38,
            "margin_v": 60,
        },
        "translate": {
            "enabled": False,
            "api_base": "https://api.deepseek.com/v1",
            "api_key": "",
            "model": "deepseek-chat",
            "max_chars": 78,
            "batch": 10,
        },
    },
    "render": {
        "vcodec": "libx264",
        "preset": "veryfast",
        "crf": 20,
        "acodec": "aac",
        "abr": "192k",
        "burn_subs": True,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """点号取值 + 默认值兜底。`cfg("asr.model")` → 'medium'"""

    def __init__(self, data: Optional[Dict[str, Any]] = None, path: Optional[str] = None):
        self.path = path
        self._d = _deep_merge(DEFAULTS, data or {})

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        if not path:
            for cand in ("config.yaml", "config.yml"):
                if os.path.exists(cand):
                    path = cand
                    break
        if not path or not os.path.exists(path):
            return cls()
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError("读取 config.yaml 需要 pyyaml：pip install pyyaml") from e
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {}, path=path)

    def __call__(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self._d
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur if cur is not None else default

    def section(self, name: str) -> Dict[str, Any]:
        v = self._d.get(name, {})
        return v if isinstance(v, dict) else {}

    def as_dict(self) -> Dict[str, Any]:
        return self._d
