# LectureForge

**Turn raw class recordings into publishable lectures — automatically transcribed, semantically cleaned, and bilingually subtitled.**

[中文说明](README.zh-CN.md) · [Setup guide](docs/setup.md) · [Design notes](docs/design.md) · [Prior art survey](docs/landscape.md)

---

## The problem

Every existing auto-editor treats silence as dead air: *loudness drops below a threshold for N seconds → cut it*. That rule is correct for talking-head videos and catastrophically wrong for teaching.

A lecturer who says *"Now think about this for a second…"* and then goes quiet for four seconds is **not hesitating** — they are waiting for the room. Cut that silence and the lesson becomes unintelligible. I measured this: `auto-editor` with `subtitle:pattern=想一想` protection produced **byte-identical output** to pure loudness thresholding, because subtitles only span the spoken line and never the silence *after* it.

LectureForge treats that distinction as the core problem, not an afterthought. Silence is classified by seven weighted signals (filler sounds, duration relative to the film's own P95, speech-rate drop, sentence-final particles, restarts, intent words, on-screen activity) and then either **cut, compressed, or protected**.

The design principle is **compress, don't delete**: a 2.4 s hesitation becomes 0.35 s. Zeroing it out produces machine-gun pacing that is exhausting to listen to.

## Status

Honest version: this is **v0.1**, and the subtitle half is finished while the editing half is not.

| Stage | State | Notes |
|---|---|---|
| Audio extraction & denoise chain | ⚪ designed | params in `config.example.yaml`, not wired yet |
| Speech-to-text (word-level timestamps) | ✅ **working** | 97.9 % char accuracy on a 13-term benchmark |
| Stutter vs. intentional pause classifier | ⚪ designed | model specified, no code yet |
| Non-linear editing / EDL + render | ⚪ designed | — |
| Bilingual subtitles (zh → en) | ✅ **working** | Chinese side ship-ready; translation needs an API key |
| Subtitle burn-in / render | ✅ **working** | libass, resolution-aware sizing |

What you can do today: **turn a recording into a finished video with bilingual subtitles in one command.** The automatic cutting is coming in v0.2.

## Measured results

Benchmark: 72.5 s synthetic lecture audio containing 13 domain terms, numbers, and homophone traps, scored against ground truth.

| Stage | Char accuracy | What fixed it |
|---|---|---|
| raw whisper `medium` / int8 | 83.2 % | — |
| + traditional→simplified conversion | 94.0 % | whisper switches to Traditional Chinese mid-sentence, which also breaks every glossary lookup |
| + term / homophone / segmentation fixes | **97.9 %** | targeted fixups + punctuation-backtrack segmentation |

Transcription speed: **1.36× realtime** on an i5-12450H (8 threads, int8, CPU only). A 45-minute lecture takes ~33 minutes.

Manual review load dropped from 37 % → **16 %** of cues, and the report tells you *which character* looks wrong instead of flagging whole lines.

The three real bottlenecks are engineering, not model capability: mixed Traditional/Simplified output, homophones (拟合 → 你和), and segmentation slicing words and decimals in half. All three are handled in `subs.py`.

## Quickstart

```bash
# 1. ffmpeg with libass -> tools/  (see docs/setup.md)
# 2. env
python -m venv .venv && .venv\Scripts\activate
pip install -e .

# 3. self-check
lectureforge doctor

# 4. subtitles only
lectureforge subtitle lecture.mp4 --no-translate

# 5. subtitles + burn-in
lectureforge subtitle lecture.mp4 --burn

# 6. with English, using an OpenAI-compatible endpoint
set LLM_API_KEY=sk-...
lectureforge subtitle lecture.mp4 --burn
```

Outputs land in `output/`:

```
lecture.ass              bilingual, for burn-in
lecture.zh.srt           Chinese only
lecture.bilingual.srt    both languages
lecture.review.md        the only lines a human needs to read
lecture.subtitled.mp4    finished video
```

## Glossary is not optional

Accuracy drops from 97.9 % to ~90 % without a term table, and English output degrades badly — "gradient boosting tree" becomes "gradient raised tree" and you have to fix every occurrence.

`terms.json` is a one-time investment that pays off on every subsequent lecture:

```json
{
  "zh2en":  { "梯度提升树": "Gradient Boosting Decision Tree" },
  "fixups": { "你和上一": "拟合上一" },
  "review": ["残差"]
}
```

Keep it under ~200 terms — a longer `initial_prompt` dilutes rather than helps. See [`examples/terms.example.json`](examples/terms.example.json).

> Fixups must be **phrase-level**. Replacing the single character 数 with 树 corrupts every 数字 in the transcript.

## How it works

```
recording ──► ffmpeg ──► 16 kHz float32 ──► faster-whisper ──► words + confidence
                                                                  │
                                        ┌─────────────────────────┘
                                        ▼
                        segmentation (punctuation backtrack, no word slicing)
                                        ▼
                        traditional→simplified → term fixups → number normalisation
                                        ▼
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
            review.md (16 % of cues)              ASS / SRT (bilingual)
                                                            ▼
                                                    libass burn-in
```

Design decisions worth knowing before you change anything:

- **Audio is decoded by ffmpeg, not PyAV.** `faster-whisper` 1.2.x calls `av.open(metadata_errors=...)`, removed in av 19. Decoding ourselves also gives the raw waveform needed for energy analysis.
- **Simplification happens after segmentation**, so punctuation-based boundary rules see the original text.
- **Numbers are merged before segmentation** — whisper splits `0.01` into `0.` + `01`, and no downstream fix can undo a wrong cue boundary.
- **Confidence threshold is 0.45, not 0.55.** Median word confidence is 0.993; at 0.55 the review list inflates to 37 % with cues that are actually correct.

## Roadmap

- [x] Word-level transcription with traditional→simplified normalisation
- [x] Segmentation, term correction, bilingual ASS/SRT, review report
- [x] Burn-in render
- [ ] Denoise chain (`highpass` → `afftdn` → `loudnorm`)
- [ ] Seven-signal hesitation classifier + silence-boundary energy refinement
- [ ] EDL generation and `trim`+`concat` single-pass render
- [ ] Subtitle retiming against the EDL (skip this and every cue drifts)
- [ ] Single `run.py` entry point, resumable

## Docs

| File | Contents |
|---|---|
| [docs/setup.md](docs/setup.md) | Installation, and the three Windows/HuggingFace traps worth avoiding |
| [docs/design.md](docs/design.md) | Full technical design, including the classifier's signal weights (Chinese) |
| [docs/landscape.md](docs/landscape.md) | Survey of existing tools with measured evidence (Chinese) |

## License

MIT. ffmpeg is GPL and is **not** bundled — download it yourself.
