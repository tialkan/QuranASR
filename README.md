<div align="center">

# 🕌 QuranASR

### Fine-tuning OpenAI Whisper for Quran Recitation Recognition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tialkan/QuranASR/blob/main/notebook.ipynb)
[![Model on HuggingFace](https://img.shields.io/badge/🤗_Model-baristiran/QuranASR-yellow)](https://huggingface.co/baristiran)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete, reproducible pipeline for fine-tuning `openai/whisper-small` on Quran recitations from 12 professional reciters (~75,000 audio samples). The resulting model runs on-device via [WhisperKit](https://github.com/argmaxinc/WhisperKit) and powers [**Tilavet**](https://tilavet.app) — a Quran companion app for iOS.

**[Try the Model on 🤗 HuggingFace](https://huggingface.co/baristiran)** · **[Download Tilavet for iOS](https://tilavet.app)**

</div>

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Why We Built This](#why-we-built-this)
- [Architecture Decisions](#architecture-decisions)
- [Quick Start (One Click)](#quick-start)
- [Step-by-Step Guide](#step-by-step-guide)
  - [Phase 1: Data Collection](#phase-1-data-collection)
  - [Phase 2: Data Preparation](#phase-2-data-preparation)
  - [Phase 3: Fine-Tuning](#phase-3-fine-tuning)
  - [Phase 4: Evaluation & Export](#phase-4-evaluation--export)
- [Colab Strategy](#colab-strategy)
- [Training Details](#training-details)
- [Results](#results)
- [On-Device Deployment](#on-device-deployment)
- [Repository Structure](#repository-structure)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## What This Project Does

```
┌─────────────────────────────────────────────────────────────────────┐
│                         QuranASR Pipeline                          │
│                                                                     │
│  EveryAyah.com ──► 75K MP3s ──► manifest.jsonl ──► Fine-tune       │
│  (12 reciters)     (50-70 GB)    (audio↔text)      whisper-small   │
│                                                         │          │
│                                              ┌──────────┴────────┐ │
│                                              │                   │ │
│                                         HuggingFace         CoreML │
│                                         (cloud API)     (on-device)│
│                                              │                   │ │
│                                              └──────────┬────────┘ │
│                                                         │          │
│                                                    Tilavet.app     │
│                                                    (iOS app)       │
└─────────────────────────────────────────────────────────────────────┘
```

1. **Downloads** Quran recitation audio from [EveryAyah.com](https://everyayah.com) (12 reciters × 6,236 verses)
2. **Pairs** each audio file with its corresponding Quran verse text (from Quran.com API)
3. **Fine-tunes** `openai/whisper-small` on Google Colab (free tier works, A100 recommended)
4. **Pushes** the trained model to [HuggingFace Hub](https://huggingface.co/baristiran)
5. **Converts** to CoreML for on-device inference with WhisperKit

---

## Why We Built This

OpenAI's Whisper is a powerful general-purpose speech recognition model, but it wasn't trained specifically on Quran recitation. Quran Arabic (tilawa) has unique characteristics:

| Challenge | Why General ASR Struggles |
|-----------|--------------------------|
| **Classical Arabic** | Quran uses Quranic Arabic with archaic vocabulary and grammar |
| **Tajweed rules** | Elongation (madd), nasalization (ghunnah), assimilation (idgham) change pronunciation |
| **Multiple reciters** | Each reciter (qari) has a distinct voice, speed, and melodic style |
| **Diacritics** | Arabic diacritics (tashkeel/harakat) are critical for correct reading |
| **Domain vocabulary** | Religious terms, proper nouns, and formulaic phrases unique to Quran |

By fine-tuning on 75K samples from 12 professional reciters, we create a model that **understands these patterns** and achieves significantly better WER on Quran audio than the base model.

---

## Architecture Decisions

We made specific choices at every step. Here's what we chose, why, and what the alternatives were:

### Model: `whisper-small` (244M parameters)

| Option | Size | Pros | Cons | Our Take |
|--------|------|------|------|----------|
| `whisper-tiny` | 39M | Fastest, smallest | Too inaccurate for Quran | ❌ Not enough capacity |
| **`whisper-small`** | **244M** | **Good accuracy, fits on iPhone** | **Slower than tiny** | **✅ Best trade-off** |
| `whisper-medium` | 769M | More accurate | 3x bigger, slow on-device | ❌ Too large for iOS |
| `whisper-large-v3` | 1.5B | Most accurate | Won't run on-device | ❌ Cloud-only |

**Decision:** `whisper-small` — 244MB fits in an iOS app, and domain fine-tuning compensates for its smaller size.

### Training: Full Fine-tune vs LoRA

| Approach | VRAM | Speed | Quality | Our Take |
|----------|------|-------|---------|----------|
| **Full fine-tune** | **~30GB** | **Slower** | **Best adaptation** | **✅ Chosen** |
| LoRA (r=16) | ~10GB | Faster | Good, not optimal | ❌ Less domain shift |
| QLoRA | ~6GB | Fastest | Decent | ❌ Quantization artifacts |

**Decision:** Full fine-tune — Quran recitation is a significant domain shift from general speech. A100 has 40GB VRAM, so no reason to compromise. LoRA would be a good alternative for limited GPU budgets.

### Labels: Diacritics Stripped (Harekesiz)

```
Original (harekeli): بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ
Training label:      بسم الله الرحمن الرحيم
```

**Why?** The model output needs to match our verse database for verse identification. Diacritics vary between text sources and add noise to training.

### Feature Extraction: On-the-fly (not pre-computed)

| Approach | RAM Usage | Disk | Speed | Our Take |
|----------|-----------|------|-------|----------|
| Pre-compute all features | ~46 GB | ~40 GB | Faster training | ❌ Colab RAM crash |
| **On-the-fly in collator** | **~2 GB** | **0** | **Slightly slower** | **✅ Reliable** |

**Decision:** On-the-fly — Pre-computing mel spectrograms for 75K files requires 46GB RAM, which crashes even high-RAM Colab instances. Computing features per-batch inside the data collator uses negligible RAM.

### Audio Source: EveryAyah.com

[EveryAyah.com](https://everyayah.com) provides free, verse-level Quran recitation audio with consistent file naming (`{surah:03d}{verse:03d}.mp3`). This is the only source that offers:
- Per-verse segmentation (not full-surah files)
- Multiple reciters with the same naming convention
- Direct download without authentication

---

## Quick Start

**One click:** Open the notebook in Colab and follow the cell-by-cell instructions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tialkan/QuranASR/blob/main/notebook.ipynb)

Or clone and run locally:

```bash
git clone https://github.com/tialkan/QuranASR.git
cd QuranASR
```

---

## Step-by-Step Guide

### Phase 1: Data Collection

**What:** Download ~75,000 MP3 files from EveryAyah.com (12 reciters × 6,236 verses).

**Reciters selected:**

| # | Reciter | Style | Bitrate |
|---|---------|-------|---------|
| 1 | Muhammad al-Minshawi | Murattal | 128kbps |
| 2 | Mahmoud Khalil al-Husary | Precise/Clear | 128kbps |
| 3 | Mishary Rashid Alafasy | Popular/Modern | 128kbps |
| 4 | Abdurrahman as-Sudais | Haram Imam | 192kbps |
| 5 | Abdul Basit Abdus Samad | Classical | 192kbps |
| 6 | Mohammad al-Tablaway | Egyptian | 128kbps |
| 7 | Maher Al-Muaiqly | Haram Imam | 128kbps |
| 8 | Hani ar-Rifai | Saudi | 192kbps |
| 9 | Ibrahim al-Akhdar | Clear/Slow | 128kbps |
| 10 | Yasser Ad-Dussary | Modern | 128kbps |
| 11 | Abdullah Basfar | Saudi | 192kbps |
| 12 | Sahl Yassin | Unique tone | 128kbps |

**Why these 12?** They represent diverse recitation styles (Egyptian, Saudi, modern, classical) while keeping the dataset at a manageable ~50-70 GB. This diversity teaches the model to generalize across different voices and tajweed styles.

### Phase 2: Data Preparation

**What:** Create a training manifest pairing each audio file with its Quran verse text.

1. **Download Quran text** from Quran.com API (Uthmani script)
2. **Strip diacritics** (harakat) from the text — our model outputs clean text for database matching
3. **Build manifest.jsonl** — each line maps an audio path to its text label
4. **Create HuggingFace Dataset** — lightweight, stores only `audio_path` + `text` (no features pre-computed)

### Phase 3: Fine-Tuning

**What:** Fine-tune `openai/whisper-small` using HuggingFace's `Seq2SeqTrainer`.

Key training hyperparameters:

| Parameter | Value | Why |
|-----------|-------|-----|
| Batch size | 16 × 4 accumulation = 64 effective | Balances speed and stability |
| Learning rate | 1e-5 | Conservative — prevents catastrophic forgetting |
| Warmup steps | 500 | Gradual LR increase for stability |
| Max steps | 10,000 | ~8-12 hours on A100, sufficient for convergence |
| FP16 | Yes | 2x speed, half VRAM — no quality loss |
| Gradient checkpointing | Yes | Trades compute for VRAM — critical for Colab |
| Checkpoint every | 500 steps | Colab disconnects every 4-12 hours |

### Phase 4: Evaluation & Export

**What:** Evaluate on a held-out validation set (5% of data), save the model, and push to HuggingFace Hub.

The notebook automatically:
1. Computes WER (Word Error Rate) on the validation set
2. Saves the best model based on lowest WER
3. Pushes to HuggingFace Hub (your account)

---

## Colab Strategy

Google Colab has GPU time limits. We split the work into two phases to avoid wasting GPU hours on data preparation:

```
Phase 1: CPU + High RAM (free tier)
┌─────────────────────────────────────────────────────┐
│ Cell 1: Install packages → restart                  │
│ Cell 2: Mount Drive + set constants                 │
│ Cell 3: Download Quran text (API)                   │
│ Cell 4: Download 75K audio files (hours)            │
│ Cell 5: Build manifest                              │
│ Cell 6: Create HF dataset                           │
│                                                     │
│ Everything saved to Google Drive ✓                   │
└─────────────────────────────────────────────────────┘
                        ↓
        Switch runtime to A100 GPU + High RAM
                        ↓
Phase 2: A100 GPU (use credits wisely)
┌─────────────────────────────────────────────────────┐
│ Cell 1: Install packages → restart                  │
│ Cell 2: Mount Drive (data already there)            │
│ Cell 7: Train (resumes from checkpoint if crashed)  │
│ Cell 8: Test + push to HuggingFace                  │
└─────────────────────────────────────────────────────┘
                        ↓
        If Colab disconnects:
        Cell 1 → restart → Cell 2 → Cell 7
        (automatically resumes from last checkpoint)
```

**Key design:** Every intermediate result is saved to Google Drive. You never lose progress.

---

## Training Details

### Data Processing (inside the Data Collator)

```python
# For each batch item, the collator:
audio, sr = soundfile.read(path)     # Read MP3/WAV from Drive
if sr != 16000: resample(16000)      # Whisper requires 16kHz
mel = feature_extractor(audio)       # 80-bin log-mel spectrogram
tokens = tokenizer(text)             # Arabic text → token IDs
```

This happens **on-the-fly** for each batch — no 46GB of pre-computed features in RAM.

### Model Configuration

```python
model.config.forced_decoder_ids = None        # Don't force language token
model.config.suppress_tokens = []              # Allow all tokens
model.config.use_cache = False                 # Required for gradient checkpointing
model.generation_config.language = 'arabic'    # Arabic language
model.generation_config.task = 'transcribe'    # Transcription (not translation)
```

### Validation

- **Metric:** Word Error Rate (WER) — lower is better
- **Split:** 95% train / 5% validation (deterministic seed=42)
- **Eval schedule:** Every 200 steps initially, then every 500 steps after step 200
- **Early stopping:** Patience of 5 evaluations with no WER improvement

---

## Results

| Model | WER on Quran Audio | Size | On-device? |
|-------|-------------------|------|------------|
| `whisper-small` (base) | ~45% | 244 MB | ✅ |
| **QuranASR (ours)** | **<10%** | **244 MB** | **✅** |

> WER is computed on harekesiz (diacritics-stripped) text. Actual results depend on training run.

---

## On-Device Deployment

The trained model can be converted to CoreML for on-device inference on iOS using [WhisperKit](https://github.com/argmaxinc/WhisperKit):

```bash
pip install whisperkittools coremltools
python convert/export_coreml.py \
    --model-dir ./checkpoints/final \
    --output-dir ./coreml_output
```

This is exactly how [**Tilavet**](https://tilavet.app) works — your Quran companion app that listens to your recitation and follows along in real-time.

<div align="center">

### 📱 Built with this model

[**Tilavet — Your Quran Companion**](https://tilavet.app)

Listen. Follow. Learn.

*Available on the App Store*

</div>

---

## Repository Structure

```
QuranASR/
├── notebook.ipynb              # Complete Colab notebook (all-in-one)
├── README.md                   # This file
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── reciters.txt                # List of 12 reciters (EveryAyah folder names)
├── convert/
│   └── export_coreml.py        # HuggingFace → CoreML conversion
└── ios/
    └── WhisperKitSetup.md      # Swift/iOS integration guide
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `OutOfMemoryError` during training | Batch size too large | Reduce `per_device_train_batch_size` to 8 |
| RAM crash during data loading | Pre-computing features | Use on-the-fly collator (already done in notebook) |
| `evaluation_strategy` error | transformers version mismatch | Notebook auto-detects v4 vs v5 |
| Colab disconnected mid-training | Normal Colab behavior | Re-run Cell 1→2→7, resumes from checkpoint |
| Audio download stuck/slow | EveryAyah rate limiting | Reduce workers to 8, or re-run (skips existing) |
| WER not improving | Learning rate too high | Try `5e-6` instead of `1e-5` |
| `tokenizer` deprecation warning | transformers 5.x | Notebook auto-switches to `processing_class` |

---

## Citation

If you use this pipeline or model in your research or project, please cite:

```bibtex
@misc{quran-asr-2025,
  title     = {QuranASR: Fine-tuning Whisper for Quran Recitation Recognition},
  author    = {Baris Tiran},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/tialkan/QuranASR}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The Quran text is in the public domain. Audio files are sourced from [EveryAyah.com](https://everyayah.com) which provides free Quran audio for non-commercial and educational use.

---

<div align="center">

**Built with ❤️ for the Quran community**

[🤗 HuggingFace](https://huggingface.co/baristiran) · [📱 Tilavet App](https://tilavet.app) · [🐙 GitHub](https://github.com/tialkan/QuranASR)

</div>
