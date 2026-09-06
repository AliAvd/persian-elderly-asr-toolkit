---
title: Persian Elderly ASR
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.8.0
app_file: app.py
pinned: false
license: apache-2.0
models:
- AliAvd/qwen3-asr-persian-elderly
---

# Persian Elderly ASR Demo

رابط آزمایشی بازشناسی گفتار فارسی با مدل Qwen3-ASR تنظیم‌دقیق‌شده برای پروژه گفتار سالمندان.

The interface accepts an uploaded or microphone-recorded audio file of up to 60 seconds and returns a Persian transcription. The model is loaded once and automatically uses a GPU when one is assigned to the Space.

## Model

`AliAvd/qwen3-asr-persian-elderly`

## Limitations

The free CPU runtime can be slow for a 1.7B-parameter model. Transcriptions may contain errors and must not be treated as authoritative in safety-critical workflows.
