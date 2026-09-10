from __future__ import annotations

import os
import time
from functools import lru_cache

import gradio as gr
import librosa
import torch
from qwen_asr import Qwen3ASRModel

MODEL_ID = os.getenv("MODEL_ID", "AliAvd/qwen3-asr-persian-elderly")
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "60"))
if MAX_AUDIO_SECONDS <= 0:
    raise ValueError("MAX_AUDIO_SECONDS must be positive")


@lru_cache(maxsize=1)
def load_model() -> Qwen3ASRModel:
    use_cuda = torch.cuda.is_available()
    return Qwen3ASRModel.from_pretrained(
        MODEL_ID,
        token=os.getenv("HF_TOKEN"),
        dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="cuda:0" if use_cuda else "cpu",
    )


def transcribe(audio_path: str | None) -> tuple[str, str]:
    if not audio_path:
        raise gr.Error("لطفاً یک فایل صوتی بارگذاری یا صدایی ضبط کنید.")

    duration = librosa.get_duration(path=audio_path)
    if duration > MAX_AUDIO_SECONDS:
        raise gr.Error(f"طول صدا باید حداکثر {MAX_AUDIO_SECONDS} ثانیه باشد.")

    started = time.perf_counter()
    results = load_model().transcribe(audio=audio_path, language="Persian")
    if not results:
        return "", "هیچ گفتاری تشخیص داده نشد."

    result = results[0]
    text = result.text.strip() if hasattr(result, "text") else str(result).strip()
    elapsed = time.perf_counter() - started
    device = "GPU" if torch.cuda.is_available() else "CPU"
    status = (
        f"مدت صوت: {duration:.1f} ثانیه — زمان پردازش: {elapsed:.1f} ثانیه — {device}"
    )
    return text, status


with gr.Blocks(title="بازشناسی گفتار فارسی سالمندان") as demo:
    gr.Markdown(
        f"""
        # بازشناسی گفتار فارسی سالمندان

        نمونه آزمایشی مدل تنظیم‌دقیق‌شده Qwen3-ASR. یک فایل صوتی کوتاه بارگذاری کنید
        یا با میکروفن صدا ضبط کنید. حداکثر طول ورودی {MAX_AUDIO_SECONDS:g} ثانیه است.
        """
    )
    audio = gr.Audio(
        sources=["upload", "microphone"],
        type="filepath",
        label="صدای ورودی",
    )
    run_button = gr.Button("تبدیل گفتار به متن", variant="primary")
    transcript = gr.Textbox(label="متن تشخیص‌داده‌شده", lines=5, rtl=True)
    status = gr.Textbox(label="وضعیت", interactive=False, rtl=True)
    run_button.click(transcribe, inputs=audio, outputs=[transcript, status])
    gr.Markdown(
        "مدل: `AliAvd/qwen3-asr-persian-elderly` — پژوهش علی الوندی، با راهنمایی حسین صامتی"
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
