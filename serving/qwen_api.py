from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from functools import lru_cache

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from silero_vad import get_speech_timestamps
from starlette.concurrency import run_in_threadpool

SAMPLING_RATE: int = int(os.getenv("SAMPLING_RATE", "16000"))
CHECKPOINT_PATH: str = os.getenv(
    "CHECKPOINT_PATH",
    "AliAvd/qwen3-asr-persian-elderly",
)

app = FastAPI(title="Qwen ASR API")
INFERENCE_LOCK = threading.Lock()


def decode_uploaded_audio(source_path: str) -> str:
    """Decode any FFmpeg-supported upload to mono 16 kHz PCM WAV."""
    descriptor, decoded_path = tempfile.mkstemp(prefix="asr_decoded_", suffix=".wav")
    os.close(descriptor)
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        source_path,
        "-ac",
        "1",
        "-ar",
        str(SAMPLING_RATE),
        "-c:a",
        "pcm_s16le",
        decoded_path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            reason = result.stderr.strip() or "unknown FFmpeg decoding error"
            raise ValueError(f"Uploaded file is not valid or supported audio: {reason}")
        return decoded_path
    except Exception:
        try:
            os.unlink(decoded_path)
        except OSError:
            pass
        raise


@lru_cache(maxsize=1)
def get_model():
    import torch
    from qwen_asr import Qwen3ASRModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = Qwen3ASRModel.from_pretrained(
        CHECKPOINT_PATH,
        dtype=torch_dtype,
        device_map=device,
    )

    return model


@lru_cache(maxsize=1)
def get_vad():
    from silero_vad import load_silero_vad

    return load_silero_vad()


@app.post("/analyse")
@app.post("/analyze", include_in_schema=False)
async def analyse(
    file: UploadFile = File(...), do_asr: bool | None = Form(True)
) -> list[dict[str, float | str]]:
    suffix = ""
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[-1]

    try:
        with tempfile.NamedTemporaryFile(
            prefix="asr_", suffix=suffix, delete=False
        ) as tmp:
            audio_path = tmp.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to read upload: {exc}"
        ) from exc

    decoded_audio_path: str | None = None
    inference_lock_acquired = False
    try:
        if os.path.getsize(audio_path) == 0:
            raise ValueError("Uploaded file is empty.")
        decoded_audio_path = await run_in_threadpool(decode_uploaded_audio, audio_path)
        await run_in_threadpool(INFERENCE_LOCK.acquire)
        inference_lock_acquired = True
        vad = await run_in_threadpool(get_vad)
        model = await run_in_threadpool(get_model)

        def _do_transcribe() -> list[dict[str, float | str]]:
            audio, _ = librosa.load(decoded_audio_path, sr=SAMPLING_RATE, mono=True)

            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak
            else:
                return []

            speech_chunks = get_speech_timestamps(
                audio,
                vad,
                sampling_rate=SAMPLING_RATE,
                speech_pad_ms=60,
            )

            if len(speech_chunks) > 0:
                chunks_meta = [{} for _ in range(len(speech_chunks))]
                if do_asr:
                    chunk_paths = []
                    for chunk in speech_chunks:
                        with tempfile.NamedTemporaryFile(
                            prefix="asr_", suffix=".wav", delete=False
                        ) as tmp:
                            audio_chunk = audio[chunk["start"] : chunk["end"]]
                            sf.write(tmp.name, audio_chunk, SAMPLING_RATE)
                            chunk_paths.append(tmp.name)

                    for t_i, chunk_path in enumerate(chunk_paths):
                        try:
                            transcription = model.transcribe(
                                chunk_path, language="Persian"
                            )

                            if (
                                isinstance(transcription, dict)
                                and "text" in transcription
                            ):
                                chunks_meta[t_i]["text"] = transcription["text"]
                            else:
                                chunks_meta[t_i]["text"] = str(transcription[0].text)

                        except Exception as e:
                            chunks_meta[t_i]["text"] = f"Transcription error: {e}"

                    for chunk_path in chunk_paths:
                        os.unlink(chunk_path)

                return [
                    {
                        "start": np.round(chunk["start"] / SAMPLING_RATE, 2),
                        "end": np.round(chunk["end"] / SAMPLING_RATE, 2),
                        "duration": np.round(
                            (chunk["end"] - chunk["start"]) / SAMPLING_RATE, 2
                        ),
                        **chunk_meta,
                    }
                    for chunk, chunk_meta in zip(speech_chunks, chunks_meta)
                ]
            else:
                return []

        texts = await run_in_threadpool(_do_transcribe)
        return texts
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ASR failed: {exc}") from exc
    finally:
        if inference_lock_acquired:
            INFERENCE_LOCK.release()
        if decoded_audio_path is not None:
            try:
                os.unlink(decoded_audio_path)
            except OSError:
                pass
        try:
            os.unlink(audio_path)
        except OSError:
            pass


@app.get("/")
async def root():
    return {"message": "Qwen ASR API is running! Visit /docs to test it."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
