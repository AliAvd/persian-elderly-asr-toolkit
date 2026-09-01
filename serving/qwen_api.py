from __future__ import annotations

import os
import tempfile
from functools import lru_cache

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from silero_vad import get_speech_timestamps
from starlette.concurrency import run_in_threadpool

# You may not need BATCH_SIZE anymore if model.transcribe processes one file at a time,
# but it is kept here in case your Qwen implementation supports list inputs.
SAMPLING_RATE: int = int(os.getenv("SAMPLING_RATE", "16000"))
CHECKPOINT_PATH: str = os.getenv("CHECKPOINT_PATH", "AliAvd/qwen3-asr-persian-elderly")

app = FastAPI(title="Qwen ASR API")


@lru_cache(maxsize=1)
def get_model():
    import torch

    # NOTE: Update this import path to wherever your Qwen3ASRModel class is defined
    from qwen_asr import Qwen3ASRModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Initialize Qwen ASR using your provided snippet
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

    try:
        vad = await run_in_threadpool(get_vad)
        model = await run_in_threadpool(get_model)

        def _do_transcribe() -> list[dict[str, float | str]]:
            audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)

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

                    # Updated transcription loop for Qwen
                    for t_i, chunk_path in enumerate(chunk_paths):
                        try:
                            # Pass the temporary chunk path to Qwen's transcribe method
                            transcription = model.transcribe(
                                chunk_path, language="Persian"
                            )

                            # Handle output robustly (whether your specific Qwen implementation
                            # returns a plain string or a dictionary containing 'text')
                            if (
                                isinstance(transcription, dict)
                                and "text" in transcription
                            ):
                                chunks_meta[t_i]["text"] = transcription["text"]
                            else:
                                chunks_meta[t_i]["text"] = str(transcription[0].text)

                        except Exception as e:
                            chunks_meta[t_i]["text"] = f"Transcription error: {e}"

                    # Clean up temporary chunk files
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
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ASR failed: {exc}") from exc
    finally:
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
