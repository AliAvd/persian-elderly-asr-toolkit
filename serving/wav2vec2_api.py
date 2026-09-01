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

BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "5"))
SAMPLING_RATE: int = int(os.getenv("SAMPLING_RATE", "16000"))


app = FastAPI(title="ASR API")


@lru_cache(maxsize=1)
def get_pipeline():
    import torch
    from safetensors.torch import load_file
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    processor = Wav2Vec2Processor.from_pretrained(
        "./wav2vec2-large-960h-lv60-self-v5/final/processor"
    )

    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-960h-lv60-self",
        local_files_only=True,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=False,  # https://huggingface.co/ylacombe/w2v-bert-2.0/discussions/12#65bb663466552223514bac27
        pad_token_id=processor.tokenizer.pad_token_id,
        # vocab_size = len(processor.tokenizer),
    )
    model.lm_head = torch.nn.Sequential(
        # torch.nn.Linear(model.config.hidden_size, model.config.hidden_size, bias=True),
        # torch.nn.GELU(),
        torch.nn.Linear(model.config.hidden_size, model.config.hidden_size, bias=True),
        torch.nn.GELU(),
        torch.nn.Linear(model.config.hidden_size, model.config.hidden_size, bias=True),
        torch.nn.GELU(),
        torch.nn.Linear(model.config.hidden_size, len(processor.tokenizer), bias=True),
    )
    model.config.vocab_size = len(processor.tokenizer)
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    state_dict = load_file(
        "./wav2vec2-large-960h-lv60-self-v5/final/model/model.safetensors"
    )
    model.load_state_dict(state_dict, strict=False)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch_dtype,
        device=device,
    )


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
        pipeline = await run_in_threadpool(get_pipeline)

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

                    transcriptions = pipeline(
                        chunk_paths,
                        batch_size=BATCH_SIZE,
                    )

                    for t_i, t in enumerate(transcriptions):
                        chunks_meta[t_i]["text"] = t["text"]

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
