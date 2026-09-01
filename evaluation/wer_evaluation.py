import json
import os

import pandas as pd
import torch
from jiwer import cer, wer
from qwen_asr import Qwen3ASRModel
from tqdm import tqdm

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# =========================================================
# CONFIG
# =========================================================

TEST_JSONL = "./jsonl_dataset/test.jsonl"

CHECKPOINT_DIRS = [
    os.getenv("QWEN_CHECKPOINT", "AliAvd/qwen3-asr-persian-elderly"),
]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# batch size for inference
BATCH_SIZE = 32

# Threshold for saving to CSV
WER_THRESHOLD = 0.4

# =========================================================
# LOAD TEST DATA
# =========================================================

test_samples = []

with open(TEST_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        audio_path = item["audio"]
        raw_text = item["text"]

        # Extract only the ASR transcription
        if "<asr_text>" in raw_text:
            text = raw_text.split("<asr_text>", 1)[1].strip()
        else:
            text = raw_text.strip()

        test_samples.append(
            {
                "audio": audio_path,
                "text": text,
            }
        )

print(f"Loaded {len(test_samples)} test samples")


# =========================================================
# EVALUATE FUNCTION
# =========================================================


def evaluate_checkpoint(checkpoint_path):

    print(f"\nLoading model from: {checkpoint_path}")

    model = Qwen3ASRModel.from_pretrained(
        checkpoint_path,
        dtype=DTYPE,
        device_map=DEVICE,
    )

    references = []
    predictions = []

    # Lists specifically for the CSV (filtered samples)
    csv_audio_paths = []
    csv_references = []
    csv_predictions = []
    csv_wers = []

    # -----------------------------------------------------
    # Batched inference
    # -----------------------------------------------------

    for i in tqdm(range(0, len(test_samples), BATCH_SIZE)):
        batch = test_samples[i : i + BATCH_SIZE]

        audio_paths = [x["audio"] for x in batch]
        ref_texts = [x["text"] for x in batch]

        try:
            results = model.transcribe(audio=audio_paths, language="Persian")
            pred_texts = [r.text.strip() for r in results]

        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue

        # Keep track of global lists for the total summary metrics
        references.extend(ref_texts)
        predictions.extend(pred_texts)

        # check individual WERs for thresholding
        for audio, ref, pred in zip(audio_paths, ref_texts, pred_texts):
            try:
                # Calculate WER for this specific sample
                sample_wer = wer(ref, pred)
            except Exception:
                # Handle edge cases (e.g., empty reference text)
                sample_wer = 1.0

            # Only add to CSV lists if it exceeds the threshold
            if sample_wer > WER_THRESHOLD:
                csv_audio_paths.append(audio)
                csv_references.append(ref)
                csv_predictions.append(pred)
                csv_wers.append(sample_wer)

    # -----------------------------------------------------
    # Compute summary metrics (for the whole dataset)
    # -----------------------------------------------------
    model_wer = wer(references, predictions)
    model_cer = cer(references, predictions)

    # Save CSV with filtered samples only
    df = pd.DataFrame(
        {
            "audio_path": csv_audio_paths,
            "reference": csv_references,
            "prediction": csv_predictions,
            "sample_wer": csv_wers,  # Included so you can see how bad the prediction was
        }
    )

    csv_path = checkpoint_path.replace("/", "_") + "_high_wer_predictions.csv"

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved CSV (Samples with WER > {WER_THRESHOLD}): {csv_path} (Total: {len(df)} samples)"
    )

    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Overall WER: {model_wer:.4f}")
    print(f"Overall CER: {model_cer:.4f}")

    return {
        "checkpoint": checkpoint_path,
        "wer": model_wer,
        "cer": model_cer,
    }


# =========================================================
# RUN EVALUATION
# =========================================================

all_results = []

for ckpt in CHECKPOINT_DIRS:
    result = evaluate_checkpoint(ckpt)
    all_results.append(result)

# =========================================================
# PRINT SUMMARY
# =========================================================

print("\n================ FINAL RESULTS ================")

for r in all_results:
    print(f"{r['checkpoint']} | WER={r['wer']:.4f} | CER={r['cer']:.4f}")

# Best checkpoint
best = min(all_results, key=lambda x: x["wer"])

print("\nBest checkpoint based on WER:")
print(best)
