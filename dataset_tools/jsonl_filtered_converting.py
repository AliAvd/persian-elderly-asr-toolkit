# Environments
import json
import os
import warnings
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


# Disable warnings
warnings.filterwarnings("ignore")

# Imports

import torch
from datasets import (
    Audio,
    DatasetDict,
    DownloadConfig,
    concatenate_datasets,
    load_dataset,
)
from jiwer import wer

# Qwen and Metrics Imports
from qwen_asr import Qwen3ASRModel

from text_preprocessor import TextPreprocessor

# ==========================================================
# CONFIGURATION & CONSTANTS
# ==========================================================
SAMPLING_RATE: int = 16_000
MAP_NUM_PROC: int = 100
LOAD_DATASET_KWARGS = {
    "trust_remote_code": True,
    "num_proc": 10,
    "download_config": DownloadConfig(local_files_only=True),
}

MIN_AUDIO_DURATION: float = 2.0
MAX_AUDIO_DURATION: float = 10.0
MIN_TEXT_LENGTH: int = 5

# Inference Model Config
FILTER_CHECKPOINT = os.getenv("FILTER_CHECKPOINT", "AliAvd/qwen3-asr-persian-elderly")
DEVICE = os.getenv("FILTER_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
WER_THRESHOLD = 0.2

# ==========================================================
# DATA LOADING & PROCESSING
# ==========================================================
def prepare_dataset():
    ## Common Voice
    cv_language = "Persian"
    cv_language_abbr = "fa"
    cv_dataset_name = "mozilla-foundation/common_voice_13_0"

    common_voice_ds = load_dataset(
        cv_dataset_name,
        cv_language_abbr,
        split={"train": "train+validation", "test": "test"},
        **LOAD_DATASET_KWARGS,
    )
    common_voice_ds = common_voice_ds.remove_columns(
        [
            "accent",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "segment",
            "up_votes",
            "variant",
        ]
    )
    common_voice_ds = common_voice_ds.cast_column(
        "audio", Audio(sampling_rate=SAMPLING_RATE, decode=False)
    )

    ## Filimo
    filimo_dataset_name = "PerSets/filimo-persian-asr"
    filimo_ds = load_dataset(
        filimo_dataset_name,
        split={"train": "unvalidated[:90%]", "test": "unvalidated[90%:]"},
        **LOAD_DATASET_KWARGS,
        cache_dir=os.getenv("HF_DATASETS_CACHE"),
    )
    filimo_ds = filimo_ds.remove_columns("file_name")
    filimo_ds = filimo_ds.rename_column("text", "sentence")
    filimo_ds = filimo_ds.cast_column(
        "audio", Audio(sampling_rate=SAMPLING_RATE, decode=False)
    )

    ## Ganjoor
    ganjoor_ds = load_dataset(
        path=os.getenv("GANJOOR_DATASET", "AliAvd/persian-asr-dataset"),
        split={"train": "train[:90%]", "test": "train[90%:]"},
        **LOAD_DATASET_KWARGS,
    )
    ganjoor_ds = ganjoor_ds.remove_columns("speaker_id")
    ganjoor_ds = ganjoor_ds.cast_column(
        "audio", Audio(sampling_rate=SAMPLING_RATE, decode=False)
    )

    ## Merging Datasets
    all_dss = [common_voice_ds, filimo_ds, ganjoor_ds]
    ds_merged = {}
    for split in ["train", "test"]:
        ds_merged[split] = concatenate_datasets([ds[split] for ds in all_dss])
    ds_merged = DatasetDict(ds_merged)

    ## Clean merged dataset
    text_preprocessor = TextPreprocessor()


    def standardize_sentence(item):
        item["sentence"] = text_preprocessor.standardize(item["sentence"])
        return item


    ds_merged = ds_merged.map(standardize_sentence, num_proc=MAP_NUM_PROC)


    def is_item_valid(item):
        try:
            audio = Audio(sampling_rate=SAMPLING_RATE).decode_example(item["audio"])
            return bool(
                MIN_AUDIO_DURATION
                < len(audio["array"]) / audio["sampling_rate"]
                <= MAX_AUDIO_DURATION
                and len(item["sentence"]) > MIN_TEXT_LENGTH
            )
        except (OSError, RuntimeError, ValueError, TypeError):
            return False


    ds_merged = ds_merged.filter(is_item_valid, num_proc=MAP_NUM_PROC)

    # NOTE: Keeping decode=False ensures that extract_audio_path gets
    # the direct disk string path required by model.transcribe()
    ds_merged = ds_merged.cast_column(
        "audio", Audio(sampling_rate=SAMPLING_RATE, decode=False)
    )


    # ==========================================================
    # EXPORT WITH WER FILTERING LOGIC
    # ==========================================================
    return ds_merged


def extract_audio_path(sample):
    if "audio" in sample and isinstance(sample["audio"], dict):
        path = sample["audio"].get("path")
        if path is not None:
            return path
    if "path" in sample:
        return sample["path"]
    raise ValueError("No audio path found in sample.")


def dataset_split_to_jsonl(split_dataset, output_path, model, max_samples=None):
    """
    Exports samples to JSONL only if the model's transcription has a WER < WER_THRESHOLD.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        skipped_count = 0

        for sample in split_dataset:
            if max_samples is not None and count >= max_samples:
                break

            try:
                audio_path = extract_audio_path(sample)
                sentence = str(sample["sentence"]).strip()

                # 1. Transcribe using your fine-tuned Qwen Model
                results = model.transcribe(audio=[audio_path], language="Persian")
                pred_text = results[0].text.strip()

                # 2. Calculate Word Error Rate (WER)
                sample_wer = wer(sentence, pred_text)

                # 3. Filter Threshold Check
                if sample_wer >= WER_THRESHOLD:
                    skipped_count += 1
                    continue

                # 4. Save to JSONL if accurate
                item = {
                    "audio": audio_path,
                    "text": f"language Persian<asr_text>{sentence}",
                }

                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1

                # Progress indicator updated for inference tracking
                if count % 100 == 0:
                    print(
                        f"Progress: Kept {count} samples | Filtered out {skipped_count} samples..."
                    )

            except Exception as e:
                print(f"Skipping sample due to error: {e}")

    print(
        f"Saved {count} filtered samples to {output_path} (Filtered out {skipped_count})"
    )


def datasetdict_to_jsonl(
    dataset_dict: DatasetDict, output_dir, model, max_samples=None
):
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_dataset in dataset_dict.items():
        print(f"\nConverting split: {split_name} ({len(split_dataset)} samples)")
        output_path = os.path.join(output_dir, f"{split_name}.jsonl")

        dataset_split_to_jsonl(
            split_dataset=split_dataset,
            output_path=output_path,
            model=model,
            max_samples=max_samples,
        )


# ==========================================================
# EXECUTION
# ==========================================================
if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description="Legacy Common Voice 13 teacher-filtering workflow; use experiments/ for CV26.").parse_args()
    ds_merged = prepare_dataset()
    # Load your best fine-tuned model for filtering
    print(f"\nLoading filtering model from: {FILTER_CHECKPOINT}")
    filter_model = Qwen3ASRModel.from_pretrained(
        FILTER_CHECKPOINT,
        dtype=DTYPE,
        device_map=DEVICE,
    )

    OUTPUT_DIR = "./jsonl_dataset_filtered"

    # Export and filter the dataset
    datasetdict_to_jsonl(
        dataset_dict=ds_merged,
        output_dir=OUTPUT_DIR,
        model=filter_model,
    )
