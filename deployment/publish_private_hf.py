#!/usr/bin/env python3
"""Publish the project's two datasets and inference-only model to private HF repos."""

from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = "AliAvd"
GANJOOR_REPO = f"{NAMESPACE}/persian-asr-dataset"
ELDERLY_REPO = f"{NAMESPACE}/persian-elderly-asr"
MODEL_REPO = f"{NAMESPACE}/qwen3-asr-persian-elderly"


def main() -> None:
    api = HfApi()
    for repo_id, repo_type in (
        (GANJOOR_REPO, "dataset"),
        (ELDERLY_REPO, "dataset"),
        (MODEL_REPO, "model"),
    ):
        api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)

    print("Uploading Ganjoor dataset...")
    load_from_disk(str(ROOT / "asr_dataset")).push_to_hub(
        GANJOOR_REPO, private=True, max_shard_size="500MB"
    )
    api.upload_file(
        path_or_fileobj=str(ROOT / "asr_dataset/README.md"),
        path_in_repo="README.md",
        repo_id=GANJOOR_REPO,
        repo_type="dataset",
        commit_message="Add Ganjoor dataset card",
    )

    print("Uploading elderly-speech dataset...")
    load_from_disk(str(ROOT / "Final_Gathered_Dataset_HF")).push_to_hub(
        ELDERLY_REPO, private=True, max_shard_size="500MB"
    )
    api.upload_folder(
        folder_path=str(ROOT / "Final_Gathered_Dataset_HF/qwen_jsonl"),
        path_in_repo="qwen_jsonl",
        repo_id=ELDERLY_REPO,
        repo_type="dataset",
        commit_message="Add Qwen JSONL manifests",
    )
    for filename in ("rejected.jsonl", "processing_summary.json", "README.md"):
        api.upload_file(
            path_or_fileobj=str(ROOT / "Final_Gathered_Dataset_HF" / filename),
            path_in_repo=filename,
            repo_id=ELDERLY_REPO,
            repo_type="dataset",
            commit_message=f"Add {filename}",
        )

    print("Uploading inference-only model...")
    checkpoint = ROOT / "qwen_asr/qwen3-asr-finetuning-v3/checkpoint-856300"
    inference_files = [
        "README.md",
        "added_tokens.json",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    api.upload_folder(
        folder_path=str(checkpoint),
        repo_id=MODEL_REPO,
        repo_type="model",
        allow_patterns=inference_files,
        commit_message="Upload inference-only Qwen3-ASR checkpoint",
    )

    print("Published private repositories:")
    print(f"https://huggingface.co/datasets/{GANJOOR_REPO}")
    print(f"https://huggingface.co/datasets/{ELDERLY_REPO}")
    print(f"https://huggingface.co/{MODEL_REPO}")


if __name__ == "__main__":
    main()
