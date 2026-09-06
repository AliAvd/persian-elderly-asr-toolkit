#!/usr/bin/env python3
"""Verify privacy and remote file manifests for the published HF repositories."""

from huggingface_hub import HfApi


def main() -> None:
    api = HfApi()
    repos = [
        ("AliAvd/persian-asr-dataset", "dataset"),
        ("AliAvd/persian-elderly-asr", "dataset"),
        ("AliAvd/qwen3-asr-persian-elderly", "model"),
    ]
    for repo_id, repo_type in repos:
        info = api.repo_info(repo_id, repo_type=repo_type, files_metadata=True)
        files = info.siblings
        print(
            repo_id,
            f"private={info.private}",
            f"files={len(files)}",
            f"bytes={sum((item.size or 0) for item in files)}",
        )
        print(*[item.rfilename for item in files], sep="\n  ")


if __name__ == "__main__":
    main()
