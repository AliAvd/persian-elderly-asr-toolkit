# ASR experiment pipeline

The catalog contains 30 scenarios and 60 model configurations for Qwen3-ASR, Wav2Vec2-base and XLSR-53: public Persian and elderly adaptation, SeniorTalk transfer, augmentation/ablations, teacher filtering, data-volume controls, long recordings and noisy evaluation. Configurations specify ten epochs. Single-experiment runs honor the catalog training budget; the campaign explicitly fixes ten epochs. Change the campaign implementation as well if a different campaign budget is required.

## Environment

Use Python 3.11 or 3.12. Run commands from the repository root. Install a compatible CUDA PyTorch build for the host before launching GPU jobs; the requirements record the original environment's direct dependencies.

```bash
python3 -m venv experiments/.venv
source experiments/.venv/bin/activate
pip install -r experiments/requirements.txt
```

## Data and checkpoints

Data and model weights are not distributed in Git. The runner resolves relative configuration paths against `experiments/`. Place prepared manifests under `experiments/data/manifests/`, or edit the catalog paths. Supply the baseline checkpoints with `--checkpoint` or the environment variable named by each model configuration.

Corpus preparation expects local source folders alongside `experiments/`: `Final_Gathered_Dataset_HF`, `Final_Gatheres_Dataset_Chunks`, `asr_dataset`, `commonvoice_fa`, and `datasets/raw/seniortalk`. Set `ASR_FILIMO_CACHE` to the directory containing the ten original `filimo-persian-asr-unvalidated-*.arrow` shards (default: `datasets/raw/filimo` under the data root). Augmentation preparation expects background/RIR assets under `qwen_asr/backgrounds` and `qwen_asr/Impulses` under the data root. Override those asset paths with `ASR_BACKGROUND_DIR` and `ASR_RIR_DIR`. Set `ASR_DATA_ROOT` to keep source corpora outside the repository; it is shared by the downloader, source preparation and final-corpus tools. Prepared experiment manifests still go under `experiments/data/`.

The optional downloader uses the upstream dataset access requirements; inspect `python experiments/download_datasets.py --help` before downloading.

```bash
python experiments/shared/prepare_data.py all
python experiments/check_readiness.py --verbose
```

`prepare_extra.py` supplies `noisy`, `filter` (with `--checkpoint`) and `long` preparation tasks. Corpus building and validation utilities also live in `dataset_tools/`.

## Run an experiment

```bash
python experiments/catalog/03_public/qwen/run.py --dry-run --print-config
python experiments/catalog/03_public/qwen/run.py
```

Run the dry-run check first after supplying local data. Adaptation scenarios require the checkpoints from their preceding training stages. `run_experiment.py --help` describes checkpoint, seed, output directory and resume overrides. No experiment runs simply by importing its catalog entry.

## Run the campaign

Supply local Persian baseline checkpoint paths through the environment before preparing a new campaign:

```bash
export ASR_WAV2VEC2_BASE_CHECKPOINT=/path/to/persian-base-checkpoint
export ASR_WAV2VEC2_XLSR_CHECKPOINT=/path/to/persian-xlsr-checkpoint
```

The campaign uses the active Python interpreter and writes state under `experiments/runs/`. Internal adaptation dependencies are relocated into the campaign output directory; external baseline paths are preserved. Baseline variables apply only to scenarios 01 and 02; scenarios 28–30 evaluate the models from scenario 07.

Campaign preparation snapshots the configurations. To apply changed paths/configurations, pass a new `--directory` consistently to `prepare`, `run` and `report`. `prepare` is idempotent for an existing campaign.

```bash
python experiments/campaign.py prepare
python experiments/campaign.py run --gpus 0
python experiments/campaign.py report
```

`retry-failed` resets failed campaign jobs to pending; run the campaign again afterwards. The campaign tracks dependencies and available GPU memory. Review data readiness and resource availability before starting training.

## Collect and test

```bash
python experiments/collect_results.py --outputs experiments/outputs
pip install fastapi python-multipart httpx
python -B -m unittest discover -s experiments/tests -v
```

Result collection writes local summaries; generated artifacts are ignored by Git. Metrics record raw and normalized WER/CER separately. Tests using local augmentation data skip when those assets are absent.
