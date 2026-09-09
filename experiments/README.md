# ASR experiment pipeline

The catalog contains 30 scenarios and 60 model configurations for Qwen3-ASR, Wav2Vec2-base and XLSR-53: public Persian and elderly adaptation, SeniorTalk transfer, augmentation/ablations, teacher filtering, data-volume controls, long recordings and noisy evaluation. Configurations specify ten epochs; change the catalog configuration when another training budget is required.

## Environment

Run commands from the repository root. Install a compatible CUDA PyTorch build for the host before launching GPU jobs; the requirements record the original environment's direct dependencies.

```bash
python3 -m venv experiments/.venv
source experiments/.venv/bin/activate
pip install -r experiments/requirements.txt
```

## Data and checkpoints

Data and model weights are not distributed in Git. The runner resolves relative configuration paths against `experiments/`. Place prepared manifests under `experiments/data/manifests/`, or edit the catalog paths. Supply the baseline checkpoints with `--checkpoint` or the environment variable named by each model configuration.

Corpus preparation expects local source folders alongside `experiments/`: `Final_Gathered_Dataset_HF`, `Final_Gatheres_Dataset_Chunks`, `asr_dataset`, `commonvoice_fa`, and `datasets/raw/seniortalk`. `shared/prepare_data.py` includes the original local Filimo cache path; configure that path for your host. Augmentation preparation expects background/RIR assets under `qwen_asr/backgrounds` and `qwen_asr/Impulses` alongside `experiments/`.

The optional downloader uses the upstream dataset access requirements; inspect `python experiments/download_datasets.py --help` before downloading.

```bash
python experiments/shared/prepare_data.py all
python experiments/shared/prepare_data.py augmentation
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

Set the baseline checkpoint paths in `campaign.py` to match the host. The campaign uses `experiments/.venv/bin/python` and writes state under `experiments/runs/`.

```bash
python experiments/campaign.py prepare
python experiments/campaign.py run --gpus 0
python experiments/campaign.py report
```

`retry-failed` retries failed campaign jobs. The campaign tracks dependencies and available GPU memory. Review data readiness and resource availability before starting training.

## Collect and test

```bash
python experiments/collect_results.py --outputs experiments/outputs
python -B -m unittest discover -s experiments/tests -v
```

Result collection writes local summaries; generated artifacts are ignored by Git. Metrics record raw and normalized WER/CER separately. Tests using local augmentation data skip when those assets are absent.
