# Persian Elderly ASR Toolkit

کدهای گردآوری داده، آماده‌سازی دیتاست، تنظیم دقیق، ارزیابی و سرویس‌دهی پروژه بازشناسی گفتار فارسی با تمرکز بر گفتار سالمندان.

This repository contains the data collection, dataset preparation, fine-tuning, evaluation, and serving utilities developed for a Persian ASR project focused on elderly speech.

## Repository structure

### `crawling/`

- `crawler.py`: خزش محتوای صوتی و متن متناظر از گنجور، دریافت فایل‌ها و تولید اطلاعات هم‌ترازی.
- `creating_dataset.py`: تجمیع فایل‌های `alignment.csv` خزش‌شده، اعتبارسنجی صوت و ساخت دیتاست محلی Hugging Face با ستون‌های `audio`، `sentence` و `speaker_id`.

### `dataset_tools/`

- `chunk_long_recordings.py`: قطعه‌بندی ضبط‌های بلند با Silero VAD و هم‌ترازی متن مرجع با چانک‌ها به کمک فرضیه موقت ASR. خروجی شامل WAV، متن هم‌نام، manifest و جدول بازبینی است.
- `build_hf_dataset.py`: ساخت `DatasetDict` استاندارد Hugging Face از manifest، اعمال فیلترهای کیفی و ایجاد تفکیک‌های بدون نشت در سطح فایل مبدأ.
- `merge_chunk_outputs.py`: ادغام امن یک اجرای افزایشی قطعه‌بندی با خروجی قبلی، همراه با کنترل شناسه‌های تکراری.
- `common_voice_local.py`: تبدیل نسخهٔ محلی Mozilla Common Voice به `DatasetDict` استاندارد.
- `jsonl_filtered_converting.py`: مسیر قدیمی پالایش معلم با Common Voice 13، Filimo و گنجور. برای آزمایش‌های فعلی با Common Voice 26 از `experiments/` استفاده کنید؛ این اسکریپت به کش محلی دیتاست‌ها نیاز دارد.
- `build_final_dataset.py` و `validate_final_dataset.py`: ساخت پیکرهٔ نهایی با هم‌ترازی CTC و بررسی صوت، متن و جدایی تقسیم‌ها.
- `publish_final_dataset.py` و `verify_final_hub.py`: آماده‌سازی نسخهٔ قابل‌انتقال، انتشار خصوصی و بررسی آن در Hugging Face.

### `text_preprocessor/`

- `__init__.py`: رابط اصلی `TextPreprocessor` و زنجیره استانداردسازی فارسی.
- `arabic_alphabet.py`: جدول‌ها و قواعد نویسه‌های عربی و صورت‌های جایگزین.
- `persian_alphabet.py`: تعریف نویسه‌ها و علائم فارسی.
- `reform.py`: اصلاح و یکسان‌سازی نویسه‌ها و نشانه‌های متن.

### `training/`

- `qwen3_asr_sft.py`: تنظیم دقیق مستقل Qwen3-ASR؛ داده‌افزایی با `--background-dir` و `--rir-dir` تنظیم یا با `--no-augmentation` غیرفعال می‌شود. خروجی نهایی شامل مدل و processor است.
- `wav2vec2_ctc.py`: اسکریپت خط‌فرمان مشترک برای تنظیم دقیق `Wav2Vec2-base` و `Wav2Vec2-XLSR-53` با هدف CTC.

### `evaluation/`

- `wer_evaluation.py`: استنتاج دسته‌ای Qwen3-ASR، محاسبه WER و CER و ذخیره نمونه‌های دارای WER بالا برای تحلیل خطا.
- `wav2vec2_evaluation.py`: ارزیابی checkpointهای Wav2Vec2 روی فایل آزمون JSONL و ذخیرهٔ پیش‌بینی‌ها.

### `serving/`

- `qwen_api.py`: سرویس FastAPI برای Qwen3-ASR همراه با قطعه‌بندی گفتار توسط Silero VAD.
- `wav2vec2_api.py`: سرویس FastAPI متناظر برای مدل Wav2Vec2-CTC و مقایسه سرویس‌دهی.

### `space/`

- `app.py`: رابط Gradio برای آزمایش تعاملی مدل منتشرشده در Hugging Face Spaces.

### `deployment/`

- `publish_private_hf.py`: انتشار خصوصی دیتاست‌های گنجور و سالمندان و checkpoint استنتاجی.
- `verify_hf_publish.py`: بررسی وضعیت و فایل‌های مخازن Hugging Face پس از انتشار.

وزن مدل‌ها و checkpointهای آموزش در این مخزن نیستند. شناسهٔ پیش‌فرض مدل Qwen در بخش مخازن داده و مدل آمده است؛ استفاده از مخزن خصوصی به حساب مجاز نیاز دارد.

## Installation

Use Python 3.11 or 3.12 and install FFmpeg. Install a PyTorch/torchaudio build appropriate for your CPU or CUDA runtime; the experiment pipeline has its own recorded dependency versions.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Common workflows

Build the crawled Ganjoor dataset:

```bash
python crawling/creating_dataset.py --chunks-dir chunks --output-dir asr_dataset
```

Chunk long elderly recordings:

```bash
python dataset_tools/chunk_long_recordings.py \
  --input-dir recordings \
  --output-dir chunks \
  --asr-checkpoint AliAvd/qwen3-asr-persian-elderly
```

Build the Hugging Face dataset after reviewing alignments:

```bash
python dataset_tools/build_hf_dataset.py \
  --manifest chunks/manifest.jsonl \
  --review-tsv chunks/review.tsv \
  --output-dir elderly_dataset_hf
```

Run the Qwen API:

```bash
CHECKPOINT_PATH=AliAvd/qwen3-asr-persian-elderly \
uvicorn serving.qwen_api:app --host 0.0.0.0 --port 8080
```

Interactive API documentation is available at `/docs`; upload audio to `POST /analyse`. Set `do_asr=false` for speech boundaries only (the ASR model is not loaded).

Run the Wav2Vec2 API with a standard saved CTC checkpoint containing both model and processor files:

```bash
WAV2VEC2_CHECKPOINT=/path/to/persian-ctc-checkpoint \
uvicorn serving.wav2vec2_api:app --host 0.0.0.0 --port 8000
```

Legacy custom MLP-head checkpoints must first be converted to a compatible model; the server does not reconstruct that historical architecture.

Train a standalone model (JSONL rows need `audio` and `text`):

```bash
python training/qwen3_asr_sft.py \
  --train_file /path/to/train.jsonl --eval_file /path/to/validation.jsonl \
  --output_dir qwen3-asr-finetuning-out --no-augmentation

python training/wav2vec2_ctc.py \
  --train-jsonl /path/to/train.jsonl --validation-jsonl /path/to/validation.jsonl \
  --output-dir outputs/wav2vec2
```

Both trainers save final model and processor files at the requested output directory. For the controlled 30-scenario study, use the experiment pipeline, whose normalization and evaluation policy are documented in [experiments/README.md](experiments/README.md). Standalone evaluator scores are not interchangeable with campaign scores because their normalization policies differ.

## Data and model repositories

- Ganjoor ASR dataset: `AliAvd/persian-asr-dataset`
- Persian elderly ASR dataset: `AliAvd/persian-elderly-asr`
- Fine-tuned Qwen3-ASR model: `AliAvd/qwen3-asr-persian-elderly`

These repositories may be private and require an authenticated Hugging Face account with access.

## Notes

- Confirm speaker consent before making elderly-speech data public.
- Respect the terms of every upstream dataset and crawled source.
- WER/CER results depend on the text-normalization policy and exact test split.
- Do not use unreviewed transcripts for safety-critical applications.

## Authors

Ali Alvandi — undergraduate project supervised by Hossein Sameti.


## Experiment pipeline

`experiments/` includes runners, 60 model configurations covering 30 scenarios, shared training/evaluation code and tests. See [the execution guide](experiments/README.md) for environment setup, data preparation, prerequisite checks, campaign execution and result collection.

`dataset_tools/*final*.py` uses `ASR_DATA_ROOT` for the external corpus root (default: repository root). The CTC aligner accepts `ASR_WAV2VEC2_BASE_CHECKPOINT` for its local checkpoint. `training/qwen3_asr_legacy.py` contains only the reusable historical definitions imported by the experiment augmentation adapter; it is not a second executable trainer.

The final corpus workflow expects `Final_Gathered_Dataset/` under the data root, with speaker subfolders containing `.m4a` recordings and matching `.txt` transcripts. It generates `Final_Gatheres_Dataset_Chunks/source_inventory.json` and the chunked/Hugging Face exports. Those source corpus assets are not included. Use the following sequence after configuring local paths:

```bash
python dataset_tools/build_final_dataset.py
python dataset_tools/validate_final_dataset.py
python dataset_tools/publish_final_dataset.py --prepare-only
```

The last command prepares files locally. Omitting `--prepare-only` uploads to the configured private Hugging Face repository; `verify_final_hub.py` subsequently checks that upload. The older combined publisher requires explicit paths: `python deployment/publish_private_hf.py --data-root /path/to/corpora --checkpoint /path/to/model`.

Run regression tests in the experiment environment (plus the API dependencies for serving tests):

```bash
pip install fastapi python-multipart httpx
python -B -m unittest discover -s experiments/tests -v
```

These tests use tiny CPU fixtures and mocks; they do not validate large-model accuracy or perform Hub publication. The augmentation-assets test skips when its local corpus is unavailable.

Model weights, datasets, checkpoints, generated results, prediction dumps and thesis files remain local.
