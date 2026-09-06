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
- `jsonl_filtered_converting.py`: ادغام Common Voice، Filimo و داده گنجور، نرمال‌سازی متن و پالایش معلم بر اساس WER برای تولید JSONL مناسب Qwen3-ASR.

### `text_preprocessor/`

- `__init__.py`: رابط اصلی `TextPreprocessor` و زنجیره استانداردسازی فارسی.
- `arabic_alphabet.py`: جدول‌ها و قواعد نویسه‌های عربی و صورت‌های جایگزین.
- `persian_alphabet.py`: تعریف نویسه‌ها و علائم فارسی.
- `reform.py`: اصلاح و یکسان‌سازی نویسه‌ها و نشانه‌های متن.

### `training/`

- `qwen3_asr_sft.py`: تنظیم دقیق Qwen3-ASR، شامل Dataset/Collator، داده‌افزایی احتمالی صوت، تزریق نویز، speed perturbation، افت کیفیت و درج مکث.
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

وزن مدل‌ها و checkpointهای آموزش عمداً در این مخزن قرار نگرفته‌اند. مدل Qwen تنظیم‌دقیق‌شده از مخزن خصوصی Hugging Face پروژه قابل دریافت است.

## Installation

Python 3.10 or newer and FFmpeg are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Common workflows

Build the crawled Ganjoor dataset:

```bash
python crawling/creating_dataset.py
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

Interactive API documentation is available at `/docs`.

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
