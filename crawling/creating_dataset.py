import glob
import os

from datasets import Audio, load_dataset

ROOT_DIR = "chunks"  # Directory containing subfolders with alignment.csv
SAMPLE_RATE = 16_000

# --------------------------------------------------
# 1. Find all alignment.csv files
# --------------------------------------------------
csv_files = glob.glob(os.path.join(ROOT_DIR, "*", "alignment.csv"))
assert len(csv_files) > 0, "No alignment.csv files found!"

print(f"Found {len(csv_files)} CSV files")

# --------------------------------------------------
# 2. Load all CSVs into one dataset
# --------------------------------------------------
dataset = load_dataset(
    "csv",
    data_files={"train": csv_files},
    keep_default_na=False,
)


# --------------------------------------------------
# 3. Convert audio_path to ABSOLUTE paths
#    Each audio_path in CSV is assumed to be relative
#    to the directory containing its alignment.csv
# --------------------------------------------------
def make_absolute(example):
    audio_path = example["audio_path"]
    # print(audio_path)

    # Remove leading slash if it exists (to avoid treating
    # a relative path like "/file.mp3" as filesystem root)
    audio_path = audio_path.removeprefix("/")
    # print(audio_path)

    # Convert to absolute path
    example["audio_path"] = os.path.abspath(audio_path)
    # print(example["audio_path"])

    return example


dataset = dataset.map(make_absolute)
print(dataset["train"]["audio_path"])

# --------------------------------------------------
# 4. Cast to Audio feature
#    This stores:
#    {
#        "bytes": None,
#        "path": "/absolute/path/to/file.mp3"
#    }
# --------------------------------------------------
dataset = dataset.cast_column("audio_path", Audio(sampling_rate=SAMPLE_RATE))

# --------------------------------------------------
# 5. Rename columns (Common Voice style)
# --------------------------------------------------
rename_map = {
    "audio_path": "audio",
    "text": "sentence",
}

# Rename narrator -> speaker_id only if it exists
if "narrator" in dataset["train"].column_names:
    rename_map["narrator"] = "speaker_id"

dataset = dataset.rename_columns(rename_map)


# --------------------------------------------------
# 6. Remove broken audio files
# --------------------------------------------------
def is_valid(example):
    try:
        # Accessing the array forces decoding
        _ = example["audio"]["array"]
        return True
    except Exception:
        return False


dataset = dataset.filter(is_valid)

# --------------------------------------------------
# 7. Save dataset
# --------------------------------------------------
print(dataset)
print(dataset["train"][0])

# Example output:
# {
#   'audio': {
#       'bytes': None,
#       'path': '/absolute/path/to/0000101024.mp3'
#   },
#   'sentence': 'از کجا فهمیدی من اومدم؟'
# }

dataset.save_to_disk("asr_dataset")

print("Dataset saved successfully!")
