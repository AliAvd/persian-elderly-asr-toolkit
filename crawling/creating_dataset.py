"""Build a local Ganjoor DatasetDict from crawler alignment CSV files."""
import argparse
import csv
from pathlib import Path


def alignment_rows(root):
    files = sorted(Path(root).glob('*/alignment.csv'))
    if not files:
        raise FileNotFoundError(f'No alignment.csv files under {root}')
    for path in files:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                audio = Path(row['audio_path']).expanduser()
                if not audio.is_absolute():
                    # Existing crawler exports include "chunks/..."; portable CSVs may
                    # instead store a filename relative to the CSV itself.
                    candidates = [path.parent / audio, Path.cwd() / audio]
                    audio = next((p for p in candidates if p.is_file()), candidates[0])
                yield {'audio': str(audio.resolve()), 'sentence': row['text'],
                       'speaker_id': row.get('narrator', 'unknown')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunks-dir', type=Path, default=Path('chunks'))
    parser.add_argument('--output-dir', type=Path, default=Path('asr_dataset'))
    args = parser.parse_args()
    import soundfile as sf
    from datasets import Audio, Dataset, DatasetDict
    rows = []; rejected = 0
    for row in alignment_rows(args.chunks_dir):
        try:
            audio, _ = sf.read(row['audio'])
            if not audio.size:
                raise ValueError('Empty audio')
        except (OSError, RuntimeError, ValueError):
            rejected += 1
            continue
        rows.append(row)
    if not rows:
        raise ValueError('No decodable audio found in the alignment files')
    dataset = DatasetDict(train=Dataset.from_list(rows).cast_column('audio', Audio(sampling_rate=16000)))
    dataset.save_to_disk(str(args.output_dir))
    print(f'Saved {len(rows)} rows to {args.output_dir}; rejected {rejected} files')


if __name__ == '__main__':
    main()
