"""Regression checks for a clean checkout; no downloads or production training."""
import csv
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments'))


class CampaignTests(unittest.TestCase):
    def test_prepare_preserves_external_baselines_and_relocates_dependencies(self):
        import campaign
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / 'external-model'
            with patch.dict(os.environ, {'ASR_WAV2VEC2_BASE_CHECKPOINT': str(baseline)}):
                campaign.prepare(root / 'campaign')
            jobs = json.loads((root / 'campaign/jobs.json').read_text())
            self.assertEqual(len(jobs), 61)
            self.assertEqual(len({j['id'] for j in jobs}), 61)
            def config(name):
                return json.loads((root / 'campaign/configs' / (name + '.json')).read_text())
            self.assertEqual(config('01_baseline_public/wav2vec2_base')['model']['init_checkpoint'], str(baseline))
            for name, source in [('04_elderly_only_adaptation', '03_public'), ('29_long_recording_vad', '07_public_elderly_aug')]:
                self.assertEqual(config(name + '/wav2vec2_base')['model']['init_checkpoint'], str(root / 'campaign/outputs' / source / 'wav2vec2_base/final'))
            self.assertEqual(campaign.PYTHON, Path(sys.executable))
            campaign.prepare(root / 'campaign')
            self.assertEqual(json.loads((root / 'campaign/jobs.json').read_text()), jobs)

    def test_catalog_regenerates_without_drift_or_duplicate_entries(self):
        import generate_catalog
        with tempfile.TemporaryDirectory() as directory, patch.object(generate_catalog, 'ROOT', Path(directory)):
            for _ in range(2):
                generate_catalog.main()
                for path in Path(directory).rglob('*.json'):
                    checked_in = REPO / 'experiments' / path.relative_to(directory)
                    self.assertEqual(json.loads(path.read_text()), json.loads(checked_in.read_text()), str(path))


class CorpusTests(unittest.TestCase):
    def test_absolute_and_csv_relative_audio_paths(self):
        from crawling.creating_dataset import alignment_rows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); chunk = root / 'poem'; chunk.mkdir()
            audio = chunk / 'part.wav'; audio.touch()
            with (chunk / 'alignment.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['audio_path', 'text', 'narrator'])
                writer.writeheader()
                for value in [str(audio), audio.name]:
                    writer.writerow(dict(audio_path=value, text='متن', narrator='speaker'))
            self.assertEqual([row['audio'] for row in alignment_rows(root)], [str(audio)] * 2)

    def test_portable_publication_with_no_rejected_rows(self):
        from dataset_tools import publish_final_dataset as publish
        with tempfile.TemporaryDirectory() as directory, patch.object(publish, 'HF', Path(directory)), patch.object(publish, 'load_from_disk', return_value={}):
            (Path(directory) / 'rejected.jsonl').write_text('')
            portable = publish.prepare_portable()
            self.assertEqual((portable / 'review/manifest.jsonl').read_text(), '')


class StandaloneTrainingTests(unittest.TestCase):
    def test_qwen_without_validation_or_augmentation_on_cpu_saves_final(self):
        from training import qwen3_asr_sft as qwen
        args = ['qwen', '--train_file', 'unused.jsonl', '--no-augmentation', '--num_workers', '0', '--output_dir', '/tmp/unused-qwen-review']
        dataset = MagicMock(); dataset.column_names = []
        dataset.add_column.return_value = dataset
        raw = MagicMock(); raw.map.return_value = {'train': dataset}
        wrapper = MagicMock(); trainer = MagicMock()
        with patch.object(sys, 'argv', args), patch.object(qwen.torch.cuda, 'is_available', return_value=False), patch.object(qwen.Qwen3ASRModel, 'from_pretrained', return_value=wrapper), patch.object(qwen, 'load_dataset', return_value=raw), patch.object(qwen, 'patch_outer_forward'), patch.object(qwen.GenerationConfig, 'from_model_config'), patch.object(qwen, 'TrainingArguments') as arguments, patch.object(qwen, 'CastFloatInputsTrainer', return_value=trainer) as factory:
            qwen.main()
        self.assertEqual(arguments.call_args.kwargs['eval_strategy'], 'no')
        self.assertFalse(arguments.call_args.kwargs['fp16'])
        self.assertFalse(arguments.call_args.kwargs['dataloader_persistent_workers'])
        self.assertIsNone(factory.call_args.kwargs['eval_dataset'])
        trainer.save_model.assert_called_once_with('/tmp/unused-qwen-review')
        wrapper.processor.save_pretrained.assert_called_once_with('/tmp/unused-qwen-review')

    def test_ctc_single_word_vocabulary(self):
        from training.wav2vec2_ctc import build_processor
        with tempfile.TemporaryDirectory() as directory:
            processor = build_processor({'sentence': ['سلام']}, Path(directory), 16000)
            self.assertIn('|', processor.tokenizer.get_vocab())
            self.assertNotEqual(processor.tokenizer.pad_token_id, processor.tokenizer.unk_token_id)


class ServingTests(unittest.TestCase):
    def test_vad_only_does_not_load_asr_model(self):
        import io
        import numpy as np
        import soundfile as sf
        from fastapi.testclient import TestClient
        for name, loader in [('qwen_api', 'get_model'), ('wav2vec2_api', 'get_pipeline')]:
            module = importlib.import_module('serving.' + name)
            wave = io.BytesIO(); sf.write(wave, np.ones(1600, dtype='float32') * .1, 16000, format='WAV')
            with patch.object(module, 'get_vad'), patch.object(module, loader) as model, patch.object(module, 'get_speech_timestamps', return_value=[{'start': 0, 'end': 1600}]):
                response = TestClient(module.app).post('/analyse', data={'do_asr': 'false'}, files={'file': ('test.wav', wave.getvalue(), 'audio/wav')})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(len(response.json()), 1)
            model.assert_not_called()

    def test_wav2vec_uses_one_checkpoint_for_model_and_processor(self):
        from serving import wav2vec2_api as server
        server.get_pipeline.cache_clear()
        try:
            with patch.dict(os.environ, {'WAV2VEC2_CHECKPOINT': '/tmp/saved-model'}), patch('transformers.Wav2Vec2Processor.from_pretrained') as processor, patch('transformers.Wav2Vec2ForCTC.from_pretrained') as model, patch('transformers.pipeline') as pipeline:
                self.assertIs(server.get_pipeline(), pipeline.return_value)
            processor.assert_called_once_with('/tmp/saved-model')
            model.assert_called_once_with('/tmp/saved-model')
        finally:
            server.get_pipeline.cache_clear()


if __name__ == '__main__':
    unittest.main()
