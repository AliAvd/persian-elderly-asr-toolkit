"""No network or pretrained weights: tiny synthetic CPU integration test."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
import json
import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.evaluation import normalize, metrics, evaluate_manifest
from shared.training import run

class MetricTests(unittest.TestCase):
    def test_digits_languages_and_empty_hypothesis(self):
        self.assertEqual(normalize('من ۱۲ سال دارم، 中国。'), 'من 12 سال دارم 中国')
        result = metrics([{'reference':'a b', 'prediction':''}])
        self.assertEqual(result['wer'], 1.)
        self.assertEqual(result['cer'], 1.)

    def test_failures_are_retained_or_abort(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); manifest=root/'test.jsonl'
            manifest.write_text(json.dumps({'audio':'missing.wav','text':'a b'})+'\n')
            def fail(item): raise OSError('missing audio')
            result=evaluate_manifest(manifest,fail,root,'count','count_as_empty')
            self.assertEqual(result['failures'],1)
            self.assertEqual(result['raw']['wer'],1.)
            self.assertEqual(len((root/'count.predictions.jsonl').read_text().splitlines()),1)
            with self.assertRaises(RuntimeError): evaluate_manifest(manifest,fail,root,'abort')


class QwenCollatorTests(unittest.TestCase):
    def test_augmentation_only_train_and_prefix_mask(self):
        import numpy as np
        import soundfile as sf
        import torch
        from shared.augmentation import legacy_components
        parts = legacy_components()
        class Tokenizer:
            eos_token = '$'
            pad_token_id = 0
        class Processor:
            tokenizer = Tokenizer()
            def __init__(self): self.calls = []
            def __call__(self, text, audio, **kwargs):
                self.calls.append(audio)
                width = max(map(len, text))
                ids = torch.zeros((len(text), width), dtype=torch.long)
                for i, value in enumerate(text): ids[i, :len(value)] = 1
                return {'input_ids': ids, 'attention_mask': ids.ne(0).long()}
        augmented = []
        def augment(samples, sample_rate):
            augmented.append(samples.copy())
            return samples * 2
        processor = Processor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.wav'
            sf.write(path, np.full(1600, .1), 16000)
            collator = parts['DataCollatorForQwen3ASRFinetuning'](processor=processor, augmentor=augment)
            base = {'audio': str(path), 'prefix_text': 'abc', 'target': 'xy'}
            batch = collator([dict(base, apply_augmentation=True), dict(base, apply_augmentation=False)])
        self.assertEqual(len(augmented), 1)
        self.assertTrue(np.allclose(processor.calls[0][0], 2 * processor.calls[0][1]))
        self.assertTrue(torch.all(batch['labels'][:, :3] == -100))
        self.assertTrue(torch.all(batch['labels'][:, 3:] == 1))

class TinyCTCTest(unittest.TestCase):
    def test_train_save_reload_and_multilingual_expansion(self):
        import numpy as np
        import soundfile as sf
        import torch
        from transformers import Wav2Vec2Config,Wav2Vec2ForCTC,Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor,Wav2Vec2Processor
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); initial=root/'initial';initial.mkdir()
            vocab={'ا':0,'ب':1,'|':2,'[UNK]':3,'[PAD]':4}
            (initial/'vocab.json').write_text(json.dumps(vocab,ensure_ascii=False))
            tokenizer=Wav2Vec2CTCTokenizer(str(initial/'vocab.json'),unk_token='[UNK]',pad_token='[PAD]',word_delimiter_token='|',bos_token=None,eos_token=None)
            processor=Wav2Vec2Processor(Wav2Vec2FeatureExtractor(sampling_rate=16000,return_attention_mask=True),tokenizer)
            processor.save_pretrained(initial)
            model=Wav2Vec2ForCTC(Wav2Vec2Config(vocab_size=len(tokenizer),hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,conv_dim=(8,8),conv_stride=(4,4),conv_kernel=(8,4),num_conv_pos_embedding_groups=2,num_conv_pos_embeddings=8,mask_time_prob=0.,mask_feature_prob=0.,pad_token_id=4,ctc_loss_reduction='mean'))
            initial_weights=model.lm_head.weight.detach().clone()
            model.save_pretrained(initial)
            for n in range(2): sf.write(root/f'{n}.wav',np.random.default_rng(n).normal(0,.05,3200).astype('float32'),16000)
            manifests={}
            for split in ('train','validation','test'):
                path=root/f'{split}.jsonl'
                path.write_text(''.join(json.dumps({'audio':str(root/f'{n}.wav'),'text':text,'language':lang},ensure_ascii=False)+'\n' for n,(text,lang) in enumerate([('اب','fa'),('你好','zh')])))
                if split=='train':
                    with path.open('a') as handle:handle.write(json.dumps({'audio':str(root/'0.wav'),'text':'…، ؟','language':'Persian'},ensure_ascii=False)+'\n')
                manifests[split]=str(path)
            cfg={'mode':'train','model':{'family':'wav2vec2_base','pretrained':str(initial),'init_checkpoint':str(initial),'expand_vocabulary':True},'data':{'train_manifest':manifests['train'],'validation_manifest':manifests['validation'],'tests':{'test':manifests['test']}},'augmentation':{'enabled':False},'training':{'output_dir':str(root/'result'),'batch_size':2,'gradient_accumulation':1,'max_steps':1,'epochs':1,'learning_rate':0.,'seed':42}}
            results=run(cfg)
            feasibility=json.loads((root/'result'/'ctc_feasibility.json').read_text())
            self.assertEqual(feasibility['train']['before'],3)
            self.assertEqual(feasibility['train']['kept_for_loss'],2)
            self.assertEqual(feasibility['train']['empty_normalized_targets'],1)
            self.assertEqual(len((root/'result'/'ctc_empty_target_exclusions.jsonl').read_text().splitlines()),1)
            final=root/'result'/'final'
            reloaded=Wav2Vec2ForCTC.from_pretrained(final)
            reloaded_processor=Wav2Vec2Processor.from_pretrained(final)
            self.assertEqual(reloaded.config.vocab_size,len(reloaded_processor.tokenizer))
            self.assertTrue(torch.equal(initial_weights,reloaded.lm_head.weight[:len(vocab)]))
            self.assertIn('你',reloaded_processor.tokenizer.get_vocab())
            self.assertEqual(results['test']['samples'],2)
            self.assertEqual(results['validation']['failures'],0)
            cfg['mode']='evaluate';cfg['model']['init_checkpoint']=str(final)
            again=run(cfg)
            self.assertEqual(results['test']['raw'],again['test']['raw'])

if __name__=='__main__': unittest.main()
