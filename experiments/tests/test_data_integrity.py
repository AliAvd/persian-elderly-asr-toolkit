import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared.prepare_data import remove_leakage,group_split
from shared.audit import audit_config

class DataIntegrity(unittest.TestCase):
    def test_group_and_pcm_precedence(self):
        rows=[dict(id='a',source_id='recording1',split='train',pcm_sha256='A'),dict(id='b',source_id='recording1',split='test',pcm_sha256='B'),dict(id='c',source_id='recording2',split='train',pcm_sha256='B')]
        clean,rejected=remove_leakage(rows)
        self.assertEqual({r['id'] for r in clean},{'a','b'})
        self.assertEqual({r['split'] for r in clean},{'test'})
        self.assertEqual(clean[0]['original_split'],'train')
        self.assertEqual(len(rejected),1)
        self.assertEqual(group_split('x'),group_split('x'))
    def test_preflight_rejects_same_recording_different_audio(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);paths=[]
            for i,source in enumerate(['same','different','same']):
                audio=p/f'{i}.wav';audio.touch();manifest=p/f'{i}.jsonl';manifest.write_text(json.dumps(dict(audio=str(audio),text='سلام',pcm_sha256=str(i),source_id=source))+'\n');paths.append(str(manifest))
            with self.assertRaisesRegex(ValueError,'overlapping groups'):
                audit_config({'mode':'train','data':dict(train_manifest=paths[0],validation_manifest=paths[1],tests={'test':paths[2]})})
if __name__=='__main__':unittest.main()
