import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared.dataset_loading import load_training_data

class MixedProvenanceSchema(unittest.TestCase):
    def test_mixed_sources_and_splits_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);train=root/'train.jsonl';val=root/'validation.jsonl'
            rows=[{'audio':'a.wav','text':'سلام','speaker_id':'a','age':'twenties'},
                  {'audio':'b.wav','text':'你好','language':'Chinese','alignment_method':{'name':'x'},'duration':2.5}]
            train.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            val.write_text(json.dumps({'audio':'c.wav','text':'درود','source_id':'c','extra':[1,2]})+'\n')
            data={'train_manifest':str(train),'validation_manifest':str(val)}
            ds=load_training_data(data)
            self.assertEqual(ds['train'].features,ds['validation'].features)
            self.assertEqual(ds['train'][1]['text'],'你好')
            self.assertEqual(ds['train'][0]['language'],'Persian')
            self.assertEqual(ds['train'][1]['duration'],2.5)
            rows[0]['text']='تغییر';train.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            self.assertEqual(load_training_data(data)['train'][0]['text'],'تغییر')
if __name__=='__main__':unittest.main()
