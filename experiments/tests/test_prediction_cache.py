import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared.training import cached_predictor
from shared.evaluation import evaluate_manifest

class ReusedPredictions(unittest.TestCase):
    def test_metrics_identical_but_cached_suite_has_no_speed_claim(self):
        calls=[]
        def predict(row):calls.append(row['audio']);return 'سلام'
        cached=cached_predictor(predict)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.jsonl';p.write_text(json.dumps({'audio':'one.wav','language':'Persian','text':'سلام','duration':1},ensure_ascii=False)+'\n')
            first=evaluate_manifest(p,cached,d,'pooled');second=evaluate_manifest(p,cached,d,'source')
            self.assertEqual(calls,['one.wav']);self.assertEqual(first['normalized'],second['normalized']);self.assertEqual(second['reused_predictions'],1);self.assertIsNone(second['real_time_factor'])
if __name__=='__main__':unittest.main()
