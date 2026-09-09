import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared.training import restore_ctc_special_tokens

class LegacyCTCBlank(unittest.TestCase):
    def test_missing_metadata_does_not_emit_literal_pad(self):
        from transformers import Wav2Vec2CTCTokenizer
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'vocab.json';p.write_text(json.dumps({'ا':0,'|':1,'[UNK]':2,'[PAD]':3},ensure_ascii=False))
            tok=Wav2Vec2CTCTokenizer(str(p))
            self.assertNotEqual(tok.pad_token_id,3)
            report=restore_ctc_special_tokens(SimpleNamespace(tokenizer=tok),SimpleNamespace(config=SimpleNamespace(pad_token_id=3,vocab_size=4)))
            self.assertTrue(report['corrected']);self.assertEqual(tok.decode([0,3,0]),'اا')
            self.assertEqual(tok.pad_token_id,3)
if __name__=='__main__':unittest.main()
