import sys,tempfile,unittest
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared.training import with_fixed_windows

class LongRecording(unittest.TestCase):
    def test_all_samples_including_tail_are_consumed(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'long.wav';sf.write(p,np.zeros(40000,dtype='float32'),16000)
            sizes=[]
            def predict(item):
                sizes.append(sf.info(item['audio']).frames)
                self.assertEqual(item['text'],'complete reference')
                return 'word'
            result=with_fixed_windows(predict,seconds=1)({'audio':str(p),'text':'complete reference'})
            self.assertEqual(sizes,[16000,16000,8000]);self.assertEqual(result,'word word word')
if __name__=='__main__':unittest.main()
