"""Exercise the real project augmentor, including its optional MP3 backend."""
import json,random,sys,unittest,warnings
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared.augmentation import build_augmentor

class RealAugmentation(unittest.TestCase):
    def test_seeded_training_assets(self):
        root=Path(__file__).resolve().parents[1]
        source=root/'data/manifests/elderly/train.jsonl'
        if not source.exists():self.skipTest('Prepare elderly data and training augmentation assets first')
        with source.open() as f:row=json.loads(f.readline())
        audio,sr=sf.read(row['audio'],dtype='float32')
        aug=build_augmentor({'enabled':True,'probability':1,'background_dir':str(root/'data/augmentation/background_train'),'rir_dir':str(root/'data/augmentation/rir_train')})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            for seed in range(20):
                random.seed(seed);np.random.seed(seed)
                result=aug(samples=audio.copy(),sample_rate=sr)
                self.assertGreater(result.size,0);self.assertEqual(result.ndim,1);self.assertTrue(np.isfinite(result).all())
if __name__=='__main__':unittest.main()
