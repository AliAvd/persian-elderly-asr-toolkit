"""Load heterogeneous provenance JSONL through an explicit training-only schema."""
import hashlib,json
from pathlib import Path

def _training_rows(path, content_sha256):
    # content_sha256 is included in the datasets cache key to invalidate changed files.
    with open(path,encoding='utf-8') as handle:
        for number,line in enumerate(handle,1):
            if not line.strip():continue
            row=json.loads(line)
            if not isinstance(row.get('audio'),str) or not isinstance(row.get('text'),str):
                raise ValueError(f'{path}:{number}: audio and text must be strings')
            yield {'audio':row['audio'],'text':row['text'],
                   'language':row.get('language') or 'Persian','prompt':row.get('prompt') or '',
                   'duration':float(row.get('duration') or 0)}

def load_training_data(data):
    from datasets import Dataset,DatasetDict,Features,Value
    features=Features({name:Value('float64' if name=='duration' else 'string')
                      for name in ['audio','text','language','prompt','duration']})
    result={}
    for split,key in [('train','train_manifest'),('validation','validation_manifest')]:
        path=Path(data[key]).resolve()
        result[split]=Dataset.from_generator(_training_rows,features=features,
            gen_kwargs={'path':str(path),'content_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return DatasetDict(result)
