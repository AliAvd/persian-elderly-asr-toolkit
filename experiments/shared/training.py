"""Executable Qwen JSONL and Wav2Vec2 CTC training, with shared final evaluation."""
import json
import os
import hashlib
from pathlib import Path
from .evaluation import LANGUAGES, plain_text, normalize, read_manifest, evaluate_manifest

def run(config, dry_run=False):
    mode = config.get('mode', 'train')
    family = config['model']['family']
    data = config['data']
    required = dict(data.get('tests', {}))
    if mode == 'train':
        required.update(train=data['train_manifest'], validation=data['validation_manifest'])
    missing = [str(p) for p in required.values() if not Path(p).is_file()]
    if missing: raise FileNotFoundError('Prepare required manifests first: ' + ', '.join(missing))
    if family != 'qwen' and mode == 'evaluate' and not config['model'].get('init_checkpoint'):
        raise ValueError('Raw SSL Wav2Vec2 checkpoints have no trained Persian CTC head. Set model.init_checkpoint to a Persian CTC model; zero-shot metrics would be meaningless.')
    if dry_run: return {'status':'configuration_valid', 'model': family, 'manifests': required}
    if mode=='evaluate' and int(os.environ.get('RANK',0))!=0:return {'status':'evaluation_rank_skipped'}
    import torch
    import transformers
    if torch.cuda.is_available(): torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    if torch.cuda.is_available() and os.environ.get('ASR_CUDA_MEMORY_LIMIT_MB'):
        limit=int(os.environ['ASR_CUDA_MEMORY_LIMIT_MB'])*1024**2
        torch.cuda.set_per_process_memory_fraction(min(.95,limit/torch.cuda.get_device_properties(0).total_memory))
    transformers.set_seed(config['training'].get('seed', 42))
    output = Path(config['training']['output_dir']); output.mkdir(parents=True, exist_ok=True)
    (output/'resolved_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2),encoding='utf-8')
    # Hash exact inputs so reported metrics can be traced to immutable manifests.
    hashes={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in required.items()}
    (output/'manifest_hashes.json').write_text(json.dumps(hashes,indent=2))
    if int(os.environ.get('RANK',0))==0:
        import platform,importlib.metadata
        packages={}
        for name in ['torch','transformers','qwen-asr','datasets','audiomentations','accelerate']:
            try:packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:packages[name]=None
        (output/'runtime.json').write_text(json.dumps({'python':platform.python_version(),'packages':packages,'cuda':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},indent=2))
    predict = _qwen(config) if family == 'qwen' else _wav2vec(config)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        if torch.distributed.get_rank()!=0: return {'status':'training_worker_complete'}
    if config.get('evaluation',{}).get('vad'): predict = with_vad(predict, config['evaluation'])
    elif config.get('evaluation',{}).get('window_seconds'):predict=with_fixed_windows(predict,config['evaluation']['window_seconds'])
    predict = cached_predictor(predict)
    suites = dict(data.get('tests', {}))
    if data.get('validation_manifest'): suites = {'validation':data['validation_manifest'], **suites}
    if config.get('evaluation',{}).get('report_train_metrics') and data.get('train_manifest'): suites['train']=data['train_manifest']
    results = {name:evaluate_manifest(path,predict,output/'evaluation',name,config.get('failure_policy','abort')) for name,path in suites.items()}
    (output/'metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))
    return results

def _arguments(config, qwen=False):
    import torch
    from transformers import TrainingArguments
    t = config['training']; cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    return TrainingArguments(output_dir=t['output_dir'],num_train_epochs=t.get('epochs',3),
        per_device_train_batch_size=t.get('batch_size',1),per_device_eval_batch_size=t.get('eval_batch_size',1),
        gradient_accumulation_steps=t.get('gradient_accumulation',8),learning_rate=t.get('learning_rate',2e-5),
        warmup_ratio=t.get('warmup_ratio',.02),logging_steps=t.get('logging_steps',10),
        eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,metric_for_best_model='eval_loss',
        greater_is_better=False,save_total_limit=t.get('save_total_limit',2),
        save_only_model=t.get('save_only_model',False),
        bf16=bf16,fp16=cuda and not bf16,remove_unused_columns=False,
        dataloader_num_workers=t.get('num_workers',0),report_to='none',seed=t.get('seed',42),
        gradient_checkpointing=t.get('gradient_checkpointing',False),
        gradient_checkpointing_kwargs={'use_reentrant':False} if t.get('gradient_checkpointing',False) else None,
        max_steps=t.get('max_steps',-1),prediction_loss_only=True)

def _qwen(config):
    import torch
    from .dataset_loading import load_training_data
    from qwen_asr import Qwen3ASRModel
    from .augmentation import legacy_components, build_augmentor
    modelcfg = config['model']; checkpoint = modelcfg.get('init_checkpoint') or modelcfg['pretrained']
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
    wrapper = Qwen3ASRModel.from_pretrained(checkpoint,dtype=dtype,device_map=None)
    if config.get('mode','train') == 'train':
        parts = legacy_components(); processor = wrapper.processor
        processor.tokenizer.padding_side = 'right'
        parts['patch_outer_forward'](wrapper.model)
        from transformers import GenerationConfig
        wrapper.model.generation_config = GenerationConfig.from_model_config(wrapper.model.config)
        data = config['data']; ds = load_training_data(data)
        preprocess = parts['make_preprocess_fn_prefix_only'](processor)
        def prepare(item):
            language = LANGUAGES.get(item.get('language','fa').lower(),item.get('language','Persian'))
            item['text'] = f'language {language}<asr_text>{plain_text(item["text"])}'
            return preprocess(item)
        for split in ds:
            ds[split] = ds[split].map(prepare,remove_columns=ds[split].column_names)
            ds[split] = ds[split].add_column('apply_augmentation',[split=='train']*len(ds[split]))
        collator = parts['DataCollatorForQwen3ASRFinetuning'](processor=processor,augmentor=build_augmentor(config.get('augmentation',{}),parts))
        trainer = parts['CastFloatInputsTrainer'](model=wrapper.model,args=_arguments(config,True),train_dataset=ds['train'],eval_dataset=ds['validation'],data_collator=collator)
        result=trainer.train(resume_from_checkpoint=config['training'].get('resume_from_checkpoint'))
        final = Path(config['training']['output_dir'])/'final'
        trainer.save_model(str(final)); trainer.save_state()
        if trainer.is_world_process_zero(): processor.save_pretrained(str(final))
        trainer.save_metrics('train',result.metrics)
    wrapper.model.to(f'cuda:{int(os.environ.get("LOCAL_RANK",0))}' if torch.cuda.is_available() else 'cpu'); wrapper.model.eval()
    def predict(item):
        language = LANGUAGES.get(item.get('language','fa').lower(),item.get('language','Persian'))
        with torch.inference_mode():
            return wrapper.transcribe(audio=item['audio'],language=language)[0].text
    return predict

def _wav2vec(config):
    import torch, librosa
    from .dataset_loading import load_training_data
    from transformers import Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor,Wav2Vec2Processor,Wav2Vec2ForCTC,Trainer
    from .augmentation import build_augmentor
    mc=config['model']; checkpoint=mc.get('init_checkpoint'); training=config.get('mode','train')=='train'
    out=Path(config['training']['output_dir'])
    ds=None
    if training:
        ds=load_training_data(config['data'])
        characters=set(''.join(normalize(x) for x in ds['train']['text']))-{' '}
    if checkpoint:
        processor=Wav2Vec2Processor.from_pretrained(checkpoint)
        model=Wav2Vec2ForCTC.from_pretrained(checkpoint)
        contract=restore_ctc_special_tokens(processor,model)
        (out/'processor_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2))
        if training:
            missing=characters-set(processor.tokenizer.get_vocab())
            if missing:
                if not mc.get('expand_vocabulary',False):
                    raise ValueError('CTC training contains new characters; set model.expand_vocabulary=true to append new symbols while preserving the learned output rows')
                old=model.lm_head
                if not isinstance(old,torch.nn.Linear): raise ValueError('Vocabulary expansion supports the standard linear CTC head only')
                processor.tokenizer.add_tokens(sorted(missing))
                new=torch.nn.Linear(old.in_features,len(processor.tokenizer),bias=old.bias is not None).to(old.weight.device,old.weight.dtype)
                with torch.no_grad():
                    new.weight[:old.out_features].copy_(old.weight)
                    if old.bias is not None:new.bias[:old.out_features].copy_(old.bias)
                model.lm_head=new;model.config.vocab_size=len(processor.tokenizer)
                (out/'vocabulary_expansion.json').write_text(json.dumps({'added':sorted(missing),'old_size':old.out_features,'new_size':len(processor.tokenizer)},ensure_ascii=False))
    else:
        if not training: raise ValueError('Persian CTC checkpoint required')
        # Optional additional TRAIN manifests predeclare Chinese characters across matched arms.
        for manifest in mc.get('vocabulary_manifests',[]):
            characters.update(''.join(normalize(x['text']) for x in read_manifest(manifest)))
        characters.discard(' '); characters.discard('|')
        vocab={c:i for i,c in enumerate(sorted(characters))}
        for token in ('|','[UNK]','[PAD]'): vocab[token]=len(vocab)
        vocabpath=out/'vocab.json'; vocabpath.write_text(json.dumps(vocab,ensure_ascii=False),encoding='utf-8')
        tokenizer=Wav2Vec2CTCTokenizer(str(vocabpath),unk_token='[UNK]',pad_token='[PAD]',word_delimiter_token='|',do_lower_case=False)
        extractor=Wav2Vec2FeatureExtractor(feature_size=1,sampling_rate=16000,padding_value=0.,do_normalize=True,return_attention_mask=True)
        processor=Wav2Vec2Processor(feature_extractor=extractor,tokenizer=tokenizer)
        model=Wav2Vec2ForCTC.from_pretrained(mc['pretrained'],vocab_size=len(tokenizer),pad_token_id=tokenizer.pad_token_id,ctc_loss_reduction='mean',ctc_zero_infinity=False,ignore_mismatched_sizes=True)
    if training:
        # CTC needs enough acoustic frames for labels plus consecutive repeats.
        # Keep final validation/test manifests intact; filter only loss datasets.
        def frame_count(n):
            for kernel,stride in zip(model.config.conv_kernel,model.config.conv_stride):
                n=(n-kernel)//stride+1
            return n
        def required_frames(text):
            ids=processor.tokenizer(normalize(text)).input_ids
            labels_needed=len(ids)+sum(a==b for a,b in zip(ids,ids[1:]))
            masking_needed=model.config.mask_time_length if model.config.mask_time_prob>0 else 0
            return max(labels_needed,masking_needed)
        feasibility={}
        exclusions=[]
        for split in ds:
            before=len(ds[split])
            def feasible(item):
                import soundfile as sf
                if not normalize(item['text']):
                    exclusions.append({'split':split,'audio':item['audio'],'text':item['text'],'reason':'empty after text normalization'})
                    return False
                samples=round(item['duration']*16000) if item.get('duration') else round(sf.info(item['audio']).duration*16000)
                return required_frames(item['text'])<=frame_count(samples)
            ds[split]=ds[split].filter(feasible,load_from_cache_file=False,desc=f'CTC alignment feasibility {split}')
            feasibility[split]={'before':before,'kept_for_loss':len(ds[split]),'excluded_from_loss':before-len(ds[split]),'empty_normalized_targets':sum(r['split']==split for r in exclusions)}
            if not len(ds[split]):raise ValueError(f'No CTC-alignable {split} records')
        (out/'ctc_feasibility.json').write_text(json.dumps(feasibility,indent=2))
        (out/'ctc_empty_target_exclusions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in exclusions),encoding='utf-8')
        aug=build_augmentor(config.get('augmentation',{}))
        for split in ds:
            ds[split]=ds[split].add_column('apply_augmentation',[split=='train']*len(ds[split]))
        def collate(items):
            audios=[]
            for item in items:
                audio,_=librosa.load(item['audio'],sr=16000,mono=True)
                if aug is not None and item['apply_augmentation']:
                    augmented=aug(samples=audio.copy(),sample_rate=16000)
                    if required_frames(item['text'])<=frame_count(len(augmented)):audio=augmented
                audios.append(audio)
            batch=processor(audios,sampling_rate=16000,padding=True,return_tensors='pt')
            targets=[normalize(i['text']) for i in items]
            if any(not t for t in targets):raise ValueError('Empty normalized CTC target reached collator: '+str([i['audio'] for i,t in zip(items,targets) if not t]))
            labels=processor.tokenizer(targets,padding=True,return_tensors='pt')
            batch['labels']=labels.input_ids.masked_fill(labels.attention_mask.ne(1),-100)
            return batch
        if config['training'].get('freeze_feature_encoder',True): model.freeze_feature_encoder()
        trainer=Trainer(model=model,args=_arguments(config),data_collator=collate,train_dataset=ds['train'],eval_dataset=ds['validation'])
        result=trainer.train(resume_from_checkpoint=config['training'].get('resume_from_checkpoint'))
        trainer.save_model(str(out/'final')); trainer.save_state(); trainer.save_metrics('train',result.metrics)
        if trainer.is_world_process_zero(): processor.save_pretrained(str(out/'final'))
    device=f'cuda:{int(os.environ.get("LOCAL_RANK",0))}' if torch.cuda.is_available() else 'cpu'; model.to(device); model.eval()
    def predict(item):
        audio,_=librosa.load(item['audio'],sr=16000,mono=True)
        inputs=processor(audio,sampling_rate=16000,return_tensors='pt',padding=True).to(device)
        with torch.inference_mode(): ids=model(**inputs).logits.argmax(dim=-1)
        return processor.batch_decode(ids)[0]
    return predict


def with_vad(predict, settings):
    """Evaluate complete held-out recordings; omitted speech remains a deletion."""
    import librosa, soundfile as sf, tempfile
    from silero_vad import load_silero_vad,get_speech_timestamps
    vad=load_silero_vad()
    def wrapped(item):
        audio,_=librosa.load(item['audio'],sr=16000,mono=True)
        regions=get_speech_timestamps(audio,vad,sampling_rate=16000,speech_pad_ms=settings.get('speech_pad_ms',60),threshold=settings.get('vad_threshold',.5),max_speech_duration_s=20)
        hypotheses=[]
        with tempfile.TemporaryDirectory(prefix='asr_eval_vad_') as directory:
            for i,span in enumerate(regions):
                path=Path(directory)/f'{i}.wav';sf.write(path,audio[span['start']:span['end']],16000)
                hypotheses.append(predict(dict(item,audio=str(path))))
        return ' '.join(hypotheses)
    return wrapped


def with_fixed_windows(predict, seconds=20):
    """No-VAD long-recording baseline with bounded memory and no omitted samples."""
    import soundfile as sf,tempfile
    if seconds<=0:raise ValueError('window_seconds must be positive')
    def wrapped(item):
        audio,sr=sf.read(item['audio'],dtype='float32')
        width=round(seconds*sr);hypotheses=[]
        with tempfile.TemporaryDirectory(prefix='asr_fixed_window_') as directory:
            for start in range(0,len(audio),width):
                path=Path(directory)/f'{start}.wav';sf.write(path,audio[start:start+width],sr)
                hypotheses.append(predict(dict(item,audio=str(path))))
        return ' '.join(hypotheses)
    return wrapped


def restore_ctc_special_tokens(processor, model):
    """Honor the checkpoint's CTC blank ID even when tokenizer metadata is absent."""
    tokenizer=processor.tokenizer;blank=model.config.pad_token_id
    before=tokenizer.pad_token_id
    if before!=blank:
        token=tokenizer.convert_ids_to_tokens(blank)
        if token is None:raise ValueError(f'Checkpoint CTC blank ID {blank} has no tokenizer entry')
        tokenizer.pad_token=token
    vocab=tokenizer.get_vocab()
    if '[UNK]' in vocab and tokenizer.unk_token not in {'[UNK]'} and tokenizer.unk_token_id>=model.config.vocab_size:
        tokenizer.unk_token='[UNK]'
    if tokenizer.pad_token_id!=blank:raise ValueError('CTC decoder blank does not match checkpoint')
    return {'checkpoint_blank_id':blank,'loaded_tokenizer_pad_id':before,'effective_tokenizer_pad_id':tokenizer.pad_token_id,'effective_pad_token':tokenizer.pad_token,'effective_unk_token':tokenizer.unk_token,'corrected':before!=blank}


def cached_predictor(predict):
    """Reuse deterministic transcriptions shared by pooled and per-source suites."""
    cache={}
    def wrapped(item):
        key=(item['audio'],item.get('language','Persian'))
        if key in cache:
            wrapped.cache_hits+=1
            return cache[key]
        answer=predict(item);cache[key]=answer;return answer
    wrapped.cache_hits=0
    return wrapped
