"""Evaluate an explicitly selected NPR checkpoint and labeled CSV/folders on CPU."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from tqdm.auto import tqdm
from inference import ROOT, CHECKPOINT, PREPROCESS_CONFIG, load_model, sha256

FIELDS = ['dataset','image_path','true_label','predicted_label','raw_logit','fake_score',
          'inference_time_ms','image_id','model_version','checkpoint_sha256']


def image_id(dataset, image_path):
    import hashlib
    return 'img-' + hashlib.sha256((str(dataset)+'\0'+str(image_path).replace('\\','/')).encode()).hexdigest()


def normalize_manifest(frame, image_root=ROOT, verify_files=True):
    """Never infer numeric labels from scores or arbitrary folder ordering."""
    required={'dataset','image_path','true_label'}
    if not required <= set(frame):
        raise ValueError(f'Manifest missing columns: {sorted(required-set(frame))}')
    frame=frame.copy()
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError('Manifest must have nonempty dataset, image_path and true_label')
    if not frame.true_label.isin([0,1]).all():
        raise ValueError('true_label must be Real=0 or Fake=1')
    frame['true_label']=frame.true_label.astype(int)
    for field in ['eligible','valid']:
        if field in frame:
            values=frame[field].astype(str).str.lower().map({'true':True,'false':False,'1':True,'0':False})
            if values.isna().any():raise ValueError(f'Invalid boolean column: {field}')
            frame[field]=values
    if 'eligible' not in frame:frame['eligible']=True
    if 'valid' in frame:frame['eligible'] &= frame.valid
    if frame.image_path.duplicated().any():raise ValueError('Duplicate image_path in manifest')
    if 'image_id' not in frame:
        frame['image_id']=[image_id(d,p) for d,p in zip(frame.dataset,frame.image_path)]
    if frame.image_id.isna().any() or frame.image_id.duplicated().any():
        raise ValueError('image_id must be present and unique; retain original IDs across separate JPEG-condition manifests')
    ImageFile.LOAD_TRUNCATED_IMAGES=False
    actual_hashes=[]
    if verify_files:
        for row in tqdm(list(frame.itertuples()),desc='Verify manifest files',mininterval=3):
            if not row.eligible:
                actual_hashes.append(getattr(row,'sha256',''));continue
            path=Path(row.image_path)
            path=path if path.is_absolute() else Path(image_root)/path
            digest=sha256(path)
            old_hash=getattr(row,'sha256',None)
            if pd.notna(old_hash) and old_hash and old_hash!=digest:
                raise ValueError(f'Image SHA256 changed: {row.image_path}')
            with Image.open(path) as im:im.verify()
            with Image.open(path) as im:im.convert('RGB').load()
            actual_hashes.append(digest)
        frame['sha256']=actual_hashes
    elif 'sha256' not in frame:
        raise ValueError('sha256 is required when reusing an already audited DataFrame')
    hashes=frame[frame.eligible].groupby('sha256').true_label.nunique()
    if (hashes>1).any():raise ValueError('The same file bytes have conflicting labels')
    return frame


def folders_manifest(specs):
    from evaluation import LABELS, EXTENSIONS
    rows=[]
    for spec in specs:
        name,sep,folder=spec.partition('=')
        if not sep or not name or not folder:raise ValueError('--data-root must be SOURCE=PATH')
        root=Path(folder).resolve()
        if not root.is_dir():raise FileNotFoundError(root)
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:continue
            relative=path.relative_to(root)
            labels={LABELS[p.lower()] for p in relative.parts[:-1] if p.lower() in LABELS}
            if len(labels)!=1:raise ValueError(f'Unknown or conflicting folder label: {path}')
            # Absolute paths also work for datasets outside the project.
            logical=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
            rows.append(dict(dataset=name,image_path=logical,true_label=next(iter(labels))))
    return pd.DataFrame(rows)


def run_evaluation(manifest, detector=None, checkpoint=CHECKPOINT, output=ROOT/'results',
                   image_root=ROOT, batch_size=16, threads=None, verify_files=True):
    from evaluation import fingerprint, compute_metrics, write_json, RESULTS
    if batch_size<1:raise ValueError('batch_size must be positive')
    detector=detector or load_model(Path(checkpoint).resolve(),threads or 4)
    if threads is not None and detector.threads!=threads:
        raise ValueError('Requested threads differ from the supplied detector')
    torch.set_num_threads(detector.threads)
    random.seed(100);np.random.seed(100);torch.manual_seed(100)
    torch.use_deterministic_algorithms(True)
    output=Path(output).resolve();image_root=Path(image_root).resolve()
    frame=normalize_manifest(manifest,image_root,verify_files)
    run_id=fingerprint(frame,detector.checkpoint_sha256,batch_size,detector.threads)
    pred_path=output/'per_image_predictions.csv'
    is_baseline=output==RESULTS.resolve()
    config_path=ROOT/'experiment_config.json' if is_baseline else output/'experiment_config.json'
    done=set(); cached=None
    if pred_path.exists():
        if not config_path.exists():raise ValueError('Existing predictions have no experiment_config.json')
        old=json.loads(config_path.read_text(encoding='utf-8'))
        if old.get('run_fingerprint')!=run_id:
            raise ValueError('Cache mismatch. Choose a NEW --output for a different checkpoint, manifest, batch size or protocol.')
        if Path(old.get('image_root',str(ROOT))).resolve()!=image_root:
            raise ValueError('Cache image_root differs from this run')
        cached=pd.read_csv(pred_path,float_precision='round_trip')
        if set(cached.columns)!=set(FIELDS) or cached.image_path.duplicated().any():
            raise ValueError('Invalid cached schema or duplicate prediction paths')
        expected=frame[frame.eligible].set_index('image_path')
        for row in cached.itertuples():
            if row.image_path not in expected.index:raise ValueError('Unexpected cached image')
            original=expected.loc[row.image_path]
            if (row.dataset!=original.dataset or row.true_label!=original.true_label or row.image_id!=original.image_id
                or row.model_version!=detector.model_version or row.checkpoint_sha256!=detector.checkpoint_sha256):
                raise ValueError('Cached identity, label or model version mismatch')
        if not np.isfinite(cached[['raw_logit','fake_score','inference_time_ms']]).all().all():
            raise ValueError('Truncated or invalid cached numeric row; restore the last complete CSV before resuming')
        if not cached.fake_score.between(0,1).all() or not ((cached.fake_score>.5).astype(int)==cached.predicted_label).all():
            raise ValueError('Invalid cached score or predicted label')
        done=set(cached.image_path)
    output.mkdir(parents=True,exist_ok=True)
    old_config=json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    versions={n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','pandas','scikit-learn','matplotlib','Pillow','tqdm','nbformat','nbclient','ipykernel']}
    config=dict(old_config,seed=100,device='cpu',batch_size=batch_size,threads=detector.threads,threshold=.5,
                threshold_operator='>',positive_label='Fake=1',negative_label='Real=0',
                checkpoint=str(detector.checkpoint),checkpoint_sha256=detector.checkpoint_sha256,
                model_version=detector.model_version,checkpoint_format=detector.checkpoint_format,
                architecture='networks.resnet.resnet50(num_classes=1)',parameters=detector.parameter_count,
                strict_load=True,state_entries=detector.state_entries,preprocessing=PREPROCESS_CONFIG,
                run_fingerprint=run_id,image_root=str(image_root),python=sys.version,packages=versions,
                source_provenance='Not Verified',scope='all eligible rows of the selected manifest; duplicates retained',
                last_resumed_utc=datetime.now(timezone.utc).isoformat(),
                timing='read/decode + preprocess + CPU forward + sigmoid; batch elapsed divided equally among images; excludes audit and plots')
    write_json(config_path,config)
    if is_baseline:
        write_json(output/'person1_experiment_config.json',config)
        (ROOT/'requirements_person1.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n',encoding='utf-8')
    frame.to_csv(output/'evaluation_manifest.csv',index=False)
    failures_path=output/'person1_inference_failures.csv'
    failures=pd.read_csv(failures_path).to_dict('records') if failures_path.exists() else []
    for name in frame.dataset.drop_duplicates():
        pending=frame[(frame.dataset==name)&frame.eligible&~frame.image_path.isin(done)]
        records=pending.to_dict('records')
        with pred_path.open('a',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS)
            if stream.tell()==0:writer.writeheader()
            for start in tqdm(range(0,len(records),batch_size),desc=f'Inference {name}',mininterval=3):
                batch=records[start:start+batch_size]
                paths=[Path(r['image_path']) if Path(r['image_path']).is_absolute() else image_root/r['image_path'] for r in batch]
                try:outputs=detector.predict_batch(paths)
                except (OSError,ValueError):
                    outputs=[]
                    for row,path in zip(batch,paths):
                        try:outputs.append(detector.predict(path))
                        except (OSError,ValueError) as exc:
                            outputs.append(None)
                            failures.append(dict(dataset=name,image_path=row['image_path'],error=str(exc)))
                for row,result in zip(batch,outputs):
                    if result is None:continue
                    writer.writerow(dict(dataset=name,image_path=row['image_path'],true_label=row['true_label'],
                                         predicted_label=int(result['fake_score']>.5),raw_logit=result['raw_logit'],
                                         fake_score=result['fake_score'],inference_time_ms=result['inference_time_ms'],
                                         image_id=row['image_id'],model_version=result['model_version'],
                                         checkpoint_sha256=result['checkpoint_sha256']))
                stream.flush()
    predictions=pd.read_csv(pred_path,float_precision='round_trip')
    done=set(predictions.image_path)
    failures=[r for r in failures if r['image_path'] not in done]
    pd.DataFrame(failures,columns=['dataset','image_path','error']).drop_duplicates().to_csv(failures_path,index=False)
    metrics=compute_metrics(predictions,frame)
    metrics['model_version']=detector.model_version
    # Preserve the measured baseline metrics file when only metadata is upgraded.
    if is_baseline and (output/'baseline_metrics.csv').exists():
        old_metrics=pd.read_csv(output/'baseline_metrics.csv')
        for col in ['Total','Accuracy','AP','Precision','Recall','F1','ROC_AUC','TN','FP','FN','TP','Time']:
            if not np.allclose(old_metrics[col],metrics[col],equal_nan=True):
                raise ValueError('Baseline metrics would change; use a separate output for a fresh experiment')
    else:metrics.to_csv(output/'baseline_metrics.csv',index=False)
    print(metrics.to_string(index=False),flush=True)
    return predictions,metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--manifest',type=Path,help='CSV with dataset,image_path,true_label; optional eligible,sha256,image_id')
    source.add_argument('--data-root',action='append',help='SOURCE=FOLDER, repeat for multiple sources; labels from real/fake folders')
    parser.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    parser.add_argument('--image-root',type=Path,default=ROOT,help='Base for relative image_path values (default: project root)')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    frame=folders_manifest(args.data_root) if args.data_root else pd.read_csv(args.manifest or ROOT/'results/data_manifest.csv')
    run_evaluation(frame,checkpoint=args.checkpoint,output=args.output,image_root=args.image_root,
                   batch_size=args.batch_size,threads=args.threads)


if __name__=='__main__':main()
