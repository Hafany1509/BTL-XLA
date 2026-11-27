"""One-time documented metadata upgrade; never recompute or replace baseline scores."""
import csv
import importlib.util
import json
from pathlib import Path
import pandas as pd
import torch
from inference import ROOT, CHECKPOINT, load_model, sha256
from evaluation import fingerprint, write_json
from evaluate_checkpoint import image_id


def main():
    backup=ROOT/'results/person1_before_handover_v2'
    proof=ROOT/'results/handover_upgrade_verification.json'
    if proof.exists():
        raise RuntimeError('Metadata upgrade already executed; do not silently migrate twice')
    for name,current in [('per_image_predictions.csv',ROOT/'results/per_image_predictions.csv'),
                         ('data_manifest.csv',ROOT/'results/data_manifest.csv'),
                         ('baseline_metrics.csv',ROOT/'results/baseline_metrics.csv')]:
        if sha256(backup/name)!=sha256(current):
            raise RuntimeError(f'Pre-upgrade baseline differs from its snapshot: {name}')
    spec=importlib.util.spec_from_file_location('npr_before_metadata_v2',backup/'inference.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    old=legacy.NPRDetector(CHECKPOINT);new=load_model()
    assert all(torch.equal(value,new.model.state_dict()[key]) for key,value in old.model.state_dict().items())
    manifest=pd.read_csv(backup/'data_manifest.csv')
    selected=manifest[manifest.eligible].groupby(['dataset','category','true_label'],sort=False).head(1)
    paths=[ROOT/p for p in selected.image_path]
    for path in paths:
        assert torch.equal(old.preprocess(legacy.read_rgb(path)),new.preprocess(legacy.read_rgb(path)))
    for start in range(0,len(paths),16):
        a=old.predict_batch(paths[start:start+16]);b=new.predict_batch(paths[start:start+16])
        for previous,current in zip(a,b):
            assert previous['raw_logit']==current['raw_logit']
            assert previous['fake_score']==current['fake_score'] and previous['label']==current['label']
    for name in ['data_manifest.csv','per_image_predictions.csv']:
        with (backup/name).open(encoding='utf-8',newline='') as stream:
            reader=csv.DictReader(stream);fields=list(reader.fieldnames);rows=list(reader)
        fields+=['image_id']
        if name=='per_image_predictions.csv':fields+=['model_version','checkpoint_sha256']
        for row in rows:
            row['image_id']=image_id(row['dataset'],row['image_path'])
            if name=='per_image_predictions.csv':
                row.update(model_version=new.model_version,checkpoint_sha256=new.checkpoint_sha256)
        with (ROOT/'results'/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    with (backup/'per_image_predictions.csv').open(encoding='utf-8',newline='') as stream:
        original=list(csv.DictReader(stream))
    with (ROOT/'results/per_image_predictions.csv').open(encoding='utf-8',newline='') as stream:
        updated=list(csv.DictReader(stream))
    assert len(original)==len(updated)==38976
    assert all(all(a[key]==b[key] for key in a) for a,b in zip(original,updated))
    assert sha256(ROOT/'results/baseline_metrics.csv')==sha256(backup/'baseline_metrics.csv')
    config=json.loads((backup/'experiment_config.json').read_text(encoding='utf-8'))
    original_fingerprint=config['run_fingerprint']
    config.update(run_fingerprint=fingerprint(pd.read_csv(ROOT/'results/data_manifest.csv'),new.checkpoint_sha256),
                  model_version=new.model_version,checkpoint_format=new.checkpoint_format,
                  image_root=str(ROOT),metadata_schema_version=2,
                  measurement_provenance=dict(original_run_fingerprint=original_fingerprint,
                      archived_sources='results/person1_before_handover_v2',
                      original_predictions_sha256=sha256(backup/'per_image_predictions.csv'),
                      full_inference_rerun=False,upgrade='Only image_id/model metadata added; all original CSV field strings preserved'))
    write_json(ROOT/'experiment_config.json',config)
    write_json(ROOT/'results/person1_experiment_config.json',config)
    flat=load_model(ROOT/'model_epoch_last_3090.pth')
    write_json(proof,dict(status='PASS',preserved_prediction_rows=len(original),all_original_fields_exact=True,
                          baseline_metrics_bytes_unchanged=True,parity_images=len(paths),
                          parity_paths=[str(p.relative_to(ROOT)) for p in paths],
                          pretrained_tensors_exact=True,preprocess_tensors_exact=True,logits_scores_labels_exact=True,
                          checkpoint_formats={new.checkpoint.name:new.checkpoint_format,flat.checkpoint.name:flat.checkpoint_format},
                          original_run_fingerprint=original_fingerprint,new_run_fingerprint=config['run_fingerprint'],
                          full_baseline_inference_rerun=False))
    print('PASS: metadata-only upgrade;',len(original),'original rows preserved;',len(paths),'real-image parity checks')


if __name__=='__main__':main()
