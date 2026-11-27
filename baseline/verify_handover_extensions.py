"""Actual-image integration tests for configurable checkpoint evaluation and handover."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import pandas as pd
from inference import ROOT, CHECKPOINT, load_model, predict, sha256
from evaluation import write_json
from evaluate_checkpoint import run_evaluation
from compare_predictions import compare


def main():
    out=ROOT/'results/handover_validation'
    out.mkdir(exist_ok=True)
    manifest=pd.read_csv(ROOT/'results/data_manifest.csv')
    selected=manifest[manifest.eligible].groupby(['dataset','true_label'],sort=False).head(2)
    selected.to_csv(out/'manifest.csv',index=False)
    baseline=pd.read_csv(ROOT/'results/per_image_predictions.csv',float_precision='round_trip')
    baseline[baseline.image_id.isin(selected.image_id)].to_csv(out/'reference_subset.csv',index=False)
    tests=[]
    def check(name,fn):
        try:tests.append(dict(test=name,status='PASS',evidence=fn()))
        except Exception:tests.append(dict(test=name,status='FAIL',error=traceback.format_exc()))
        print(name,tests[-1]['status'],flush=True)
    def cli(args,log,should_fail=False):
        result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'evaluate_checkpoint.py'),*map(str,args)],cwd=ROOT,text=True,encoding='utf-8',capture_output=True)
        (out/log).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if should_fail:
            assert result.returncode!=0
        elif result.returncode:
            raise RuntimeError(result.stdout+'\n'+result.stderr)
        return result
    def models():
        original=load_model();flat=load_model(ROOT/'model_epoch_last_3090.pth')
        path=ROOT/selected.image_path.iloc[0]
        for model in [original,flat]:
            result=predict(path,model)
            assert result['model_version']==model.model_version
            assert result['checkpoint_sha256']==sha256(model.checkpoint)
            assert load_model(model.checkpoint) is load_model(model.checkpoint)
            assert not model.load_result.missing_keys and not model.load_result.unexpected_keys
            json.dumps(result,allow_nan=False)
        assert original.model_version!=flat.model_version
        return dict(wrapped=original.checkpoint_format,flat=flat.checkpoint_format,strict=True,frontend_api=True)
    check('Checkpoint formats and predict(image, model)',models)
    args=['--manifest',out/'manifest.csv','--checkpoint',CHECKPOINT,'--output',out/'baseline','--batch-size',4,'--threads',2]
    def custom_cli():
        cli(args,'baseline_cli.log')
        g=pd.read_csv(out/'baseline/per_image_predictions.csv')
        assert len(g)==len(selected)==12
        assert set(g.image_id)==set(selected.image_id)
        assert g.model_version.nunique()==1
        return dict(actual_images=len(g),datasets=list(g.dataset.unique()),batch_size=4,threads=2)
    check('Explicit manifest/checkpoint/output CLI',custom_cli)
    def resume():
        before=sha256(out/'baseline/per_image_predictions.csv')
        cli(args,'baseline_resume.log')
        assert sha256(out/'baseline/per_image_predictions.csv')==before
        return 'Completed cache preserved byte-for-byte'
    check('Completed-run resume',resume)
    def partial():
        resume_dir=out/'partial_resume';resume_dir.mkdir(exist_ok=True)
        with (out/'baseline/per_image_predictions.csv').open(encoding='utf-8',newline='') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
        with (resume_dir/'per_image_predictions.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows[:4])
        shutil.copy2(out/'baseline/experiment_config.json',resume_dir/'experiment_config.json')
        detector=load_model(CHECKPOINT,2);calls=[];original=detector.predict_batch
        def spy(images):calls.extend(str(p) for p in images);return original(images)
        detector.predict_batch=spy
        try:g,_=run_evaluation(selected,detector,output=resume_dir,batch_size=4,threads=2,verify_files=False)
        finally:detector.predict_batch=original
        assert len(calls)==8 and len(g)==12 and g.image_id.is_unique
        assert not set(calls)&{str(ROOT/r['image_path']) for r in rows[:4]}
        return dict(cached_actual_images=4,newly_inferred_actual_images=8,total=12)
    check('Interrupted-run resume',partial)
    def mismatch():
        before=sha256(out/'baseline/per_image_predictions.csv')
        wrong=list(args);wrong[wrong.index('--checkpoint')+1]=ROOT/'model_epoch_last_3090.pth'
        result=cli(wrong,'checkpoint_mismatch.log',True)
        assert 'Cache mismatch' in result.stderr
        assert sha256(out/'baseline/per_image_predictions.csv')==before
        return 'Different checkpoint rejected without modifying prior predictions'
    check('Checkpoint cache isolation',mismatch)
    def folder_cli():
        fixture=ROOT.parent/'tmp/npr_handover_fixture'
        for label in [0,1]:
            source=selected[selected.true_label==label].iloc[0]
            target=fixture/('real' if label==0 else 'fake')/Path(source.image_path).name
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():os.link(ROOT/source.image_path,target)
        cli(['--data-root',f'folder_smoke={fixture}','--checkpoint',ROOT/'model_epoch_last_3090.pth',
             '--output',out/'flat_checkpoint','--batch-size',2,'--threads',2],'flat_folder_cli.log')
        g=pd.read_csv(out/'flat_checkpoint/per_image_predictions.csv')
        assert len(g)==2 and set(g.true_label)=={0,1} and set(g.dataset)=={'folder_smoke'}
        assert (g.checkpoint_sha256==sha256(ROOT/'model_epoch_last_3090.pth')).all()
        return dict(actual_images=2,source_outside_project=True,flat_checkpoint_strict=True)
    check('External labeled folder and flat-checkpoint CLI',folder_cli)
    def invalid_labels():
        bad=selected.head(1).copy();bad['true_label']=2;bad.to_csv(out/'bad_labels.csv',index=False)
        result=cli(['--manifest',out/'bad_labels.csv','--output',out/'invalid_rejected'],'bad_labels.log',True)
        assert 'Real=0 or Fake=1' in result.stderr
        assert not (out/'invalid_rejected/per_image_predictions.csv').exists()
        return 'Unknown numeric label rejected before inference'
    check('Unknown label rejection',invalid_labels)
    def comparison():
        result=compare(out/'reference_subset.csv',out/'baseline/per_image_predictions.csv',out/'comparison',True)
        assert result['paired_images']==12 and result['equivalent_within_tolerance']
        return result
    check('Paired image-ID replay comparison',comparison)
    def history():
        old=ROOT/'results/person1_before_handover_v2'
        with (old/'per_image_predictions.csv').open(encoding='utf-8',newline='') as f:a=list(csv.DictReader(f))
        with (ROOT/'results/per_image_predictions.csv').open(encoding='utf-8',newline='') as f:b=list(csv.DictReader(f))
        assert len(a)==len(b)==38976
        assert all(all(x[k]==y[k] for k in x) for x,y in zip(a,b))
        assert sha256(old/'baseline_metrics.csv')==sha256(ROOT/'results/baseline_metrics.csv')
        return dict(original_rows=38976,all_original_csv_fields_exact=True,metrics_file_unchanged=True)
    check('Historical full-baseline preservation',history)
    write_json(ROOT/'results/handover_extension_tests.json',dict(tests=tests,scope='real small integration tests; historical full baseline preserved; not a new full benchmark run'))
    if any(t['status']!='PASS' for t in tests):sys.exit(1)


if __name__=='__main__':main()
