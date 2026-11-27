"""Meaningful checks against the author pipeline and actual complete local results."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
import numpy as np
import pandas as pd
from PIL import Image
import torch
import nbformat
from inference import ROOT, CHECKPOINT, load_model, predict, sha256
from evaluation import RESULTS, DATASETS, compute_metrics, fingerprint, write_json


def main(core_only=False):
    tests=[]
    started=time.perf_counter()
    detector=None
    manifest=pd.read_csv(RESULTS/'data_manifest.csv')
    def check(name,fn):
        try:
            evidence=fn()
            tests.append(dict(test=name,status='PASS',evidence=evidence))
        except Exception:
            tests.append(dict(test=name,status='FAIL',error=traceback.format_exc()))
        print(name,tests[-1]['status'],flush=True)
    def environment():
        import torchvision
        assert torch.ones(2).sum().item()==2
        return dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,device='cpu')
    check('1 Environment',environment)
    def checkpoint():
        nonlocal detector
        detector=load_model()
        assert detector.parameter_count==1437761 and detector.state_entries==146
        assert not detector.load_result.missing_keys and not detector.load_result.unexpected_keys
        assert not detector.model.training
        return dict(strict=True,parameters=detector.parameter_count,sha256=detector.checkpoint_sha256)
    check('2 Checkpoint',checkpoint)
    def data():
        assert manifest.image_path.is_unique
        assert manifest.true_label.isin([0,1]).all()
        for name in DATASETS:
            group=manifest[(manifest.dataset==name)&manifest.eligible]
            assert group.true_label.nunique()==2
            assert all((ROOT/p).is_file() for p in group.image_path)
        return pd.read_csv(RESULTS/'data_audit.csv')[['Dataset','Real','Fake','Total','Valid','Invalid']].to_dict('records')
    check('3 Dataset',data)
    def inference_check():
        assert detector is not None
        samples=manifest[manifest.eligible].groupby(['true_label','format']).head(1)
        results=[]
        for row in samples.itertuples():
            path=ROOT/row.image_path
            output=predict(path)
            assert {'label','raw_logit','fake_score','inference_time_ms'}<=output.keys()
            json.dumps(output,allow_nan=False)
            assert 0<=output['fake_score']<=1
            assert (output['label']=='FAKE')==(output['fake_score']>.5)
            assert np.isclose(output['fake_score'],torch.tensor(output['raw_logit']).sigmoid().item(),atol=1e-7)
            with Image.open(path) as im:
                assert abs(predict(im)['raw_logit']-output['raw_logit'])<1e-6
            assert abs(predict(path)['raw_logit']-output['raw_logit'])<1e-6
            results.append(dict(image_path=row.image_path,true_label=int(row.true_label),format=row.format,**output))
        assert load_model() is detector
        for invalid in [ROOT/'results/THIS_IMAGE_DOES_NOT_EXIST.png',ROOT/'README_PERSON1.md']:
            try:predict(invalid)
            except (FileNotFoundError,OSError):pass
            else:raise AssertionError('Invalid image accepted')
        try:predict(123)
        except TypeError:pass
        else:raise AssertionError('Invalid image type accepted')
        write_json(RESULTS/'inference_smoke_results.json',results)
        # Use author preprocessing implementation on real PNG and JPEG images.
        from data.datasets import binary_dataset
        opt=SimpleNamespace(isTrain=False,no_crop=True,no_flip=True,no_resize=False,loadSize=256,cropSize=224)
        parity=[]
        for dataset,category in [('progan','airplane'),('sdv1','imagenet')]:
            source=binary_dataset(opt,str(ROOT/'datasets'/dataset/category))
            candidates=[0,len(source)-1]
            for index in candidates:
                path,label=source.samples[index]
                author_tensor,author_label=source[index]
                from inference import read_rgb
                ours=detector.preprocess(read_rgb(path))
                assert torch.equal(author_tensor,ours)
                with torch.inference_mode():raw=detector.model(author_tensor.unsqueeze(0)).item()
                assert abs(raw-predict(path)['raw_logit'])<1e-6
                parity.append(dict(path=str(path),tensor_exact_match=True,label_match=int(label)==int(author_label)))
        write_json(RESULTS/'author_pipeline_parity.json',parity)
        return 'Real/Fake, PNG/JPEG, PIL/path, missing/corrupt input, repeated calls, original author transform and logit parity passed'
    check('4 Inference',inference_check)
    if not core_only:
        predictions=pd.read_csv(RESULTS/'per_image_predictions.csv')
        metrics=pd.read_csv(RESULTS/'baseline_metrics.csv')
        def evaluation_check():
            calculated=compute_metrics(predictions,manifest)
            for col in ['Accuracy','AP','Precision','Recall','F1','ROC_AUC','TN','FP','FN','TP']:
                assert np.allclose(calculated[col],metrics[col],rtol=1e-12,atol=1e-12)
            assert metrics.Complete.all()
            return metrics[['Dataset','Total','Accuracy','AP']].to_dict('records')
        check('5 Evaluation',evaluation_check)
        def results_check():
            assert predictions.image_path.is_unique
            assert set(predictions.image_path)==set(manifest.loc[manifest.eligible,'image_path'])
            assert np.isfinite(predictions[['raw_logit','fake_score','inference_time_ms']]).all().all()
            assert predictions.fake_score.between(0,1).all()
            assert ((predictions.fake_score>.5).astype(int)==predictions.predicted_label).all()
            assert np.allclose(torch.tensor(predictions.raw_logit.to_numpy(),dtype=torch.float32).sigmoid().numpy(),predictions.fake_score,atol=1e-7)
            merged=predictions.merge(manifest[['image_path','dataset','true_label']],on='image_path',suffixes=('','_manifest'),validate='one_to_one')
            assert (merged.true_label==merged.true_label_manifest).all()
            assert (merged.dataset==merged.dataset_manifest).all()
            return dict(real_evaluated_paths=len(predictions),all_eligible_covered=True)
        check('6 Results',results_check)
        def plots():
            files=[]
            for name in DATASETS:
                for kind in ['confusion_matrix','roc_curve','pr_curve','fake_score_distribution','correct_samples','incorrect_samples','near_threshold_samples']:
                    path=RESULTS/f'figures/{name}_{kind}.png'
                    with Image.open(path) as im:
                        im.verify()
                    files.append(str(path.relative_to(ROOT)))
            for name in ['baseline_comparison','npr_visualization','npr_architecture']:
                with Image.open(RESULTS/f'figures/{name}.png') as im:im.verify()
            return dict(per_source_plots=len(files),summary_plots=3)
        check('7 Visualization',plots)
        def notebook():
            nb=nbformat.read(ROOT/'NPR_Reproduction_Complete.ipynb',as_version=4);nbformat.validate(nb)
            status=json.loads((RESULTS/'person1_notebook_execution.json').read_text())
            assert status['status']=='PASS'
            expected_cells=sum(cell.cell_type=='code' for cell in nb.cells)
            assert status['executed_cells']==status['code_cells']==expected_cells
            assert expected_cells>=18
            assert all(o.output_type!='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
            return status
        check('8 Notebook',notebook)
        def reproducibility():
            config=json.loads((ROOT/'experiment_config.json').read_text())
            assert config['run_fingerprint']==fingerprint(manifest,detector.checkpoint_sha256)
            assert config['seed']==100 and config['checkpoint_sha256']==sha256(CHECKPOINT)
            # Actually exercise resume, asserting it does not change the saved rows.
            before=sha256(RESULTS/'per_image_predictions.csv')
            from evaluation import evaluate
            evaluate(manifest,detector)
            assert sha256(RESULTS/'per_image_predictions.csv')==before
            return dict(seed=100,fingerprint=config['run_fingerprint'],resume_preserved_predictions=True)
        check('9 Reproducibility',reproducibility)
        def handover():
            path=ROOT/manifest.loc[manifest.eligible,'image_path'].iloc[0]
            code='from inference import predict; import json,sys; print(json.dumps(predict(sys.argv[1])))'
            completed=subprocess.run([sys.executable,'-X','utf8','-c',code,str(path)],cwd=ROOT,text=True,capture_output=True,check=True)
            result=json.loads(completed.stdout)
            assert result['label'] in ['REAL','FAKE']
            return dict(independent_process=True,result=result)
        check('10 Handover',handover)
    else:
        for name in ['5 Evaluation','6 Results','7 Visualization','8 Notebook','9 Reproducibility','10 Handover']:
            tests.append(dict(test=name,status='NOT RUN',reason='Core-only run; full evaluation still running'))
    report=dict(tests=tests,elapsed_seconds=time.perf_counter()-started)
    write_json(RESULTS/'person1_tests.json',report)
    (RESULTS/'person1_tests.log').write_text('\n\n'.join(json.dumps(t,ensure_ascii=False,indent=2) for t in tests),encoding='utf-8')
    if any(t['status']=='FAIL' for t in tests):sys.exit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--core-only',action='store_true')
    main(parser.parse_args().core_only)
