"""Compare paired model predictions on exactly the same labeled image IDs."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from evaluation import compute_metrics


def compare(reference,candidate,output,expect_identical=False):
    reference=Path(reference);candidate=Path(candidate);output=Path(output)
    a=pd.read_csv(reference,float_precision='round_trip');b=pd.read_csv(candidate,float_precision='round_trip')
    for frame in [a,b]:
        if not {'image_id','dataset','true_label','model_version','fake_score','raw_logit','predicted_label'}<=set(frame):
            raise ValueError('Missing handover prediction columns')
        if frame.image_id.duplicated().any():raise ValueError('Duplicate image IDs')
        if frame.model_version.nunique()!=1:raise ValueError('Mixed model versions')
        if not np.isfinite(frame[['raw_logit','fake_score','inference_time_ms']]).all().all():raise ValueError('Non-finite output')
        if not frame.fake_score.between(0,1).all():raise ValueError('Invalid fake_score')
        if not ((frame.fake_score>.5).astype(int)==frame.predicted_label).all():raise ValueError('Invalid predicted label')
    if set(a.image_id)!=set(b.image_id):raise ValueError('Different test image IDs; comparison is not paired')
    merged=a.merge(b,on='image_id',suffixes=('_reference','_candidate'),validate='one_to_one')
    for col in ['dataset','true_label']:
        if not (merged[col+'_reference']==merged[col+'_candidate']).all():raise ValueError(f'{col} differs for the same image ID')
    manifest=a[['dataset','image_id']].assign(eligible=True)
    ma=compute_metrics(a,manifest);mb=compute_metrics(b,manifest)
    table=ma[['Dataset','Total','Accuracy','AP','F1']].merge(mb[['Dataset','Total','Accuracy','AP','F1']],on='Dataset',suffixes=('_reference','_candidate'))
    for metric in ['Accuracy','AP','F1']:table[metric+'_delta']=table[metric+'_candidate']-table[metric+'_reference']
    merged['fake_score_delta']=merged.fake_score_candidate-merged.fake_score_reference
    merged['label_changed']=merged.predicted_label_candidate!=merged.predicted_label_reference
    same=(not merged.label_changed.any() and np.allclose(merged.fake_score_reference,merged.fake_score_candidate,atol=1e-6,rtol=1e-6)
          and np.allclose(merged.raw_logit_reference,merged.raw_logit_candidate,atol=1e-4,rtol=1e-5))
    if expect_identical:
        if a.model_version.iloc[0]!=b.model_version.iloc[0]:raise ValueError('Reproduction requires the same checkpoint hash/version')
        if not same:raise ValueError('Baseline replay differs beyond tolerance')
    output.mkdir(parents=True,exist_ok=True)
    merged.to_csv(output/'paired_predictions.csv',index=False);table.to_csv(output/'metrics_comparison.csv',index=False)
    report=dict(status='PASS',paired_images=len(merged),baseline_replay_requested=expect_identical,
                equivalent_within_tolerance=bool(same),reference_model=a.model_version.iloc[0],candidate_model=b.model_version.iloc[0],
                full_benchmark_verified=False)
    (output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(table.to_string(index=False));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True,type=Path);p.add_argument('--candidate',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--expect-identical',action='store_true')
    a=p.parse_args();compare(a.reference,a.candidate,a.output,a.expect_identical)
