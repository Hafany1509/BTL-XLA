"""Verify exclusion membership, preserved measured outputs and cleaned metrics."""
import csv
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score
from inference import ROOT, sha256
from evaluation import write_json


def main():
    original=ROOT/'results'
    clean=original/'cleaned_black103'
    manifest=pd.read_csv(clean/'data_manifest.csv')
    excluded=pd.read_csv(clean/'excluded_images.csv')
    predictions=pd.read_csv(clean/'per_image_predictions.csv')
    metrics=pd.read_csv(clean/'baseline_metrics.csv')
    before=pd.read_csv(original/'data_manifest.csv')
    assert len(manifest)==len(predictions)==38873 and len(excluded)==103
    assert excluded.dataset.eq('sdv1').all() and excluded.true_label.eq(1).all()
    assert set(manifest.image_path).isdisjoint(excluded.image_path)
    assert set(manifest.image_path)|set(excluded.image_path)==set(before.loc[before.eligible,'image_path'])
    assert manifest.image_id.is_unique and predictions.image_id.is_unique
    assert set(manifest.image_id)==set(predictions.image_id)
    with (original/'per_image_predictions.csv').open(encoding='utf-8',newline='') as stream:
        source={row['image_id']:row for row in csv.DictReader(stream)}
    with (clean/'per_image_predictions.csv').open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            assert row==source[row['image_id']]
    for row in metrics.itertuples():
        group=predictions[predictions.dataset==row.Dataset]
        assert np.isclose(row.Accuracy,accuracy_score(group.true_label,group.predicted_label))
        assert np.isclose(row.AP,average_precision_score(group.true_label,group.fake_score))
        assert np.isclose(row.F1,f1_score(group.true_label,group.predicted_label))
    sdv1=predictions[predictions.dataset=='sdv1']
    assert confusion_matrix(sdv1.true_label,sdv1.predicted_label,labels=[0,1]).tolist()==[[4914,86],[277,9620]]
    proof=json.loads((clean/'verification.json').read_text(encoding='utf-8'))
    for name,item in proof['original_artifacts'].items():
        assert sha256(original/name)==item['sha256'] and item['unchanged']
    assert proof['all_original_input_hashes_verified'] and not proof['inference_reexecuted']
    result=dict(status='PASS',retained_paths=38873,excluded_paths=103,
                original_measured_fields_exact=True,image_ids_preserved=True,
                cleaned_metrics_recomputed_and_verified=True,sdv1_confusion_matrix=[[4914,86],[277,9620]],
                original_baseline_artifacts_preserved=True,source_images_preserved=True)
    write_json(clean/'clean_test_checks.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
