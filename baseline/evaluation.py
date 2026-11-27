"""Audit all local test images, run resumable original NPR CPU evaluation."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score)
import torch
from tqdm.auto import tqdm
from inference import ROOT, UPSTREAM, CHECKPOINT, PREPROCESS_CONFIG, load_model, sha256

RESULTS = ROOT / 'results'
DATASETS = ('progan', 'stylegan2', 'sdv1')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.ppm', '.pgm'}
LABELS = {'0_real': 0, 'real': 0, '1_fake': 1, 'fake': 1}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def table_md(frame):
    # Avoid an optional tabulate dependency.
    cols = list(frame.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for row in frame.fillna('N/A').itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(str(v).replace('|', '/') for v in row) + ' |')
    return '\n'.join(lines)


def inspect_image(item):
    name, path, root = item
    relative = path.relative_to(root)
    labels = {LABELS[p.lower()] for p in relative.parts[:-1] if p.lower() in LABELS}
    row = dict(dataset=name, image_path=path.relative_to(ROOT).as_posix(),
               true_label=next(iter(labels)) if len(labels) == 1 else None,
               label_status='known' if len(labels) == 1 else 'manual_review',
               category=relative.parts[0] if len(relative.parts) > 2 else '',
               extension=path.suffix.lower(), width=None, height=None, format='',
               valid=False, error='', sha256='', file_bytes=path.stat().st_size)
    try:
        row['sha256'] = sha256(path)
        with Image.open(path) as im:
            row.update(width=im.width, height=im.height, format=im.format)
            im.verify()
        with Image.open(path) as im:
            im.convert('RGB').load()
        row['valid'] = True
    except (OSError, ValueError, SyntaxError) as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    return row


def audit():
    RESULTS.mkdir(exist_ok=True)
    # Strict decoding: damaged/truncated files are explicitly excluded.
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    items, nonimages = [], []
    for name in DATASETS:
        root = ROOT / 'datasets' / name
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob('*')):
            if path.is_file():
                if path.suffix.lower() in EXTENSIONS:
                    items.append((name, path, root))
                else:
                    nonimages.append(dict(dataset=name, path=str(path)))
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(tqdm(pool.map(inspect_image, items), total=len(items), desc='Audit images'))
    manifest = pd.DataFrame(rows)
    manifest['eligible'] = manifest['valid'] & manifest['true_label'].notna()
    from evaluate_checkpoint import image_id
    manifest['image_id'] = [image_id(d, p) for d, p in zip(manifest.dataset, manifest.image_path)]
    manifest.to_csv(RESULTS / 'data_manifest.csv', index=False)
    manifest.loc[~manifest.valid].to_csv(RESULTS / 'invalid_images.csv', index=False)
    manifest.loc[manifest.label_status != 'known'].to_csv(RESULTS / 'manual_review.csv', index=False)
    pd.DataFrame(nonimages, columns=['dataset', 'path']).to_csv(RESULTS / 'non_image_files.csv', index=False)
    duplicates = manifest.loc[manifest.sha256.ne('') & manifest.duplicated('sha256', keep=False)].copy()
    duplicates['cross_dataset'] = duplicates.groupby('sha256').dataset.transform('nunique') > 1
    duplicates['label_conflict'] = duplicates.groupby('sha256').true_label.transform('nunique') > 1
    duplicates.to_csv(RESULTS / 'exact_duplicates.csv', index=False)
    if duplicates.label_conflict.any():
        conflict_hashes = set(duplicates.loc[duplicates.label_conflict, 'sha256'])
        manifest.loc[manifest.sha256.isin(conflict_hashes), 'eligible'] = False
        manifest.to_csv(RESULTS / 'data_manifest.csv', index=False)
    summary = []
    for name in DATASETS:
        group = manifest[manifest.dataset == name]
        dup = duplicates[duplicates.dataset == name]
        summary.append(dict(Dataset=name, Path=str(ROOT/'datasets'/name),
                            Real=int((group.true_label == 0).sum()), Fake=int((group.true_label == 1).sum()),
                            Total=len(group), Valid=int(group.valid.sum()), Invalid=int((~group.valid).sum()),
                            Unknown=int(group.true_label.isna().sum()), Eligible=int(group.eligible.sum()),
                            Formats=json.dumps(group.format.value_counts().to_dict()),
                            Dimensions=json.dumps(Counter(f'{int(r.width)}x{int(r.height)}' for r in group.itertuples() if pd.notna(r.width))),
                            DuplicateFiles=len(dup), CrossDatasetDuplicateFiles=int(dup.cross_dataset.sum()),
                            Provenance='Not Verified'))
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS/'data_audit.csv', index=False)
    report = '# Kiểm tra dữ liệu thực tế\n\n' + table_md(summary.drop(columns=['Dimensions']))
    report += '\n\n## Kích thước ảnh\n\n' + table_md(summary[['Dataset','Dimensions']])
    report += ('\n\nNhãn chỉ lấy từ thành phần thư mục chính xác: 0_real/real = 0, 1_fake/fake = 1. '
               'Ảnh không rõ nhãn hoặc có nhãn xung đột không được đánh giá. Không thay đổi dữ liệu gốc. '
               'File lỗi: invalid_images.csv; cần kiểm tra nhãn: manual_review.csv; trùng byte: exact_duplicates.csv. '
               'Giữ toàn bộ đường dẫn test hợp lệ, kể cả bản trùng cùng nhãn; không gộp chỉ số giữa các nguồn. '
               'SHA256 phát hiện trùng byte, không phát hiện ảnh giống nhau sau nén/chỉnh sửa, không chứng minh không có train/test leakage.\n\n'
               '## Nguồn dữ liệu\n\nKhông có metadata nguồn trong ba thư mục ảnh hiện tại. '
               'Tên thư mục và nội dung ảnh không đủ chứng minh generator, checkpoint sinh ảnh hoặc split. '
               'ProGAN/StyleGAN2 có cấu trúc tương thích ForenSynths nhưng chưa xác minh danh sách ảnh benchmark. '
               'sdv1 có nhãn thư mục và được đánh giá như nguồn local sdv1; không xác nhận thực sự được sinh bởi Stable Diffusion v1. '
               'Không coi sdv1 tương đương sdv1_new. DIRE có dữ liệu ảnh RGB, ảnh tái dựng và residual DIRE; '
               'NPR nhận ảnh RGB, tự tính NPR, không nhận residual DIRE. Không có metadata để chứng minh các file local là gói RGB gốc nào. '
               'README tác giả tổ chức lại benchmark dưới DiffusionForensics/sdv1_new; đổi tên thư mục không chứng minh tương đương.\n\n'
               'datasets/test được kiểm tra riêng, không tự động trộn vào ba nguồn: ')
    report += str(sum(p.is_file() for p in (ROOT/'datasets/test').rglob('*'))) + ' file.\n'
    report += f'\nThời gian audit: {time.perf_counter()-started:.2f} giây.\n'
    (RESULTS/'data_audit.md').write_text(report, encoding='utf-8')
    return manifest


def fingerprint(manifest, checkpoint_hash, batch_size=16, threads=4):
    columns = ['dataset','image_path','true_label','sha256','eligible']
    if 'image_id' in manifest: columns.append('image_id')
    values = manifest[columns].to_csv(index=False)
    payload = values + json.dumps(PREPROCESS_CONFIG, sort_keys=True) + checkpoint_hash
    for path in [ROOT/'inference.py', ROOT/'evaluation.py', ROOT/'evaluate_checkpoint.py',
                 UPSTREAM/'networks'/'resnet.py']:
        payload += sha256(path)
    payload += f'|strict_score_gt_0.5|cpu|batch{batch_size}|threads{threads}|' 
    payload += torch.__version__ + importlib.metadata.version('torchvision')
    return hashlib.sha256(payload.encode()).hexdigest()


def evaluate(manifest=None, detector=None, *, checkpoint=CHECKPOINT, output=None,
             image_root=ROOT, batch_size=16, threads=None, verify_files=False):
    """Shared notebook/script evaluator; explicit outputs isolate model variants."""
    from evaluate_checkpoint import run_evaluation
    if manifest is None:
        manifest = pd.read_csv(RESULTS/'data_manifest.csv')
    elif isinstance(manifest, (str, Path)):
        manifest = pd.read_csv(manifest)
        verify_files = True
    return run_evaluation(manifest, detector, checkpoint, output or RESULTS,
                          image_root, batch_size, threads, verify_files)


def compute_metrics(predictions, manifest):
    rows=[]
    for name in manifest.dataset.drop_duplicates():
        g=predictions[predictions.dataset==name]
        eligible=int(((manifest.dataset==name)&manifest.eligible).sum())
        y=g.true_label; p=g.predicted_label; s=g.fake_score
        if g.empty:
            rows.append(dict(Dataset=name,Total=0,Eligible=eligible,Complete=False));continue
        both=y.nunique()==2
        tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
        rows.append(dict(Dataset=name,Real=int((y==0).sum()),Fake=int((y==1).sum()),Total=len(g),
                         Accuracy=accuracy_score(y,p),Precision=precision_score(y,p,zero_division=0) if both else None,
                         Recall=recall_score(y,p,zero_division=0) if both else None,F1=f1_score(y,p,zero_division=0) if both else None,
                         AP=average_precision_score(y,s) if both else None,
                         ROC_AUC=roc_auc_score(y,s) if both else None,TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),
                         Time=float(g.inference_time_ms.sum()/1000),MeanTime_ms=float(g.inference_time_ms.mean()),
                         Eligible=eligible,Complete=len(g)==eligible,Metric_note='Fake positive; one-class AP/AUC/PR/F1 undefined' if not both else 'Fake=1 positive'))
    return pd.DataFrame(rows)


if __name__=='__main__':
    # Preserve the original audit CLI and expose the general checkpoint evaluator.
    if any(arg in sys.argv for arg in ['--checkpoint','--manifest','--data-root','--output','--image-root','--batch-size','--threads']):
        from evaluate_checkpoint import main
        main()
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--audit-only',action='store_true')
        parser.add_argument('--reuse-audit',action='store_true')
        args=parser.parse_args()
        manifest=pd.read_csv(RESULTS/'data_manifest.csv') if args.reuse_audit else audit()
        if not args.audit_only:
            evaluate(manifest)
