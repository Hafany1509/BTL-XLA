"""Create the user-requested all-black-excluded test version from measured predictions."""
import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from inference import ROOT, CHECKPOINT, sha256
from evaluation import compute_metrics, fingerprint, table_md, write_json

RESULTS = ROOT / 'results'
OUTPUT = RESULTS / 'cleaned_black103'
SOURCE_STATEMENT = 'Người cung cấp dữ liệu xác nhận đã tải một phần dataset từ nguồn do tác giả cung cấp vì giới hạn tài nguyên máy.'


def check_file(row):
    path = ROOT / row.image_path
    if sha256(path) != row.sha256:
        raise ValueError(f'Image differs from the audited snapshot: {row.image_path}')
    with Image.open(path) as image:
        extrema = image.convert('RGB').getextrema()
    return row.image_path if all(maximum == 0 for minimum, maximum in extrema) else None


def select_csv(source, destination, paths):
    """Preserve every selected measured field as its original CSV string."""
    with source.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = [row for row in reader if row['image_path'] in paths]
    with destination.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    original_paths = [RESULTS / name for name in ['data_manifest.csv', 'per_image_predictions.csv', 'baseline_metrics.csv']]
    original_hashes = {path.name: sha256(path) for path in original_paths}
    manifest = pd.read_csv(original_paths[0])
    predictions = pd.read_csv(original_paths[1], float_precision='round_trip')
    config = json.loads((ROOT / 'experiment_config.json').read_text(encoding='utf-8'))
    if sha256(CHECKPOINT) != config['checkpoint_sha256']:
        raise ValueError('Baseline checkpoint changed')
    if not (predictions.checkpoint_sha256 == config['checkpoint_sha256']).all():
        raise ValueError('Mixed checkpoint predictions')
    if manifest.image_path.duplicated().any() or predictions.image_path.duplicated().any():
        raise ValueError('Duplicate logical paths')
    eligible = manifest[manifest.eligible]
    if set(eligible.image_path) != set(predictions.image_path):
        raise ValueError('Measured baseline does not cover the audited eligible paths')
    # Apply one pixel-based quality criterion to all sources and both classes.
    with ThreadPoolExecutor(max_workers=4) as pool:
        inspected = list(tqdm(pool.map(check_file, eligible.itertuples()), total=len(eligible), desc='Verify files and all-black criterion'))
    excluded_paths = {path for path in inspected if path is not None}
    if len(excluded_paths) != 103:
        raise ValueError(f'Expected the reviewed 103-file snapshot, found {len(excluded_paths)} all-black files. Review the changed dataset first.')
    flagged = pd.read_csv(RESULTS / 'image_quality_flags.csv')
    if excluded_paths != set(flagged.loc[flagged.flag == 'all_black', 'image_path']):
        raise ValueError('The reviewed all-black exclusion list changed')
    retained_paths = set(eligible.image_path) - excluded_paths
    OUTPUT.mkdir(exist_ok=True)
    selected_rows = select_csv(original_paths[0], OUTPUT / 'data_manifest.csv', retained_paths)
    selected_predictions = select_csv(original_paths[1], OUTPUT / 'per_image_predictions.csv', retained_paths)
    select_csv(original_paths[0], OUTPUT / 'excluded_images.csv', excluded_paths)
    selection = manifest.copy()
    selection['excluded_all_black'] = selection.image_path.isin(excluded_paths)
    selection['eligible'] &= ~selection.excluded_all_black
    selection['exclusion_reason'] = np.where(selection.excluded_all_black, 'all RGB pixels exactly zero', '')
    selection.to_csv(OUTPUT / 'selection_manifest.csv', index=False)
    cleaned_manifest = pd.read_csv(OUTPUT / 'data_manifest.csv')
    cleaned_predictions = pd.read_csv(OUTPUT / 'per_image_predictions.csv', float_precision='round_trip')
    metrics = compute_metrics(cleaned_predictions, cleaned_manifest)
    metrics.to_csv(OUTPUT / 'baseline_metrics.csv', index=False)
    before = compute_metrics(predictions, manifest)
    comparison = before[['Dataset','Real','Fake','Total','Accuracy','AP','F1','FP','FN']].merge(
        metrics[['Dataset','Real','Fake','Total','Accuracy','AP','F1','FP','FN']], on='Dataset', suffixes=('_original','_cleaned'))
    for metric in ['Accuracy', 'AP', 'F1']:
        comparison[metric + '_delta_pp'] = (comparison[metric + '_cleaned'] - comparison[metric + '_original']) * 100
    comparison.to_csv(OUTPUT / 'before_after_metrics.csv', index=False)
    counts = metrics[['Dataset','Real','Fake','Total']]
    cleaned_config = dict(config)
    cleaned_config.update(run_fingerprint=fingerprint(cleaned_manifest, config['checkpoint_sha256']),
        scope='author-provided local subset, all-black-excluded version, all remaining eligible paths',
        source_statement=SOURCE_STATEMENT, source_statement_evidence='user statement in this conversation',
        exact_archive_and_official_test_split='Not Verified',
        derivation=dict(method='filter measured full-baseline rows by image ID/path, recompute sklearn metrics',
            inference_reexecuted=False, source_predictions='results/per_image_predictions.csv',
            source_predictions_sha256=original_hashes['per_image_predictions.csv'],
            retained_rows=len(cleaned_predictions), excluded_rows=len(excluded_paths),
            script='prepare_clean_test.py', script_sha256=sha256(Path(__file__))),
        timing='sum/mean of original per-image allocated batch times for retained paths; no fresh CPU inference timing')
    write_json(OUTPUT / 'experiment_config.json', cleaned_config)
    write_json(OUTPUT / 'test_protocol.json', dict(
        status='user-selected cleaned test version; teammate replay pending',
        manifest='results/cleaned_black103/data_manifest.csv',
        manifest_sha256=sha256(OUTPUT / 'data_manifest.csv'),
        checkpoint='NPR.pth', checkpoint_sha256=config['checkpoint_sha256'], model_version=config['model_version'],
        rows=len(cleaned_manifest), datasets=counts.to_dict('records'),
        preprocessing=config['preprocessing'], threshold=0.5, threshold_operator='>', positive_label='Fake=1',
        quality_policy='exclude all images whose decoded RGB pixels are all zero, across all sources and both labels',
        excluded_manifest='results/cleaned_black103/excluded_images.csv',
        source_statement=SOURCE_STATEMENT, exact_archive_and_official_test_split='Not Verified',
        full_table1_reproduced=False))
    audit_rows = []
    for name, group in cleaned_manifest.groupby('dataset', sort=False):
        audit_rows.append(dict(Dataset=name, Real=int((group.true_label==0).sum()), Fake=int((group.true_label==1).sum()),
                               Total=len(group), Excluded_black=int(((manifest.dataset==name)&manifest.image_path.isin(excluded_paths)).sum())))
    pd.DataFrame(audit_rows).to_csv(OUTPUT / 'data_audit.csv', index=False)
    # Reuse the same figure and analysis implementations with an isolated output directory.
    import reporting
    reporting.RESULTS = OUTPUT
    reporting.FIGURES = OUTPUT / 'figures'
    reporting.figures(cleaned_predictions, metrics, scope_label='cleaned local test')
    reporting.analysis(cleaned_predictions, metrics)
    analysis_path = OUTPUT / 'error_analysis.md'
    analysis_text = analysis_path.read_text(encoding='utf-8')
    fake_ratio = float(metrics.loc[metrics.Dataset=='sdv1','Fake'].iloc[0] / metrics.loc[metrics.Dataset=='sdv1','Total'].iloc[0])
    analysis_text = analysis_text.replace('Do tỷ lệ Fake của sdv1 là 2/3', f'Do tỷ lệ Fake của sdv1 đã làm sạch là {fake_ratio:.4%}')
    analysis_path.write_text(SOURCE_STATEMENT + '\n\nTập test được chọn sau khi loại toàn bộ ảnh có pixel RGB đều bằng 0.\n\n' + analysis_text, encoding='utf-8')
    sdv1 = metrics[metrics.Dataset=='sdv1'].iloc[0]
    readme = ('# Kết quả NPR trên tập test đã làm sạch\n\n' + SOURCE_STATEMENT + '\n\n' +
        'Đã kiểm tra hash và extrema RGB của toàn bộ 38.976 ảnh trong snapshot test. Tiêu chí loại là mọi pixel RGB bằng 0, áp dụng cho cả hai nhãn và mọi nguồn. '
        'Có 103 đường dẫn đáp ứng tiêu chí, cùng nằm trong sdv1/Fake và có cùng SHA256. Danh sách cụ thể nằm trong excluded_images.csv.\n\n' +
        table_md(pd.DataFrame(audit_rows)) + '\n\n' + table_md(metrics) + '\n\n' +
        f'sdv1 đã làm sạch có {int(sdv1.Total)} ảnh, FP={int(sdv1.FP)}, FN={int(sdv1.FN)}. '
        'Accuracy/AP/F1 trong CSV dùng thang 0–1. before_after_metrics.csv ghi cả hai phiên bản và chênh lệch điểm phần trăm.\n\n' +
        'Các logit, score, nhãn dự đoán và thời gian mỗi ảnh được lấy nguyên từ lần inference baseline đã thực chạy. Metrics và biểu đồ được tính lại trên các dòng còn lại. '
        'Lần tạo phiên bản này dùng lại dự đoán đã lưu. Time/MeanTime_ms được tổng hợp từ thời gian batch phân bổ ban đầu.\n\n' +
        'Người 2 sử dụng results/cleaned_black103/data_manifest.csv cho mọi checkpoint so sánh. Giữ image_id, nhãn, preprocessing và threshold 0.5. '
        'Dùng output riêng cho mỗi lần replay hoặc mô hình cải tiến. Để chạy inference mới toàn bộ tập đã làm sạch, dùng lệnh sau từ root dự án.\n\n' +
        '```powershell\n& D:\\XLA\\.venv\\Scripts\\python.exe -X utf8 evaluate_checkpoint.py --manifest results\\cleaned_black103\\data_manifest.csv --checkpoint NPR.pth --output results\\cleaned_replay --batch-size 16 --threads 4\n```\n\n' +
        'Nguồn tải từ tác giả và việc chỉ tải một phần được ghi theo thông tin người dùng cung cấp. Link/gói archive và sự tương ứng chính xác với sdv1_new vẫn cần đối chiếu. '
        'Tiêu chí làm sạch là một thay đổi protocol phải ghi khi trình bày với giảng viên.\n')
    (OUTPUT / 'README.md').write_text(readme, encoding='utf-8')
    write_json(OUTPUT / 'verification.json', dict(status='PASS', audited_files=len(eligible),
        excluded_paths=len(excluded_paths), retained_paths=len(retained_paths),
        excluded_fake=103, excluded_real=0, all_original_input_hashes_verified=True,
        selected_prediction_fields_preserved=True, inference_reexecuted=False,
        original_artifacts={path.name: dict(sha256=original_hashes[path.name],unchanged=sha256(path)==original_hashes[path.name]) for path in original_paths}))
    if not all(sha256(path)==original_hashes[path.name] for path in original_paths):
        raise AssertionError('An original baseline artifact changed')
    assert len(selected_rows) == len(selected_predictions) == 38873
    assert int(sdv1.Real)==5000 and int(sdv1.Fake)==9897 and int(sdv1.FN)==277 and int(sdv1.FP)==86
    print(metrics.to_string(index=False))
    print('PASS: cleaned manifest and real-score-derived metrics saved to', OUTPUT)


if __name__ == '__main__':
    main()
