"""Record source audit and produce a compact handover manifest without copying data."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo
import pandas as pd
from inference import ROOT, sha256
from evaluation import RESULTS, table_md, write_json


def git(*args):
    return subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}',*args],cwd=ROOT,text=True,encoding='utf-8').strip()


def source_audit():
    files=['networks/resnet.py','data/datasets.py','data/__init__.py','options/base_options.py','options/test_options.py','test.py','validate.py','NPR.pth']
    checks=[]
    for path in files:
        entry=git('ls-tree','HEAD',path)
        expected=entry.split()[2]
        actual=git('hash-object',path)
        checks.append(dict(path=path,git_blob=actual,head_blob=expected,unchanged=actual==expected,sha256=sha256(ROOT/path)))
    write_json(RESULTS/'source_hashes.json',dict(commit=git('rev-parse','HEAD'),files=checks))
    old=json.loads((RESULTS/'person1_originals/NPR_Reproduction_Complete.ipynb').read_text(encoding='utf-8'))
    old_code='\n'.join(''.join(c['source']) for c in old['cells'] if c['cell_type']=='code')
    config_lines=[line for line in old_code.splitlines() if any(t in line for t in ['EXIF_TRANSPOSE =','MAX_IMAGES','ForenSynths','sdv1_new'])]
    text='# Audit source và notebook trước triển khai\n\n'
    text+=f'Checkout local HEAD: `{git("rev-parse","HEAD")}`.\n\n'
    text+=table_md(pd.DataFrame(checks)[['path','unchanged','git_blob']])
    text+='''

## Inference/evaluation của tác giả

`test.py` xây mạng **custom** `networks.resnet.resnet50(num_classes=1)`. `ResNet` chỉ khởi tạo layer1/layer2 (3 và 4 Bottleneck), không phải torchvision ResNet50. 1.437.761 tham số; checkpoint có 146 state entries.

`NPR.pth` thực tế có khóa `model`, `optimizer`, `total_steps`, state model có tiền tố `module.`. `test.py` gọi load trực tiếp một state dict nên không xử lý được wrapper này; `validate.py` trong main có lấy khóa model nhưng không bỏ module. Module bổ sung chỉ tháo wrapper/prefix rồi strict=True; không sửa kiến trúc/trọng số. Checkpoint khớp blob NPR.pth trong Git HEAD local; không thay bằng checkpoint 3090. Khớp repository local không tự chứng minh đầy đủ lịch sử training của checkpoint.

`test.py` đặt no_resize=False và no_crop=True cho ForenSynths và DiffusionForensics; `options/base_options.py` mặc định loadSize=256. `data/datasets.py` thực thi Resize((256,256)) bilinear, ToTensor, Normalize ImageNet. `ImageFolder` mặc định PIL loader convert RGB, không EXIF transpose. Không flip khi test, augmentation đã comment. Một protocol khác cho GANGen-Detection là giữ nguyên kích thước; không dùng protocol đó cho ba nguồn local này. Giao thức no_resize/no_crop của AIGCDetectBenchmark trong README cũng không được áp dụng nhầm.

Trong model.forward: NPR=x−nearest_up(nearest_down(x,0.5),2), sau đó NPR×2/3 → Conv/BN/ReLU/MaxPool → layer1 → layer2 → avgpool → fc1. Không tính NPR trước khi đưa RGB normalized vào model. Sigmoid ngoài model; Fake=1 khi score>0.5 (không phải ≥); Real=0. Accuracy=sklearn.accuracy_score(y, score>0.5); AP=sklearn.average_precision_score(y, score). `data/__init__.py` nối các danh mục; tính metric trên các ảnh nối lại của từng nguồn.

## Đối chiếu notebook cũ và sửa đổi

Đã đọc cả `01_NPR_Full_RunAll.ipynb` (18 cell) và bản `NPR_Reproduction_Complete.ipynb` cũ (31 cell), source/results hiện có trước khi sửa. Notebook chính cũ đã resize 256×256, không crop, normalize đúng, load model đúng và NPR bên trong forward đúng. Không cần đổi phương pháp NPR. Notebook `01_NPR_Full_RunAll.ipynb` dùng EXIF transpose vô điều kiện; cách này khác PIL loader tác giả với ảnh có EXIF orientation. Notebook chính cũ có cờ EXIF_TRANSPOSE; phiên bản module mới luôn False theo source.

Notebook cũ tự chứa nhiều logic đánh giá và dùng tên benchmark đầy đủ khi tìm dữ liệu, kết quả cũ ghi 0 ảnh evaluation. Phiên bản mới quét chính xác ba thư mục local được chỉ định, giữ nhãn từ đường dẫn và chia sẻ inference/evaluation với script. Bản notebook cũ đã sao lưu trong `results/person1_originals/`; `01_NPR_Full_RunAll.ipynb` giữ nguyên như tư liệu cũ, không dùng để tạo baseline bàn giao.

Code tác giả bật LOAD_TRUNCATED_IMAGES=True. Theo yêu cầu loại file lỗi, audit mới verify và decode nghiêm ngặt với False, ghi file lỗi và loại khỏi manifest đủ điều kiện. Trong lượt này không có file lỗi nên không ảnh hưởng số ảnh đánh giá. Không resize/crop hay sửa dữ liệu trên ổ đĩa.

Điều chỉnh kỹ thuật: bỏ gọi `.cuda()` bắt buộc, dùng CPU, model.eval(), torch.inference_mode(), batch16, threads4, seed100. Không thay đổi dtype float32, không train/fine-tune/augmentation. Có tqdm, timing, per-image logit/score và fingerprint để tiếp tục. Thời gian ghi theo batch và có thể chịu ảnh hưởng tải CPU nền; không dùng như benchmark tốc độ tác giả.

Các kết quả cũ dưới `results/runs/` giữ nguyên, không được dùng như số đo của baseline này. Đầu ra mới chính thức là data_audit.csv, data_manifest.csv, baseline_metrics.csv, per_image_predictions.csv và báo cáo liên quan trong README_PERSON1.md.

## Bằng chứng bổ sung

`author_pipeline_parity.json` ghi so sánh tensor chính xác và raw logit trên ảnh thật PNG/JPEG với `data.datasets.binary_dataset` gốc. `inference_smoke_results.json` ghi dự đoán thật khi kiểm thử API. `source_hashes.json` ghi Git blob và SHA256 nguồn/checkpoint. `person1_tests.json` ghi trạng thái kiểm thử; không suy diễn PASS từ mô tả.

## Nguồn tham khảo

- [NPR code](https://github.com/chuangchuangtan/NPR-DeepfakeDetection).
- [Paper tác giả, arXiv v2](https://arxiv.org/pdf/2312.10461v2), Tables 1/3. Bản PDF CVPR cuối cùng chưa truy cập được trong phiên.
- [DIRE cấu trúc dataset](https://github.com/ZhendongWang6/DIRE#diffusionforensics-dataset). Phân biệt images/recons/dire với cấu trúc generator/nhãn mà NPR dùng; không chứng minh tương đương chỉ từ tên.
'''
    text+='\n\nMột số dòng cấu hình từ notebook cũ:\n\n```text\n'+'\n'.join(config_lines[:20])+'\n```\n'
    text+=('\n\n## Bổ sung hợp đồng bàn giao v2\n\n'
           'inference.py hiện nhận wrapper model hoặc state dict phẳng bằng strict=True. '
           'model_epoch_last_3090.pth đã được kiểm tra thực tế nhưng không thay NPR.pth trong baseline. '
           'predict(image, model) và CSV có model_version/checkpoint_sha256; image_id ổn định để ghép đối chứng. '
           'evaluate_checkpoint.py nhận checkpoint/manifest/data-root/output riêng; evaluation.py và notebook dùng chung evaluator đó. '
           'Các trường số đo gốc của 38.976 dòng giữ nguyên; xem handover_upgrade_verification.json cho kiểm chứng old/new trên 50 ảnh thật '
           'và metadata nguồn phép đo cũ trong experiment_config.json. Không coi nâng cấp metadata là một lần chạy lại toàn bộ inference. '
           'Bản trước nâng cấp nằm trong results/person1_before_handover_v2/.\n')
    (RESULTS/'source_audit.md').write_text(text,encoding='utf-8')
    return checks


def finalize():
    checks=source_audit()
    metrics=pd.read_csv(RESULTS/'baseline_metrics.csv')
    tests=json.loads((RESULTS/'person1_tests.json').read_text(encoding='utf-8'))
    audit=pd.read_csv(RESULTS/'data_audit.csv')
    now=datetime.now(ZoneInfo('Asia/Saigon')).isoformat()
    summary='# Kết quả bàn giao Người 1\n\nThời điểm lập báo cáo (Asia/Saigon): '+now+'\n\n'
    summary+='## Dataset\n\n'+table_md(audit[['Dataset','Real','Fake','Total','Valid','Invalid']])
    summary+='\n\n## Baseline thực tế (metric thang 0–1)\n\n'+table_md(metrics)
    summary+='\n\n## Kiểm thử\n\n'+table_md(pd.DataFrame(tests['tests'])[['test','status']])
    summary+='\n\n## Mức hoàn thành\n\n'
    summary+='- Hoàn thành: audit toàn bộ dữ liệu local, inference baseline toàn bộ ảnh hợp lệ, metrics/đồ thị, notebook, API và hướng dẫn bàn giao.\n'
    summary+='- Hoàn thành một phần: đối chiếu paper theo bản arXiv v2; Comparable=No do chưa xác minh provenance/split và sdv1_new.\n'
    summary+='- Chưa hoàn thành: xác minh archive gốc, danh tính generator sdv1 và benchmark CVPR cuối cùng; không thể suy ra từ các file ảnh local.\n'
    summary+='\nCờ chất lượng: xem image_quality_audit.md; ảnh đơn sắc decode được vẫn giữ trong baseline chính nhưng cần kiểm tra thủ công. 103 bản sao ảnh PNG đen của sdv1 gây 103 FN, không phải 103 nội dung độc lập.\n'
    summary+='\nXem README_PERSON1.md, data_audit.md, paper_comparison.md và error_analysis.md để biết nguồn, cách chạy và giới hạn. Không tuyên bố đã tái lập đầy đủ paper.\n'
    extension=json.loads((RESULTS/'handover_extension_tests.json').read_text(encoding='utf-8'))
    summary+='\n\n## Bổ sung bàn giao v2\n\n'+table_md(pd.DataFrame(extension['tests'])[['test','status']])
    summary+='\n\nĐã bổ sung lựa chọn checkpoint/manifest/output, model_version và image_id; kiểm chứng metadata không đổi số đo gốc. Xem HANDOVER_PERSON1.md. Nghiệm thu bởi thành viên khác: NOT RUN / chưa được xác nhận trong phiên này.\n'
    cleaned=RESULTS/'cleaned_black103'
    if (cleaned/'clean_test_checks.json').exists():
        cleaned_metrics=pd.read_csv(cleaned/'baseline_metrics.csv')
        summary+='\n\n## Phiên bản test đã làm sạch theo lựa chọn người dùng\n\n'+table_md(cleaned_metrics)
        summary+='\n\nManifest được dùng cho so sánh tiếp theo nằm trong results/cleaned_black103/data_manifest.csv. Loại 103 ảnh có pixel RGB đều bằng 0, còn 38.873 ảnh. Metrics tính lại từ dự đoán đã đo. Người dùng xác nhận dữ liệu là một phần tải từ nguồn tác giả do giới hạn máy. Chi tiết trong results/cleaned_black103/README.md.\n'
    (RESULTS/'PERSON1_COMPLETION.md').write_text(summary,encoding='utf-8')
    include=['NPR_Reproduction_Complete.ipynb','inference.py','evaluation.py','reporting.py','verify_person1.py',
             'execute_person1_notebook.py','build_person1_notebook.py','finalize_person1.py','README_PERSON1.md',
             'requirements_person1.txt','requirements_person1_environment.txt','experiment_config.json','NPR.pth','inspect_provenance.py','complete_person1.py','audit_image_quality.py',
             'evaluate_checkpoint.py','compare_predictions.py','verify_handover_extensions.py','upgrade_handover_metadata.py',
             'HANDOVER_PERSON1.md','test_protocol.json','team_acceptance.json','model_epoch_last_3090.pth',
             'prepare_clean_test.py','verify_clean_test.py']
    include += [str(p.relative_to(ROOT)) for p in (ROOT/'networks').glob('*.py')]
    result_names=['data_audit.csv','data_audit.md','data_manifest.csv','invalid_images.csv','manual_review.csv',
                  'exact_duplicates.csv','non_image_files.csv','baseline_metrics.csv','per_image_predictions.csv',
                  'paper_comparison.csv','paper_comparison.md','error_analysis.csv','error_analysis.md',
                  'category_metrics.csv','person1_inference_failures.csv','person1_tests.json','person1_tests.log',
                  'person1_notebook_execution.json','source_audit.md','source_hashes.json','author_pipeline_parity.json',
                  'inference_smoke_results.json','PERSON1_COMPLETION.md','person1_run.log','person1_experiment_config.json',
                  'duplicate_summary.csv','provenance_header_audit.json','provenance_header_audit.log','hardware_info.json',
                  'image_quality_flags.csv','image_quality_predictions.csv','image_quality_summary.csv','image_quality_audit.md',
                  'handover_upgrade_verification.json','handover_extension_tests.json','evaluation_manifest.csv']
    config=json.loads((ROOT/'experiment_config.json').read_text(encoding='utf-8'))
    manifest=pd.read_csv(RESULTS/'data_manifest.csv')
    original_protocol=dict(status='technical_snapshot_prepared; awaiting team approval',
        manifest='results/data_manifest.csv',manifest_sha256=sha256(RESULTS/'data_manifest.csv'),
        rows=int(manifest.eligible.sum()),datasets=audit[['Dataset','Real','Fake','Total']].to_dict('records'),
        checkpoint='NPR.pth',checkpoint_sha256=config['checkpoint_sha256'],model_version=config['model_version'],
        baseline_metrics_sha256=sha256(RESULTS/'baseline_metrics.csv'),
        preprocessing=config['preprocessing'],threshold=.5,threshold_operator='>',positive_label='Fake=1',
        image_id='sha256(dataset + NUL + logical image_path with forward slashes)',
        quality_policy='retain all decoded-valid paths including 103 copies of black PNG; manually flagged',
        source_and_official_split_verification='Not Verified',full_table1_reproduced=False)
    write_json(RESULTS/'original_test_protocol.json',original_protocol)
    active_protocol=original_protocol
    if (cleaned/'test_protocol.json').exists():
        active_protocol=json.loads((cleaned/'test_protocol.json').read_text(encoding='utf-8'))
    write_json(ROOT/'test_protocol.json',active_protocol)
    acceptance=ROOT/'team_acceptance.json'
    if not acceptance.exists():
        write_json(acceptance,dict(status='NOT RUN',reviewer=None,review_date=None,team_approval=None,
                   reason='Requires another actual team member to replay and review; agent tests do not constitute human acceptance',
                   instructions='HANDOVER_PERSON1.md'))
    include += ['results/'+p for p in result_names]
    include += [str(p.relative_to(ROOT)) for p in (RESULTS/'figures').glob('*.png')]
    include += [str(p.relative_to(ROOT)) for p in RESULTS.glob('*_samples.csv')]
    include += [str(p.relative_to(ROOT)) for p in (RESULTS/'handover_validation').rglob('*') if p.is_file()]
    include += ['results/original_test_protocol.json']
    if cleaned.exists():
        include += [str(p.relative_to(ROOT)) for p in cleaned.rglob('*') if p.is_file()]
    artifacts=[dict(path=p.replace('\\','/'),bytes=(ROOT/p).stat().st_size,sha256=sha256(ROOT/p)) for p in sorted(set(include))]
    write_json(ROOT/'handover_manifest.json',dict(created_asia_saigon=now,files=artifacts,
               dataset_reference=active_protocol['manifest'],
               checkpoint_reference='NPR.pth; reuse one existing file',unchanged_author_sources=all(x['unchanged'] for x in checks)))
    print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-only',action='store_true')
    if parser.parse_args().source_only:source_audit()
    else:finalize()
