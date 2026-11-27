"""Build the Vietnamese teaching notebook, preserving the original notebook."""
import json
from pathlib import Path
import shutil
import nbformat as nbf
from inference import ROOT

backup=ROOT/'results/person1_originals'
backup.mkdir(exist_ok=True)
for name in ['NPR_Reproduction_Complete.ipynb','paper_comparison.csv']:
    source=ROOT/name if name.endswith('.ipynb') else ROOT/'results'/name
    if source.exists() and not (backup/name).exists():shutil.copy2(source,backup/name)
cells=[]
def section(title,text,code):
    cells.append(nbf.v4.new_markdown_cell(f'## {title}\n\n{text}'))
    cells.append(nbf.v4.new_code_cell(code))

section('1. Introduction',
        'Tái lập **baseline NPR gốc** trên toàn bộ ảnh test local hợp lệ của ProGAN, StyleGAN2 và sdv1. '
        'Không train/fine-tune, không đổi kiến trúc, không tải dữ liệu. Real=0, Fake=1. '
        'Kết quả local không tự động tương đương benchmark paper. Các cell dùng chung inference.py, evaluation.py và reporting.py. '
        'Mặc định đọc kết quả đã chạy; bật các cờ dưới đây để chạy lại bước dài.',
        """from pathlib import Path
import os, sys, json
ROOT = Path.cwd()
if not (ROOT / 'inference.py').is_file():
    ROOT = ROOT / 'NPR-DeepfakeDetection'
assert (ROOT / 'inference.py').is_file(), 'Mở notebook từ thư mục dự án hoặc D:/XLA'
os.chdir(ROOT)
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
RERUN_AUDIT = False
RERUN_EVALUATION = False
RERUN_REPORTS = False
RESULTS = ROOT / 'results'
print('Project:', ROOT)
""")
section('2. Environment Check',
        'CPU là cấu hình đã thực nghiệm. Không gọi CUDA. Danh sách phiên bản thực tế và seed được ghi trong experiment_config.json. '
        'Notebook không tự pip install.',
        """import torch, torchvision, numpy as np, pandas as pd
from IPython.display import display, Image as DisplayImage, Markdown
from inference import load_model, read_rgb, PREPROCESS_CONFIG
from evaluation import audit, evaluate, fingerprint
print(sys.version)
print('torch:', torch.__version__, '| torchvision:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available(), '| evaluation device: CPU')
""")
section('3. Dataset Audit',
        'Quét đệ quy và chỉ gán nhãn từ thư mục có tên xác định. Ảnh hỏng, nhãn không rõ hoặc hash có nhãn xung đột bị loại khỏi đánh giá. '
        'Không xóa dữ liệu gốc; giữ các đường dẫn trùng cùng nhãn và báo cáo riêng. Metadata nguồn chưa được xác minh. '
        'Valid nghĩa là verify/decode được. Ảnh đơn sắc được gắn cờ trong image_quality_flags.csv để kiểm tra thủ công; '
        'vẫn giữ trong baseline chính, không coi số bản sao là các tình huống thị giác độc lập.',
        """if RERUN_AUDIT:
    manifest = audit()
else:
    manifest = pd.read_csv(RESULTS / 'data_manifest.csv')
data_audit = pd.read_csv(RESULTS / 'data_audit.csv')
display(data_audit[['Dataset','Real','Fake','Total','Valid','Invalid','Unknown','Eligible','DuplicateFiles','CrossDatasetDuplicateFiles']])
display(Markdown((RESULTS / 'data_audit.md').read_text(encoding='utf-8')))
""")
section('4. Load NPR Model',
        'Dùng custom networks.resnet.resnet50(num_classes=1), không dùng ResNet50 của torchvision. '
        'NPR.pth có khóa model; module cũng hỗ trợ state dict phẳng như model_epoch_last_3090.pth, bỏ đúng tiền tố module. và strict=True. '
        'Baseline vẫn dùng NPR.pth; model_version gắn với SHA256 checkpoint. Bộ nhớ cache tái dùng từng phiên bản. '
        'Checkpoint trùng Git blob của repository local; xem source_audit.md.',
        """detector = load_model()
model = detector.model
print(detector.load_result)
print('Parameters:', detector.parameter_count, '| State entries:', detector.state_entries)
print('Checkpoint SHA256:', detector.checkpoint_sha256)
assert not model.training
config = json.loads((ROOT / 'experiment_config.json').read_text(encoding='utf-8'))
if not RERUN_EVALUATION:
    assert fingerprint(manifest, detector.checkpoint_sha256) == config['run_fingerprint'], 'Cache không khớp; chạy evaluation.py'
""")
section('5. Image Preprocessing',
        'Theo test.py cho ForenSynths/DiffusionForensics: RGB → Resize((256,256), bilinear) → ToTensor → Normalize ImageNet. '
        'Không crop, flip, augmentation hay EXIF transpose. data/datasets.py đặt LOAD_TRUNCATED_IMAGES=True; '
        'pipeline này kiểm tra decode nghiêm ngặt để loại file lỗi theo yêu cầu. Không tính NPR ở bước preprocessing.',
        """print(json.dumps(PREPROCESS_CONFIG, indent=2))
sample = manifest[manifest.eligible].iloc[0]
sample_path = ROOT / sample.image_path
image = read_rgb(sample_path)
x = detector.preprocess(image).unsqueeze(0)
print('Ảnh:', sample.image_path, '| Original size:', image.size, '| Tensor:', tuple(x.shape))
display(image.resize((256,256)))
""")
section('6. NPR Visualization',
        'Trong forward, NPR = x − up_nearest(down_nearest(x,0.5),2), rồi nhân 2/3 trước convolution. '
        'NPR biểu diễn chênh lệch so với pixel tham chiếu trong ô 2×2. Hình dưới chỉ minh họa; '
        'không đưa residual trở lại mạng để tránh tính NPR hai lần.',
        """from reporting import model_visuals
if RERUN_REPORTS or not (RESULTS/'figures/npr_visualization.png').exists():
    model_visuals(detector, manifest)
display(DisplayImage(filename=str(RESULTS/'figures/npr_visualization.png')))
""")
section('7. CNN Architecture',
        'Mạng nhẹ 1.437.761 tham số: Conv3×3 stride2 → BN/ReLU/MaxPool → layer1 với 3 Bottleneck → '
        'layer2 với 4 Bottleneck → AdaptiveAvgPool → Linear(512,1). Tên resnet50 không có nghĩa là mạng ResNet50 đầy đủ.',
        """print(model)
display(DisplayImage(filename=str(RESULTS/'figures/npr_architecture.png')))
""")
section('8. Single Image Inference',
        'Raw logit đi qua sigmoid thành fake_score. Chỉ khi score > 0.5 mới dự đoán Fake, đúng validate.py tác giả. '
        'Score là điểm mô hình, chưa được hiệu chuẩn thành xác suất thực tế.',
        """single_result = detector.predict(sample_path)
display(pd.DataFrame([single_result]))
assert 0 <= single_result['fake_score'] <= 1
json.dumps(single_result)
""")
section('9. Multi-Image Inference',
        'Dùng cùng detector, không load lại checkpoint. Batch có chung resize 256×256; thời gian mỗi ảnh là thời gian batch '
        '(decode/preprocess/forward/sigmoid) chia cho số ảnh, không phải phép đo latency ảnh đơn độc lập.',
        """examples = manifest[manifest.eligible].groupby('true_label',sort=True).head(2)
outputs = detector.predict_batch([ROOT/p for p in examples.image_path])
display(pd.concat([examples[['dataset','image_path','true_label']].reset_index(drop=True),pd.DataFrame(outputs)],axis=1))
""")
section('10. ProGAN Evaluation',
        'Mặc định đọc kết quả đã lưu từ inference thật, đã bổ sung image_id/model_version mà giữ nguyên số đo gốc. '
        'Xem handover_upgrade_verification.json để kiểm chứng nâng cấp metadata. Nếu bật RERUN_EVALUATION, đánh giá chung cả ba nguồn và tiếp tục từ '
        'các batch đã lưu có fingerprint khớp. Không giới hạn 1.000 ảnh. AP lấy fake_score liên tục, positive=Fake.',
        """if RERUN_EVALUATION:
    predictions, metrics = evaluate(manifest, detector)
else:
    predictions = pd.read_csv(RESULTS/'per_image_predictions.csv')
    metrics = pd.read_csv(RESULTS/'baseline_metrics.csv')
display(metrics[metrics.Dataset=='progan'])
""")
section('11. StyleGAN2 Evaluation',
        'Đánh giá toàn bộ đường dẫn hợp lệ hiện có; không bổ sung ảnh thiếu để ép bằng số lượng benchmark. '
        'Đây là nguồn khác ProGAN, nhưng chưa đủ metadata để xác nhận split chuẩn.',
        "display(metrics[metrics.Dataset=='stylegan2'])")
section('12. Stable Diffusion Evaluation',
        'Tên sdv1 là tên nguồn local, chưa chứng minh generator Stable Diffusion v1 hay sự tương đương sdv1_new. '
        'Tỷ lệ Real/Fake mất cân bằng được giữ nguyên để đánh giá toàn bộ dữ liệu hiện có. '
        'Không dùng residual DIRE làm thay thế ảnh RGB.',
        "display(metrics[metrics.Dataset=='sdv1'])")
section('13. Confusion Matrix',
        'Hàng là nhãn thật, cột là nhãn dự đoán; thứ tự Real rồi Fake. FP = Real bị coi là Fake, FN = Fake bị bỏ sót. '
        'Đọc số đếm tuyệt đối cùng tỷ lệ lớp.',
        """from reporting import generate_reports
if RERUN_REPORTS: generate_reports(detector)
for name in ['progan','stylegan2','sdv1']:
    display(DisplayImage(filename=str(RESULTS/f'figures/{name}_confusion_matrix.png')))
""")
section('14. ROC / PR Curves',
        'ROC-AUC và AP dùng score liên tục, không dùng nhãn threshold. PR thể hiện trade-off precision/recall; '
        'AP khác Accuracy và phụ thuộc tỷ lệ positive. Nếu chỉ có một lớp, không báo cáo AP/AUC như hợp lệ.',
        """for name in ['progan','stylegan2','sdv1']:
    for kind in ['roc_curve','pr_curve','fake_score_distribution']:
        path=RESULTS/f'figures/{name}_{kind}.png'
        if path.exists(): display(DisplayImage(filename=str(path)))
        else: print(name,kind,': không đủ điều kiện, xem báo cáo')
""")
section('15. Paper Comparison',
        'Các số paper được kiểm tra từ bản tác giả arXiv v2, Tables 1/3. Chưa truy cập được PDF CVPR cuối cùng. '
        'Comparable=No do chưa xác minh archive/test split, và riêng sdv1 chưa xác minh sdv1_new. '
        'Chênh lệch điểm phần trăm chỉ mô tả, không chứng minh tái lập paper.',
        """display(pd.read_csv(RESULTS/'paper_comparison.csv'))
display(Markdown((RESULTS/'paper_comparison.md').read_text(encoding='utf-8')))
""")
section('16. Error Analysis',
        'Xem mẫu đúng, sai tự tin cao và gần threshold 0.5. Chỉ kết luận theo số đo thực tế, '
        'không suy diễn nguyên nhân thị giác khi chưa kiểm chứng. Danh sách đầy đủ có trong error_analysis.csv.',
        """display(Markdown((RESULTS/'error_analysis.md').read_text(encoding='utf-8')))
for name in ['progan','stylegan2','sdv1']:
    for kind in ['correct','incorrect','near_threshold']:
        display(DisplayImage(filename=str(RESULTS/f'figures/{name}_{kind}_samples.png')))
""")
section('17. Experiment Summary',
        'Tổng hợp thực nghiệm local. Accuracy/Precision/Recall/F1/AP/AUC trong CSV có thang [0,1]; '
        'bảng đối chiếu paper dùng phần trăm. Thời gian tổng là tổng thời gian inference theo batch, '
        'không gồm audit/đồ thị. Toàn bộ ảnh hợp lệ phải có đúng một dòng dự đoán.',
        """display(metrics)
assert not predictions.image_path.duplicated().any()
assert set(predictions.image_path)==set(manifest.loc[manifest.eligible,'image_path'])
assert metrics.Complete.all()
display(DisplayImage(filename=str(RESULTS/'figures/baseline_comparison.png')))
print('Tổng ảnh thực tế đã đánh giá:',len(predictions))
print('Cấu hình:',ROOT/'experiment_config.json')
""")
section('18. Conclusions',
        'Baseline gốc được giữ nguyên; kết quả phản ánh các tập local đã audit. '
        'Không kết luận tái lập đầy đủ benchmark khi provenance/split chưa được xác minh. '
        'Người 2 dùng cùng data_manifest.csv, checkpoint hash, preprocessing, threshold và metric; '
        'lưu dự đoán cải tiến vào thư mục riêng, không ghi đè baseline. Đọc README_PERSON1.md để tích hợp phần mềm.',
        """display(metrics[['Dataset','Total','Accuracy','AP','F1','Complete']])
print('Inference độc lập: from inference import predict; result = predict(image_path)')
print('Bàn giao:', ROOT/'README_PERSON1.md')
""")
if (ROOT/'results/cleaned_black103/baseline_metrics.csv').exists():
    section('19. Kết quả trên tập test đã làm sạch',
            'Theo lựa chọn của người dùng, loại các ảnh có mọi pixel RGB bằng 0 khỏi phiên bản test đã làm sạch. '
            'Tiêu chí được kiểm tra trên tất cả nguồn và cả hai nhãn. Có 103 ảnh sdv1/Fake đáp ứng tiêu chí. '
            'Người dùng xác nhận đã tải một phần từ nguồn tác giả do giới hạn máy. '
            'Metrics tính lại từ dự đoán thực tế đã lưu. Time tổng hợp từ lần inference ban đầu. '
            'Người 2 dùng manifest trong results/cleaned_black103 để đánh giá các phiên bản trên cùng ảnh.',
            """CLEAN_RESULTS = ROOT/'results/cleaned_black103'
clean_manifest = pd.read_csv(CLEAN_RESULTS/'data_manifest.csv')
clean_predictions = pd.read_csv(CLEAN_RESULTS/'per_image_predictions.csv')
clean_metrics = pd.read_csv(CLEAN_RESULTS/'baseline_metrics.csv')
assert len(clean_manifest) == len(clean_predictions) == 38873
assert set(clean_manifest.image_id) == set(clean_predictions.image_id)
display(clean_metrics[['Dataset','Real','Fake','Total','Accuracy','Precision','Recall','F1','AP','ROC_AUC','FP','FN']])
display(pd.read_csv(CLEAN_RESULTS/'before_after_metrics.csv'))
display(Markdown((CLEAN_RESULTS/'README.md').read_text(encoding='utf-8')))
for figure in ['baseline_comparison','sdv1_confusion_matrix','sdv1_pr_curve']:
    display(DisplayImage(filename=str(CLEAN_RESULTS/f'figures/{figure}.png')))
""")
nb=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='Python (NPR CPU)',language='python',name='npr-person1'),language_info=dict(name='python',version='3.11.9')))
nbf.validate(nb)
nbf.write(nb,ROOT/'NPR_Reproduction_Complete.ipynb')
print('Created notebook:',len(cells),'cells; original backed up at',backup)
