# Audit source và notebook trước triển khai

Checkout local HEAD: `781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a`.

| path | unchanged | git_blob |
| --- | --- | --- |
| networks/resnet.py | True | 2a5f40ae344bb6e292c89cfb265dbf059829e5b4 |
| data/datasets.py | True | d2d8a9b603abed4287e4d0dd1a7d29f6624d8d74 |
| data/__init__.py | True | 664e6eeaa752b54f53bdaf076766094a5dd86c1f |
| options/base_options.py | True | 1f67932869cb1e76049b9dbdee7ec606c3b541cc |
| options/test_options.py | True | 1dba350e3cadf758c2dc39e99a55b4e844e7b16d |
| test.py | True | 07966fbe294b93c0eb1f09b4bc119a05c4275817 |
| validate.py | True | 1f55e0fc6bd2d49cfada6b69d3f0fa354fe667f7 |
| NPR.pth | True | 2430d34a6d6a1143d46d4276a05536770b1a2a97 |

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


Một số dòng cấu hình từ notebook cũ:

```text
EXIF_TRANSPOSE = False  # Pipeline benchmark gốc không sửa EXIF; bật sẽ là giao thức khác.
EVAL_BATCH_SIZE = 8     # Toàn bộ ảnh, không cắt MAX_IMAGES; num_workers=0 phù hợp Windows/CPU.
    'ForenSynths': PROJECT_ROOT / 'datasets/Generalization_Test/ForenSynths_test',
# Nếu dữ liệu theo README nằm ở ForenSynths_train_val/test, đổi DATASET_ROOTS['ForenSynths'].
TRAIN_REFERENCE_ROOTS = [PROJECT_ROOT / 'datasets/ForenSynths_train_val/train',
                         PROJECT_ROOT / 'datasets/ForenSynths_train_val/val']
    for suite in ('ForenSynths', 'DiffusionForensics')
requested_sources = [('ForenSynths', 'progan', 'Nguồn training theo paper'),
    ('ForenSynths', 'stylegan', 'GAN khác'), ('ForenSynths', 'stylegan2', 'GAN khác'),
    ('ForenSynths', 'biggan', 'GAN khác')]
    for s in ['adm', 'ddpm', 'iddpm', 'ldm', 'pndm', 'vqdiffusion', 'sdv1_new', 'sdv2']]
 'ForenSynths': {'progan': (99.8, 100.0), 'stylegan': (96.3, 99.8), 'stylegan2': (97.3, 100.0),
    'sdv1_new': (97.4, 99.8), 'sdv2': (93.8, 100.0)}}
PAPER_MEANS = {'ForenSynths': (92.5, 96.1), 'DiffusionForensics': (95.3, 99.8)}
8. **Tiếp theo:** bổ sung ForenSynths/DiffusionForensics; kiểm tra provenance và split; chạy toàn bộ nguồn;
```


## Bổ sung hợp đồng bàn giao v2

inference.py hiện nhận wrapper model hoặc state dict phẳng bằng strict=True. model_epoch_last_3090.pth đã được kiểm tra thực tế nhưng không thay NPR.pth trong baseline. predict(image, model) và CSV có model_version/checkpoint_sha256; image_id ổn định để ghép đối chứng. evaluate_checkpoint.py nhận checkpoint/manifest/data-root/output riêng; evaluation.py và notebook dùng chung evaluator đó. Các trường số đo gốc của 38.976 dòng giữ nguyên; xem handover_upgrade_verification.json cho kiểm chứng old/new trên 50 ảnh thật và metadata nguồn phép đo cũ trong experiment_config.json. Không coi nâng cấp metadata là một lần chạy lại toàn bộ inference. Bản trước nâng cấp nằm trong results/person1_before_handover_v2/.
