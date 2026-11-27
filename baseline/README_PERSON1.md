# NPR baseline — bàn giao Người 1

**Phiên bản test đã làm sạch được người dùng chọn để so sánh tiếp theo** nằm trong `results/cleaned_black103/`. Dùng `results/cleaned_black103/data_manifest.csv` cho cả NPR gốc và các checkpoint cải tiến. Có 38.873 ảnh sau khi loại 103 ảnh sdv1/Fake có mọi pixel RGB bằng 0. `results/cleaned_black103/baseline_metrics.csv` và `before_after_metrics.csv` trình bày kết quả sau làm sạch và đối chiếu trước/sau. Metrics mới được tính từ các dòng dự đoán đã thực đo. Hướng dẫn đầy đủ ở `results/cleaned_black103/README.md`, notebook phần 19 và `test_protocol.json`.

Người dùng xác nhận các ảnh local là phần đã tải từ nguồn tác giả cung cấp do giới hạn dung lượng/tài nguyên máy. Tên archive, link tải chính xác và độ tương ứng với split paper hiện chưa được đối chiếu. Những nhận xét trong báo cáo ban đầu về metadata nguồn phản ánh thông tin tại thời điểm audit.

**Bổ sung theo phân công mới:** đọc [HANDOVER_PERSON1.md](HANDOVER_PERSON1.md) để dùng API `predict(image, model)`, chọn checkpoint, truyền manifest/output, đối chiếu từng image_id và thực hiện nghiệm thu nhóm. API/CSV hiện có model_version và checkpoint_sha256; image_id đã thêm vào manifest và CSV. Hỗ trợ NPR.pth có wrapper và checkpoint phẳng bằng strict=True. Xem handover_upgrade_verification.json để xác nhận mọi giá trị đo baseline gốc được giữ nguyên khi thêm metadata.

Mục tiêu: đánh giá checkpoint NPR gốc trên toàn bộ ảnh test local hợp lệ, giữ kết quả có thể truy vết. Không train/fine-tune, sửa kiến trúc hay tải dataset. Kết quả và giới hạn nguồn dữ liệu nằm trong `results/data_audit.md`, `results/baseline_metrics.csv`, `results/paper_comparison.md`, `results/error_analysis.md`.

## Cấu trúc và nguồn

- `networks/resnet.py`, `data/`, `options/`, `test.py`, `validate.py`: source tác giả, giữ nguyên.
- `NPR.pth`: checkpoint trong [NPR repository](https://github.com/chuangchuangtan/NPR-DeepfakeDetection); Git blob local khớp checkpoint trong HEAD, xem `results/source_audit.md`. Không dùng `model_epoch_last_3090.pth` cho baseline này.
- `inference.py`: load một lần, preprocess RGB, predict ảnh hoặc batch trên CPU.
- `evaluation.py`: audit, nhãn, hash, inference tiếp tục được, metrics sklearn.
- `reporting.py`: biểu đồ và phân tích từ dự đoán thật.
- `NPR_Reproduction_Complete.ipynb`: 18 phần có giải thích tiếng Việt, dùng các module trên.
- `datasets/progan`, `datasets/stylegan2`, `datasets/sdv1`: dữ liệu local được cung cấp. Không có metadata archive/split chính thức; tên thư mục không đủ xác minh nguồn. `datasets/test` không tự trộn vào đánh giá.
- `results/person1_originals/`: bản notebook và bảng paper cũ được bảo toàn. Các run cũ trong `results/runs/` vẫn giữ nguyên.

Nguồn benchmark được tác giả nêu: [CNNDetection](https://github.com/PeterWang512/CNNDetection) cho ForenSynths; [DIRE](https://github.com/ZhendongWang6/DIRE) cho DiffusionForensics. Không tuyên bố dữ liệu local là đúng archive đó khi chưa có bằng chứng. sdv1 local chưa được xác nhận là Stable Diffusion v1 hoặc tương đương `sdv1_new` của NPR. DIRE phân biệt ảnh RGB (`images`), ảnh tái dựng (`recons`), và reconstruction error (`dire`); NPR tự tạo biểu diễn từ RGB.

## Chạy bằng PowerShell

```powershell
Set-Location D:\XLA\NPR-DeepfakeDetection
$nprPython = 'D:\XLA\.venv\Scripts\python.exe'
$nprImage = (Import-Csv results\data_manifest.csv | Where-Object eligible -eq 'True' | Select-Object -First 1).image_path
& $nprPython -X utf8 inference.py $nprImage
& $nprPython -X utf8 evaluation.py
& $nprPython -X utf8 reporting.py
& $nprPython -X utf8 verify_person1.py
& $nprPython -X utf8 execute_person1_notebook.py
```

Sau khi evaluation hoàn tất, có thể dùng `& $nprPython -X utf8 complete_person1.py` để lần lượt tạo báo cáo, chạy notebook kernel mới, kiểm thử và cập nhật manifest bàn giao. Thêm `--wait` nếu evaluation còn đang chạy.

Không cần activate khi gọi thẳng interpreter. Nếu muốn: `. D:\XLA\.venv\Scripts\Activate.ps1`. Dependency chính được ghi trong `requirements_person1.txt` và `experiment_config.json`; toàn bộ environment thực tế (bao gồm scipy/opencv/Jupyter) ở `requirements_person1_environment.txt`. Không chạy lại requirements cũ để hạ phiên bản environment đang dùng. Package `tqdm` được bổ sung riêng trong phiên triển khai. Notebook không tự cài package.

`evaluation.py` mặc định audit lại toàn bộ file, sau đó dùng lại dự đoán đã lưu khi fingerprint khớp. `--audit-only` chỉ audit. `--reuse-audit` chỉ dùng khi bạn bảo đảm ảnh không đổi. Mỗi batch 16 ảnh được flush ra CSV; chạy lại cùng lệnh để tiếp tục nếu ngắt. Không chạy đồng thời hai evaluation vào cùng file kết quả. Nếu bị tắt máy giữa lúc ghi một dòng CSV, kiểm tra và khôi phục dòng cuối từ bản sao trước khi tiếp tục.

Cache gắn với hash ảnh, nhãn, checkpoint, preprocessing, code, batch/threads và phiên bản torch/torchvision. Khi thay checkpoint/tập ảnh, dùng evaluate_checkpoint.py với --output mới; không trộn hai protocol hoặc ghi đè baseline. Nâng cấp metadata v2 được kiểm chứng và ghi rõ nguồn gốc phép đo cũ trong experiment_config.json; không tự bỏ kiểm tra fingerprint. Không tự giới hạn 1.000 ảnh và không tự loại bản trùng cùng nhãn. Không gộp các nguồn thành một metric chung khi có ảnh trùng giữa nguồn.

Mở notebook từ thư mục dự án, chọn kernel của `D:\XLA\.venv\Scripts\python.exe`, Run All từ kernel mới. Mặc định đọc kết quả thật đã lưu; `RERUN_AUDIT`, `RERUN_EVALUATION`, `RERUN_REPORTS` điều khiển chạy lại. Script `execute_person1_notebook.py` tạo kernelspec trong workspace, chạy kernel mới và lưu notebook đã có output cùng log.

## Tích hợp Người 2

```python
from inference import load_model, predict

detector = load_model()  # cached; CPU; NPR.pth mặc định tương đối với module
result = predict('path/to/actual/image.jpg')
# result: label, fake_score, raw_logit, inference_time_ms, model_version, checkpoint_sha256
# predict(image, detector) cũng được hỗ trợ.
# Cũng hỗ trợ PIL.Image.Image:
# result = detector.predict(pil_image)
# results = detector.predict_batch([path1, path2])
```

Đặt thư mục dự án vào Python path hoặc chạy backend từ đó. Bàn giao cùng `inference.py`, `networks/` và một bản `NPR.pth`; không cần mở Jupyter. Đường dẫn không tồn tại/ảnh lỗi ném exception rõ ràng; backend bắt `FileNotFoundError`, `OSError`, `ValueError`, `TypeError` và trả lỗi cho người dùng. Không coi lỗi đọc ảnh là Real. Khởi tạo detector khi backend khởi động; gọi tuần tự hoặc quản lý hàng đợi inference khi dùng nhiều request.

Preprocessing: convert RGB, resize bilinear 256×256, ToTensor, normalize mean `[.485,.456,.406]` / std `[.229,.224,.225]`, không crop/EXIF transpose. NPR chỉ tính **bên trong** forward, nhân 2/3 trước convolution. Raw logit → sigmoid, Fake khi score **> 0.5**, Real khi score ≤ 0.5. Không normalize/NPR lần nữa trước khi gọi API. Fake score chưa được calibration, không trình bày như xác suất chắc chắn ảnh giả.

## Đọc và so sánh kết quả

`baseline_metrics.csv`: Accuracy là tỷ lệ dự đoán đúng tại threshold 0.5; Precision/Recall/F1 coi Fake là positive; AP dùng fake_score liên tục. CSV này dùng thang 0–1. `paper_comparison.csv` dùng phần trăm và chênh lệch điểm phần trăm. ROC-AUC/AP để trống khi chỉ một lớp. `Time` là giây; `MeanTime_ms` là ms/ảnh. Thời gian gồm decode/preprocess/forward/sigmoid, chia thời gian batch cho các ảnh; không gồm audit/plot, không phải latency ảnh đơn.

Dùng đúng `results/data_manifest.csv` (`eligible=True`) làm tập test cho cả hai mô hình; kiểm tra SHA256, thứ tự không ảnh hưởng metric nhưng mỗi path phải xuất hiện đúng một lần. Giữ Real=0, Fake=1, cùng resize và threshold 0.5 cho so sánh baseline. Nếu phương pháp cải tiến cần preprocessing riêng, mô tả rõ và xuất một cấu hình riêng. Không chỉnh threshold trên test; dùng validation riêng. Lưu per-image logit/score của mô hình cải tiến vào thư mục khác, dùng `evaluation.compute_metrics(predictions, manifest)` cho cùng công thức; các cột bắt buộc giống CSV baseline. Báo cáo cùng số mẫu, FP/FN, Accuracy/AP/F1, từng nguồn và thời gian trên cùng CPU/batch. Không ghi đè kết quả baseline.

## Hạn chế và kiểm chứng

Nguồn archive, generator của sdv1 và test split chính thức chưa xác minh. Phiên bản environment khác tác giả; chạy CPU. SHA256 chỉ tìm trùng byte, không chứng minh không có leakage theo nội dung/train. sdv1 có tỷ lệ lớp 1:2 nên AP phải đọc cùng prevalence. PDF CVPR cuối cùng không truy cập được trong phiên; các số paper được dẫn từ bản tác giả arXiv v2, ghi rõ trong báo cáo. `Comparable=No` cho đến khi có bằng chứng split/generator tương ứng; không tuyên bố tái hiện đầy đủ benchmark.

Kiểm tra nội dung bổ sung: `results/image_quality_audit.md` và `image_quality_flags.csv` gắn cờ ảnh đơn sắc sau khi kiểm tra extrema RGB của toàn bộ ảnh hợp lệ. sdv1 có 103 bản sao của một ảnh PNG đen, gây 103 FN. Chúng decode được và được giữ trong số liệu baseline chính, đồng thời chuyển vào danh sách cần kiểm tra thủ công. Không đổi nhãn hoặc tự suy đoán nguyên nhân ảnh đen. Muốn loại ảnh sau khi xác minh nguồn, tạo lượt đánh giá mới với cùng manifest mới cho cả hai mô hình. Có thể chạy lại kiểm tra bằng `audit_image_quality.py`.

`results/person1_tests.json` và `results/person1_tests.log` ghi trạng thái kiểm thử đã thực chạy. `results/person1_notebook_execution.json` ghi riêng việc thực thi notebook kernel mới; `results/hardware_info.json` ghi CPU đã dùng. `handover_manifest.json` liệt kê file và hash bàn giao, không sao chép dataset/checkpoint vào ZIP.
