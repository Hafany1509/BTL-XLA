# Bàn giao Người 1 và nghiệm thu baseline

## Cập nhật tập test đã làm sạch

Người dùng đã chọn loại 103 ảnh toàn đen khỏi phiên bản test dùng cho so sánh tiếp theo. Manifest được chọn là `results/cleaned_black103/data_manifest.csv`, có 38.873 ảnh. sdv1 gồm 5.000 Real và 9.897 Fake. Tiêu chí loại được áp dụng cho tất cả nguồn và cả hai nhãn bằng cách kiểm tra mọi pixel RGB bằng 0. Xem `results/cleaned_black103/README.md`, `baseline_metrics.csv`, `before_after_metrics.csv` và `excluded_images.csv` để trình bày với giảng viên.

Khi chạy lại hoặc đánh giá mô hình cải tiến, truyền manifest đã làm sạch ở trên và chọn output riêng. Ví dụ lệnh từ root dự án là `python evaluate_checkpoint.py --manifest results/cleaned_black103/data_manifest.csv --checkpoint NPR.pth --output results/cleaned_replay`. Khi đối chiếu bản chạy lại, dùng reference `results/cleaned_black103/per_image_predictions.csv` trong compare_predictions.py. Các lệnh dùng `results/data_manifest.csv` ở phần bên dưới phục vụ tái lập lượt baseline ban đầu.

Người dùng xác nhận đã tải một phần dataset từ nguồn do tác giả cung cấp vì giới hạn máy. Ghi nguồn và phạm vi tập con theo thông tin này khi viết báo cáo. Gói archive và split chính xác với paper cần được đối chiếu thêm.

Phần kỹ thuật đã bổ sung API chọn mô hình, model_version, image_id, checkpoint phẳng và evaluator có tham số. Không train hoặc fine-tune. Kết quả baseline vẫn dùng NPR.pth trên 38.976 đường dẫn test đã audit. Đối chiếu đầy đủ paper, nguồn sdv1 và xác nhận của thành viên khác vẫn cần bằng chứng bên ngoài; không ghi nhận các bước này là đã hoàn tất.

## API cho thành viên frontend

```python
from inference import load_model, predict

baseline = load_model()  # NPR.pth mặc định
other = load_model('model_epoch_last_3090.pth')  # checkpoint tác giả khác, KHÔNG phải bản cải tiến của nhóm
result = predict(image_path_or_pil, baseline)
other_result = predict(image_path_or_pil, other)
```

`predict(image)` cũ vẫn hoạt động. Tham số `model` nhận NPRDetector do load_model trả về hoặc đường dẫn checkpoint. Không đưa một mạng CNN tùy ý vào API này. Cả wrapper có khóa model lẫn state dict phẳng đều được kiểm tra tensor, bỏ tiền tố module. nếu có và load strict=True. Không sửa kiến trúc để ép checkpoint không tương thích vào mạng.

Kết quả gồm `label`, `fake_score`, `raw_logit`, `inference_time_ms`, `model_version`, `checkpoint_sha256`. `model_version` là `npr-sha256:` cộng SHA256 đầy đủ của file checkpoint, không phụ thuộc tên file. Frontend có thể hiển thị nhãn thân thiện kèm 12 ký tự hash đầu, nhưng lưu version đầy đủ trong log. Điểm số chưa được hiệu chuẩn thành xác suất; hiển thị là “điểm nghiêng về ảnh AI”.

Cache giữ tối đa bốn detector theo đối số khởi tạo để tái dùng các phiên bản. Backend nên gọi load_model một lần cho mỗi checkpoint khi khởi động và giữ các detector đó. API chạy CPU; threads mặc định 4. Không tính NPR hoặc normalize thêm trước khi gọi.

## Schema CSV bàn giao

`results/per_image_predictions.csv` có các cột:

| Cột | Ý nghĩa |
| --- | --- |
| image_id | ID ổn định từ dataset và đường dẫn logical; dùng để ghép đối chứng |
| dataset | Tên nguồn local; không tự chứng minh generator thực tế |
| image_path | Đường dẫn ảnh đã đánh giá |
| true_label | Real=0, Fake=1 |
| predicted_label | Fake khi fake_score > 0.5 |
| raw_logit, fake_score | Đầu ra thực tế; AP lấy score liên tục |
| model_version, checkpoint_sha256 | Phiên bản và hash checkpoint dùng cho dòng này |
| inference_time_ms | Thời gian batch chia số ảnh, gồm đọc/preprocess/forward/sigmoid |

Các trường kết quả gốc của toàn bộ 38.976 dòng được giữ nguyên dưới dạng chuỗi CSV khi bổ sung metadata, không suy diễn lại timing. Bản trước nâng cấp nằm trong `results/person1_before_handover_v2/`. `results/handover_upgrade_verification.json` ghi 50 ảnh kiểm chứng tiền xử lý/logit/score cũ và mới trùng khớp, trạng thái model weights giống hệt và file metrics không đổi. Không tuyên bố đã chạy inference lại toàn bộ tập trong lần nâng cấp metadata này. Fingerprint cấu hình mới được cập nhật có ghi nguồn gốc phép đo cũ.

## Lệnh chạy dành cho thành viên khác

```powershell
Set-Location D:\XLA\NPR-DeepfakeDetection
$nprPython = 'D:\XLA\.venv\Scripts\python.exe'
$nprImage = (Import-Csv results\data_manifest.csv | Select-Object -First 1).image_path
& $nprPython -X utf8 inference.py $nprImage --checkpoint NPR.pth
& $nprPython -X utf8 verify_person1.py
& $nprPython -X utf8 verify_handover_extensions.py

# Chạy lại toàn bộ baseline vào thư mục riêng; dự kiến khoảng 30 phút CPU.
& $nprPython -X utf8 evaluate_checkpoint.py --manifest results\data_manifest.csv --checkpoint NPR.pth --output results\teammate_replay --batch-size 16 --threads 4
& $nprPython -X utf8 compare_predictions.py --reference results\per_image_predictions.csv --candidate results\teammate_replay\per_image_predictions.csv --output results\teammate_comparison --expect-identical
```

So sánh replay yêu cầu cùng image_id, dataset, nhãn và checkpoint hash. Chấp nhận sai số số học nhỏ giữa batch/CPU cho logit và score, nhưng nhãn phải giống. `comparison.json` ghi số ảnh thực sự được ghép. Script kiểm thử tự động là bằng chứng kỹ thuật; không thay thế việc một thành viên khác thực hiện và nhóm xác nhận nghiệm thu.

Có thể dùng `evaluation.py` với cùng các cờ `--manifest`, `--checkpoint`, `--output`, `--image-root`, `--batch-size`, `--threads`. Lệnh audit cũ `evaluation.py --audit-only` và `--reuse-audit` vẫn dùng được.

Để chỉ định thư mục dữ liệu, dùng `--data-root "progan=D:\XLA\NPR-DeepfakeDetection\datasets\progan"`; lặp cờ này cho nhiều nguồn. Chỉ nhận nhãn từ thành phần thư mục `0_real/real` và `1_fake/fake`. Nhãn không rõ/xung đột gây lỗi, không tự đoán. Manifest có tối thiểu dataset,image_path,true_label, có thể thêm image_id,sha256,eligible. Đường dẫn ảnh tương đối được tính từ `--image-root` (mặc định thư mục dự án), không tự tính từ nơi đặt CSV. CLI kiểm tra hash, verify/decode và nhãn trước khi chạy.

Mỗi output chứa `evaluation_manifest.csv`, `per_image_predictions.csv`, `baseline_metrics.csv`, cấu hình và danh sách lỗi. Output tùy chọn có `experiment_config.json` riêng. Output baseline gốc dùng cấu hình tại root và `results/person1_experiment_config.json` để giữ tương thích notebook. Khác checkpoint/tập ảnh/protocol thì dùng output mới; cache sai phiên bản bị từ chối trước khi ghi dự đoán. Chạy lại cùng lệnh để tiếp tục; không chạy đồng thời hai tiến trình vào cùng output.

## Người 2 đánh giá checkpoint fine-tune

Dùng cùng `results/data_manifest.csv`, thay giá trị --checkpoint bằng file fine-tune thực tế và chọn --output khác cho từng phiên bản/điều kiện. Không ghi đè kết quả gốc. Sau đó dùng compare_predictions.py với reference baseline và candidate mới; **không** thêm --expect-identical khi so sánh mô hình cải tiến. Bảng so sánh có Accuracy/AP/F1 và chênh lệch trên cùng image_id; không tự chứng minh mức cải thiện có ý nghĩa thống kê.

Với ảnh JPEG q90/q70/q50, giữ image_id của ảnh gốc trong từng manifest điều kiện, thay image_path và tính SHA256 của file JPEG mới (không giữ hash file gốc trong cột sha256). Các biến thể của một ảnh phải thuộc cùng split; dùng một manifest/output riêng cho từng điều kiện. Công cụ sẽ từ chối ghép hai tập khác image_id hoặc khác nhãn. Không chọn threshold/checkpoint bằng test; dùng validation riêng.

## Phạm vi và quyết định cần nhóm xác nhận

- Tập local được chốt về mặt kỹ thuật: ProGAN 8.000, StyleGAN2 15.976, sdv1 15.000; xem test_protocol.json và hash manifest.
- Baseline giữ 103 bản sao ảnh PNG đen sdv1 vì decode hợp lệ; chúng gây 103 FN. Gắn cờ thủ công trong image_quality_flags.csv. Nếu nhóm quyết định loại chúng, tạo manifest và kết quả mới cho mọi mô hình, giữ baseline lịch sử.
- Tên sdv1 chưa chứng minh generator hoặc sự tương đương sdv1_new. Archive/split benchmark chính thức chưa xác minh. Chưa tái lập đầy đủ toàn bộ Bảng 1; mới có hai nguồn GAN local và một nguồn local sdv1.
- Số paper được xác minh từ arXiv v2; PDF CVPR cuối cùng và so sánh trực tiếp vẫn Not Verified/Comparable=No như báo cáo.
- Không cần train lại nếu giảng viên chỉ yêu cầu đánh giá pretrained. Chỉ bổ sung training khi phạm vi được giảng viên yêu cầu rõ ràng.

## Phiếu nghiệm thu nhóm

Trạng thái ban đầu: **CHƯA XÁC NHẬN BỞI THÀNH VIÊN KHÁC**. Không điền sẵn người nghiệm thu hoặc giả lập kết quả của họ.

- Người chạy lại và ngày chạy: để thành viên thực hiện điền.
- Đã dùng đúng NPR.pth và hash cấu hình: chưa xác nhận.
- Đã chạy toàn bộ replay hoặc ghi rõ phạm vi kiểm tra: chưa xác nhận.
- Đã đối chiếu CSV theo image_id và công thức Accuracy/AP: chưa xác nhận.
- Nhóm thống nhất phạm vi pretrained/hai nguồn GAN/sdv1 và cách xử lý ảnh đen: chưa xác nhận.
- Giảng viên yêu cầu phạm vi nào và nhóm có đủ bằng chứng đáp ứng: chưa xác nhận.
- Kết luận cho phép chuyển sang fine-tune: do nhóm xác nhận sau khi các bước trên hoàn tất.
