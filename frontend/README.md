# Web demo NPR

Trang web cho phép tải một ảnh JPG/PNG lên, chọn mô hình và xem nhãn "Ảnh thật" hoặc "Ảnh AI" kèm điểm `fake_score`. Có thể nén lại ảnh ở JPEG q90/q70/q50 ngay trên máy chủ trước khi dự đoán, dùng đúng cách nén của thí nghiệm (lấy mẫu màu 4:2:0).

## Cấu trúc

| Tệp | Nội dung |
|---|---|
| `app.py` | Backend FastAPI. Nạp các checkpoint một lần khi khởi động và gọi `predict()` của `baseline/inference.py` |
| `static/index.html`, `static/style.css`, `static/app.js` | Giao diện: tải ảnh, xem trước, chọn mô hình và mức nén, hiển thị kết quả |
| `samples/` | 3 ảnh mẫu: ảnh thật ProGAN, ảnh giả SD v1 (PNG gốc) và cùng ảnh giả đó sau khi nén q50 |

## Chạy

Cần môi trường đã chạy được phần `baseline/` (PyTorch, torchvision, Pillow) và submodule `upstream/npr`. Cài thêm thư viện web:

```powershell
python -m pip install -r frontend/requirements.txt
```

Chạy từ thư mục gốc của repo:

```powershell
python -m uvicorn frontend.app:app --port 8000
```

Mở http://localhost:8000.

## Mô hình

| Khóa | Tên hiển thị | Checkpoint |
|---|---|---|
| `baseline` | NPR gốc | `upstream/npr/NPR.pth` (trọng số tác giả) |
| `control` | Fine-tune đối chứng | `outputs/finetune-pilot/npr_control.pth` |
| `jpeg` | Fine-tune JPEG | `outputs/finetune-pilot/npr_jpeg.pth` |
| `jpeg_autodl` | NPR + JPEG AutoDL | `experiments/autodl_jpeg/models/npr_jpeg_50ep.pth` |

Hai checkpoint fine-tune không có trong Git. Nếu đặt ở thư mục khác, khai báo bằng biến môi trường `NPR_FINETUNE_DIR` trước khi chạy. Thiếu checkpoint nào thì mô hình đó hiện "Không khả dụng", các mô hình còn lại vẫn chạy.

### Model JPEG train đủ 50 epoch trên AutoDL

Xem [bàn giao NPR + JPEG AutoDL](../experiments/autodl_jpeg/README.md). Model này
được train từ đầu trên 144.024 ảnh thuộc car/cat/chair/horse, không phải checkpoint
fine-tune hai epoch ở bảng demo cũ. SHA-256 bắt đầu bằng `d7b07f944112`.
Mã khóa API là `jpeg_autodl`, chẳng hạn `models_csv=jpeg_autodl`.
Nếu checkpoint B chưa được bổ sung vào nhánh, model sẽ hiện “Không khả dụng”.
Kiểm tra trạng thái bàn giao trong README/manifest của thư mục `autodl_jpeg`.

Model AutoDL có sẵn kiến trúc trong gói bàn giao, chạy CPU và không cần tải
submodule/checkpoint tác giả. Các model cũ tiếp tục hiện riêng nếu có đủ file.
Chạy từ gốc repo bằng môi trường PyTorch/torchvision tương thích:

```powershell
python -m pip install -r frontend/requirements.txt
python -m uvicorn frontend.app:app --port 8000
```

Mở http://localhost:8000 rồi chọn **NPR + JPEG AutoDL**. Chọn **giữ nguyên** để
không nén thêm ảnh đầu vào. Chọn q90/q70/q50 để minh họa độ bền JPEG. Máy chủ
chỉ nén một lần trước resize; model không tự nén thêm trong lần gọi đó.
Các ví dụ/điểm trong bảng demo ngày 27/11 ở dưới thuộc checkpoint pilot cũ,
không phải kết quả của model AutoDL. Ảnh SD v1 mẫu cũng không thuộc bộ 8 nguồn
ForenSynths dùng để báo cáo kết quả AutoDL.

## API

| Phương thức | Đường dẫn | Mô tả |
|---|---|---|
| GET | `/api/health` | Trạng thái máy chủ và danh sách mô hình |
| POST | `/api/predict` | Form multipart: `image` (JPG/PNG, tối đa 10 MB), `models_csv` (ví dụ `baseline,jpeg`), `jpeg_quality` (0, 90, 70 hoặc 50) |
| GET | `/api/samples` | Danh sách ảnh mẫu |

Lỗi trả về mã 413 (file quá lớn), 415 (không phải JPG/PNG), 422 (file hỏng, tham số sai) hoặc 503 (mô hình chưa nạp được), kèm thông báo tiếng Việt.

## Kịch bản demo đã kiểm tra (27/11/2026)

| Ảnh | NPR gốc | Đối chứng | Fine-tune JPEG |
|---|---|---|---|
| ProGAN thật, PNG gốc | Ảnh thật | Ảnh thật | Ảnh thật |
| SD v1 giả, PNG gốc | Ảnh AI (1,0000) | Ảnh AI | Ảnh AI |
| SD v1 giả, nén q90 trên máy chủ | Ảnh thật (0,0235) | Ảnh thật (0,0275) | Ảnh AI (0,6741) |
| SD v1 giả, nén q50 (file mẫu hoặc nén trên máy chủ) | Ảnh thật | Ảnh thật | Ảnh AI (0,9367) |

Điểm 0,9367 trùng với kết quả của thí nghiệm trong `experiments/HANDOVER.md`, xác nhận demo dùng cùng tiền xử lý và cùng cách nén với thí nghiệm.
