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
| `model_a` | Mô hình A | `outputs/main-experiment/npr_A.pth` (SHA-256 `84b1c6d4…6fec717ec9`) |
| `model_b` | Mô hình B | `outputs/main-experiment/npr_B_jpeg.pth` (SHA-256 `d7b07f94…f15f6354`) |

A và B là hai mô hình nhóm huấn luyện từ đầu trong thí nghiệm chính (báo cáo, Mục 5.6), cùng cấu hình, chỉ khác ở bước tăng cường nén JPEG của B. Hai checkpoint không có trong Git. Nếu đặt ở thư mục khác, khai báo bằng biến môi trường `NPR_MODEL_DIR` trước khi chạy. Thiếu checkpoint nào thì mô hình đó hiện "Không khả dụng", các mô hình còn lại vẫn chạy.

## API

| Phương thức | Đường dẫn | Mô tả |
|---|---|---|
| GET | `/api/health` | Trạng thái máy chủ và danh sách mô hình |
| POST | `/api/predict` | Form multipart: `image` (JPG/PNG, tối đa 10 MB), `models_csv` (ví dụ `model_a,model_b`), `jpeg_quality` (0, 90, 70 hoặc 50) |
| GET | `/api/samples` | Danh sách ảnh mẫu |

Lỗi trả về mã 413 (file quá lớn), 415 (không phải JPG/PNG), 422 (file hỏng, tham số sai) hoặc 503 (mô hình chưa nạp được), kèm thông báo tiếng Việt.

## Kịch bản demo đã kiểm tra (28/11/2026)

fake_score theo thứ tự NPR gốc / Mô hình A / Mô hình B:

| Ảnh | Giữ nguyên | Nén q90 trên máy chủ | Nén q50 trên máy chủ |
|---|---|---|---|
| ProGAN thật (`progan_real_goc.png`) | 0,00 / 0,00 / 0,00 | 0,00 / 0,00 / 0,00 | 0,00 / 0,00 / 0,05 |
| ProGAN giả (`progan_fake_goc.png`) | 1,00 / 1,00 / 1,00 | 0,00 / 0,00 / **1,00** | 0,00 / 0,00 / **1,00** |
| StyleGAN2 giả (`stylegan2_fake_goc.png`) | 1,00 / 1,00 / 1,00 | 0,00 / 0,00 / **0,95** | 0,00 / 0,00 / 0,01 |
| SD v1 giả (`sdv1_fake_goc.png`) | 1,00 / 1,00 / 0,00 | 0,02 / 0,00 / 0,00 | 0,00 / 0,00 / 0,00 |

Ảnh ProGAN giả ở q90 là kịch bản demo chính: A báo ảnh thật, B vẫn báo ảnh AI. Với ảnh Stable Diffusion, thử trên 6 ảnh SD v1 (ảnh mẫu và `outputs/demo-testset/anh_ai/sdv1_*.png`), B nhận đúng 4/6 khi chưa nén nhưng bỏ sót cả 6 khi nén q90. Tập kiểm thử 8 mô hình sinh của thí nghiệm chính không có mô hình khuếch tán, nên điểm yếu này không hiện trong Bảng 5.1. Không dùng ảnh SD v1 cho demo.
