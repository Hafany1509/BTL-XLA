# Bài tập lớn Xử lý ảnh

Nhóm tái lập mô hình NPR để phân biệt ảnh thật và ảnh do AI tạo. Sau khi nghiệm thu bản gốc, nhóm đánh giá một bản fine-tune có bổ sung JPEG augmentation.

## Lấy mã nguồn

```powershell
git clone --recurse-submodules https://github.com/Hafany1509/BTL-XLA.git
Set-Location BTL-XLA
```

Nếu đã clone repo nhóm, chạy `git submodule update --init --recursive` trong thư mục `BTL-XLA`. Thư mục `upstream/npr/` trỏ đến mã nguồn và checkpoint của tác giả tại commit `781ced3`. Tài liệu tham khảo gồm [bài báo NPR](https://arxiv.org/abs/2312.10461) và [repo chính thức](https://github.com/chuangchuangtan/NPR-DeepfakeDetection).

## Chạy thử mô hình gốc

Dùng Python 3.10 trở lên với PyTorch và NumPy. Lệnh dưới đây chạy trên CPU với tensor RGB mẫu và kiểm tra checkpoint cùng lượt suy luận. Phạm vi đánh giá trên ảnh có nhãn được mô tả trong thư mục baseline.

```powershell
python -m pip install torch numpy
python baseline/smoke_npr.py
```

## Phân công theo thư mục

| Thư mục | Phụ trách | Công việc |
|---|---|---|
| [`baseline/`](baseline/) | Thành viên 1 | Dựng môi trường, suy luận và đánh giá NPR gốc |
| [`experiments/`](experiments/) | Thành viên 2 | Chuẩn bị dữ liệu, thí nghiệm JPEG và fine-tune sau mốc baseline |
| [`frontend/`](frontend/) | Thành viên 3 | Giao diện demo và tích hợp dự đoán |
| [`docs/`](docs/) | Thành viên 3 tổng hợp | Phân công, hướng dẫn chạy, báo cáo và slide |

Mỗi thành viên tạo nhánh từ `main` theo phần việc, chẳng hạn `baseline/reproduction`, `experiments/jpeg` hoặc `frontend/demo`. Đưa thay đổi vào `main` qua pull request. Đặt dữ liệu, checkpoint huấn luyện và đầu ra cục bộ trong `datasets/`, `checkpoints/` và `outputs/`.
