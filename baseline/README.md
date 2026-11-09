# Tái lập NPR gốc

Thành viên 1 phụ trách nạp checkpoint, tiền xử lý, suy luận và đánh giá. Thêm mã của nhóm tại đây. Dùng `upstream/npr/` làm mốc đối chiếu.

- Chạy `python baseline/smoke_npr.py` từ thư mục gốc để kiểm tra nạp checkpoint và forward trên CPU.
- Chốt checkpoint, bộ dữ liệu và protocol đánh giá theo phạm vi giảng viên yêu cầu. Xử lý đường dẫn dữ liệu và chọn CPU/CUDA trong luồng đánh giá của nhóm.
- Bàn giao lệnh chạy, cấu hình và CSV theo từng ảnh. Tổng hợp Accuracy tại ngưỡng 0,5 và Average Precision (AP) theo nguồn.
- Thống nhất hàm `predict(image, model)` với frontend. Kết quả dự kiến gồm `label`, `fake_score` và `model_version`.

Tái lập được nghiệm thu khi thành viên khác chạy lại theo hướng dẫn trên cùng bộ ảnh đã chốt. Nếu giảng viên yêu cầu huấn luyện lại hoặc tái lập bảng cụ thể, bổ sung kết quả tương ứng trước khi chuyển giai đoạn.
