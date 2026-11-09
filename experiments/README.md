# Thí nghiệm và cải tiến NPR

Thành viên 2 phụ trách dữ liệu và huấn luyện.

1. Trong giai đoạn tái lập, chuẩn bị danh sách ảnh có nhãn, phân chia train/validation/test theo ảnh gốc và hỗ trợ kiểm tra tập test. Chuẩn bị script tạo bản JPEG ở quality 90, 70 và 50.
2. Sau khi nhóm nghiệm thu NPR gốc, fine-tune hai bản từ cùng checkpoint và cùng cấu hình. Dùng JPEG augmentation cho bản cải tiến và fine-tune thông thường cho bản đối chứng.
3. Dùng cùng công cụ đánh giá để đo NPR gốc, bản đối chứng và bản cải tiến trên ảnh gốc và từng mức nén. Bàn giao cấu hình, log, bảng Accuracy/AP và thông tin checkpoint cho frontend.
