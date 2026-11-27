# Kiểm tra chất lượng nội dung ảnh

Đã decode và kiểm tra extrema RGB của toàn bộ 38976 ảnh đủ điều kiện. Gắn cờ ảnh có mọi pixel cùng giá trị trên từng kênh RGB; đây là kiểm tra ảnh đơn sắc, không phải bộ phát hiện mọi loại lỗi chất lượng.

| dataset | flag | true_label | files | unique_hashes | errors |
| --- | --- | --- | --- | --- | --- |
| sdv1 | all_black | 1 | 103 | 1 | 103 |

Các ảnh này đọc được, có nhãn thư mục rõ ràng và được giữ trong baseline chính theo giao thức toàn bộ ảnh decode hợp lệ. Chúng được đưa vào danh sách cần kiểm tra thủ công, không đổi nhãn hoặc xóa file. Không suy đoán chúng là lỗi sinh ảnh, lọc nội dung hay lỗi đóng gói khi chưa có metadata nguồn. Không dùng các bản sao ảnh đen để kết luận về khả năng phát hiện ảnh diffusion thực tế. Nếu nhóm quyết định loại ảnh này sau kiểm tra nguồn, tạo manifest và lượt đánh giá mới, áp dụng giống nhau cho cả hai mô hình; giữ baseline gốc để đối chiếu.
