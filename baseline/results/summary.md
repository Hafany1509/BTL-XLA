
### Tổng kết thực nghiệm `20261111T111954519507Z`
1. **Bài toán:** phân biệt ảnh thật/ảnh sinh và kiểm tra trên generator chưa gặp khi training.
2. **Phương pháp:** normalized RGB → NPR cục bộ 2×2 × 2/3 → CNN gốc → logit → sigmoid.
3. **Pretrained:** strict load thành công; 1,437,761 tham số, 146 state entries; thiết bị cpu.
   Ảnh đơn: inference thành công.
4. **Số ảnh:** input_images thành công 1/1; evaluation 0;
   pooled unique-bytes 0. Ảnh lỗi evaluation: 0.
5. **Metrics:** Chưa có dataset hợp lệ: Accuracy/AP **Not evaluated**.
6. **Generalization:** đã đo 0 nguồn có kết quả. Chỉ suy luận từ bảng per-source thực tế;
   nếu chưa có nguồn ngoài training được xác minh thì chưa có bằng chứng generalization.
7. **Hạn chế:** score chưa hiệu chuẩn, phụ thuộc resize/JPEG và phân phối dữ liệu; kiểm tra leakage chỉ dựa hash byte;
   môi trường phần mềm có thể khác tác giả. So sánh paper chỉ có nghĩa khi cùng checkpoint/split/giao thức.
8. **Tiếp theo:** bổ sung ForenSynths/DiffusionForensics; kiểm tra provenance và split; chạy toàn bộ nguồn;
   đối chiếu Tables 1/3; bàn giao predict(), CSV, config và environment cho thành viên khác.

Kết quả của lượt này: `D:\XLA\NPR-DeepfakeDetection\results\runs\20261111T111954519507Z`. Không fine-tune, không training và không sửa thuật toán NPR.
