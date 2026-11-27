# Đối chiếu paper NPR

| Dataset | Paper_Accuracy | Our_Accuracy | Paper_AP | Our_AP | Accuracy_Difference_pp | AP_Difference_pp | Comparable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| progan | 99.8 | 99.7875 | 100.0 | 99.99163052795261 | -0.012500000000002842 | -0.008369472047391469 | No |
| stylegan2 | 97.3 | 98.39133700550826 | 100.0 | 99.93641769193573 | 1.0913370055082652 | -0.06358230806426945 | No |
| sdv1 | 97.4 | 96.89333333333333 | 99.8 | 99.58196328519983 | -0.5066666666666748 | -0.21803671480016362 | No |

Đơn vị: phần trăm; chênh lệch là điểm phần trăm, không phải phần trăm tương đối. Đây là chênh lệch mô tả, không phải bằng chứng tái lập benchmark.

Nguồn số liệu đã đọc: [bản tác giả arXiv v2, trang 6, Tables 1 và 3](https://arxiv.org/pdf/2312.10461v2). Table 1: NPR ProGAN Acc/AP 99.8/100.0; StyleGAN2 97.3/100.0. Table 3: NPR Stable Diffusion v1 97.4/99.8. PDF CVPR cuối cùng không truy cập được qua công cụ trong phiên; không tuyên bố đã xác minh bản CVPR cuối cùng.

Checkpoint NPR.pth được xác nhận trùng Git blob của repository local; không đổi sang model_epoch_last_3090.pth. Tiền xử lý khớp test.py local cho ForenSynths/DiffusionForensics. Chưa có manifest chính thức để so sánh toàn bộ test split/ảnh và chưa xác minh nguồn archive local; mọi hàng Comparable=No. Chạy toàn bộ ảnh local không đồng nghĩa đã chạy toàn bộ benchmark paper.

NPR README chỉ rõ sdv1_new trong DiffusionForensics. DIRE gốc tổ chức images/recons/dire theo split và domain; NPR sắp xếp theo generator/nhãn. Các tên này không chứng minh hai tập local tương đương. Tham khảo [NPR repository](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) và [DIRE dataset layout](https://github.com/ZhendongWang6/DIRE#diffusionforensics-dataset).

**progan:** Local source archive and exact test split not verified; original NPR.pth matches repository, protocol matches local author source; environment differs.

**stylegan2:** Local source archive and exact test split not verified; original NPR.pth matches repository, protocol matches local author source; environment differs.

**sdv1:** Local source archive and exact test split not verified; original NPR.pth matches repository, protocol matches local author source; environment differs. Generator identity and sdv1_new equivalence unverified; local ImageNet-only source is imbalanced (1:2).