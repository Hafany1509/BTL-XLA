# Kiểm tra dữ liệu thực tế

| Dataset | Path | Real | Fake | Total | Valid | Invalid | Unknown | Eligible | Formats | DuplicateFiles | CrossDatasetDuplicateFiles | Provenance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| progan | D:\XLA\NPR-DeepfakeDetection\datasets\progan | 4000 | 4000 | 8000 | 8000 | 0 | 0 | 8000 | {"PNG": 8000} | 400 | 400 | Not Verified |
| stylegan2 | D:\XLA\NPR-DeepfakeDetection\datasets\stylegan2 | 7988 | 7988 | 15976 | 15976 | 0 | 0 | 15976 | {"PNG": 15976} | 404 | 400 | Not Verified |
| sdv1 | D:\XLA\NPR-DeepfakeDetection\datasets\sdv1 | 5000 | 10000 | 15000 | 15000 | 0 | 0 | 15000 | {"PNG": 10000, "JPEG": 5000} | 105 | 0 | Not Verified |

## Kích thước ảnh

| Dataset | Distinct_sizes | Width_range | Height_range | Top_dimensions |
| --- | --- | --- | --- | --- |
| progan | 1 | 256–256 | 256–256 | 256×256: 8000 |
| stylegan2 | 2 | 256–512 | 256–384 | 256×256: 11982; 512×384: 3994 |
| sdv1 | 964 | 77–3888 | 60–5005 | 512×512: 10000; 500×375: 1577; 375×500: 463; 500×333: 302; 500×500: 174; 333×500: 97; 500×334: 87; 500×400: 70 |

Phân bố kích thước đầy đủ trong data_audit.csv (Dimensions), từng ảnh trong data_manifest.csv.

Nhãn chỉ lấy từ thành phần thư mục chính xác: 0_real/real = 0, 1_fake/fake = 1. Ảnh không rõ nhãn hoặc có nhãn xung đột không được đánh giá. Không thay đổi dữ liệu gốc. File lỗi: invalid_images.csv; cần kiểm tra nhãn: manual_review.csv; trùng byte: exact_duplicates.csv. Giữ toàn bộ đường dẫn test hợp lệ, kể cả bản trùng cùng nhãn; không gộp chỉ số giữa các nguồn. SHA256 phát hiện trùng byte, không phát hiện ảnh giống nhau sau nén/chỉnh sửa, không chứng minh không có train/test leakage.

## Nguồn dữ liệu

Không có metadata nguồn trong ba thư mục ảnh hiện tại. Tên thư mục và nội dung ảnh không đủ chứng minh generator, checkpoint sinh ảnh hoặc split. ProGAN/StyleGAN2 có cấu trúc tương thích ForenSynths nhưng chưa xác minh danh sách ảnh benchmark. sdv1 có nhãn thư mục và được đánh giá như nguồn local sdv1; không xác nhận thực sự được sinh bởi Stable Diffusion v1. Không coi sdv1 tương đương sdv1_new. DIRE có dữ liệu ảnh RGB, ảnh tái dựng và residual DIRE; NPR nhận ảnh RGB, tự tính NPR, không nhận residual DIRE. Không có metadata để chứng minh các file local là gói RGB gốc nào. README tác giả tổ chức lại benchmark dưới DiffusionForensics/sdv1_new; đổi tên thư mục không chứng minh tương đương.

datasets/test được kiểm tra riêng, không tự động trộn vào ba nguồn: 0 file.

Thời gian audit: 108.37 giây.


## Chi tiết trùng lặp

| Dataset | WithinDatasetDuplicateFiles | WithinDatasetDuplicateGroups | WithinDatasetExtraCopies | CrossDatasetDuplicateFiles |
| --- | --- | --- | --- | --- |
| progan | 0 | 0 | 0 | 400 |
| stylegan2 | 4 | 2 | 2 | 400 |
| sdv1 | 105 | 2 | 103 | 0 |

DuplicateFiles trong bảng audit chính tính các file có bản trùng ở bất kỳ nguồn nào. Bảng này tách trùng nội bộ và trùng giữa nguồn; ExtraCopies đếm số bản dư ngoài một bản cho mỗi hash. Các nguồn có ảnh trùng không độc lập hoàn toàn.

Đã kiểm tra header/EXIF của 38976 ảnh; có 0 metadata ứng viên liên quan generator/prompt/checkpoint. Chi tiết ở provenance_header_audit.json. Không có metadata xác minh generator trong lượt này; tên file sdv1 gồm hậu tố real/ILSVRC và fake không đủ xác minh nguồn sinh ảnh.


### Cờ chất lượng nội dung

| dataset | flag | true_label | files | unique_hashes | errors |
| --- | --- | --- | --- | --- | --- |
| sdv1 | all_black | 1 | 103 | 1 | 103 |

Valid/Invalid trong bảng audit là khả năng verify/decode ảnh. Ảnh đơn sắc đọc được vẫn thuộc baseline chính nhưng cần kiểm tra thủ công về chất lượng nội dung; xem image_quality_audit.md và image_quality_flags.csv. Không sửa, đổi nhãn hoặc xóa các ảnh này.
