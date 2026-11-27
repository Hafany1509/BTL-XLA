"""Flag constant-color images without deleting or silently excluding decoded images."""
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from PIL import Image
from tqdm import tqdm
from inference import ROOT
from evaluation import RESULTS, table_md


def inspect(row):
    with Image.open(ROOT/row.image_path) as im:
        extrema=im.convert('RGB').getextrema()
    constant=all(lo==hi for lo,hi in extrema)
    if not constant:return None
    return dict(dataset=row.dataset,image_path=row.image_path,true_label=int(row.true_label),sha256=row.sha256,
                flag='all_black' if all(hi==0 for lo,hi in extrema) else 'constant_color',
                rgb_value=str([lo for lo,hi in extrema]),action='manual_review; retained in primary all-decoded-valid baseline')


manifest=pd.read_csv(RESULTS/'data_manifest.csv')
rows=list(manifest[manifest.eligible].itertuples())
with ThreadPoolExecutor(max_workers=4) as pool:
    flags=[x for x in tqdm(pool.map(inspect,rows),total=len(rows),desc='Image quality audit') if x is not None]
flag_frame=pd.DataFrame(flags,columns=['dataset','image_path','true_label','sha256','flag','rgb_value','action'])
flag_frame.to_csv(RESULTS/'image_quality_flags.csv',index=False)
pred=pd.read_csv(RESULTS/'per_image_predictions.csv')
joined=flag_frame.merge(pred[['image_path','predicted_label','fake_score']],on='image_path',validate='one_to_one')
joined['misclassified']=joined.true_label!=joined.predicted_label
joined.to_csv(RESULTS/'image_quality_predictions.csv',index=False)
summary=joined.groupby(['dataset','flag','true_label'],as_index=False).agg(files=('image_path','size'),unique_hashes=('sha256','nunique'),errors=('misclassified','sum'))
summary.to_csv(RESULTS/'image_quality_summary.csv',index=False)
text=('# Kiểm tra chất lượng nội dung ảnh\n\n'
      f'Đã decode và kiểm tra extrema RGB của toàn bộ {len(rows)} ảnh đủ điều kiện. '
      'Gắn cờ ảnh có mọi pixel cùng giá trị trên từng kênh RGB; đây là kiểm tra ảnh đơn sắc, không phải bộ phát hiện mọi loại lỗi chất lượng.\n\n'+table_md(summary)+
      '\n\nCác ảnh này đọc được, có nhãn thư mục rõ ràng và được giữ trong baseline chính theo giao thức toàn bộ ảnh decode hợp lệ. '
      'Chúng được đưa vào danh sách cần kiểm tra thủ công, không đổi nhãn hoặc xóa file. '
      'Không suy đoán chúng là lỗi sinh ảnh, lọc nội dung hay lỗi đóng gói khi chưa có metadata nguồn. '
      'Không dùng các bản sao ảnh đen để kết luận về khả năng phát hiện ảnh diffusion thực tế. '
      'Nếu nhóm quyết định loại ảnh này sau kiểm tra nguồn, tạo manifest và lượt đánh giá mới, áp dụng giống nhau cho cả hai mô hình; giữ baseline gốc để đối chiếu.\n')
(RESULTS/'image_quality_audit.md').write_text(text,encoding='utf-8')
print(text)
