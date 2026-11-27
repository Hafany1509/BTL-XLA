"""Figures, verified paper references and error analysis from saved real predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, roc_curve,
                             precision_recall_curve, average_precision_score, roc_auc_score)
from inference import ROOT, read_rgb
from evaluation import RESULTS, DATASETS, table_md

FIGURES=RESULTS/'figures'
PAPER_URL='https://arxiv.org/pdf/2312.10461v2'


def save(fig,name):
    fig.tight_layout()
    fig.savefig(FIGURES/name,dpi=160,bbox_inches='tight')
    plt.close(fig)


def gallery(group,name,kind,scope_label=''):
    if kind=='correct':
        selected=group[group.true_label==group.predicted_label].sort_values('image_path').head(6)
    elif kind=='incorrect':
        selected=group[group.true_label!=group.predicted_label].sort_values('fake_score',key=lambda s:(s-.5).abs(),ascending=False).head(6)
    else:
        selected=group.assign(distance=(group.fake_score-.5).abs()).sort_values(['distance','image_path']).head(6)
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for ax in axes.flat:ax.axis('off')
    for ax,row in zip(axes.flat,selected.itertuples()):
        with Image.open(ROOT/row.image_path) as im:ax.imshow(im.convert('RGB'))
        truth='Fake' if row.true_label else 'Real'
        predicted='Fake' if row.predicted_label else 'Real'
        ax.set_title(f'True: {truth} / Pred: {predicted}\nScore: {row.fake_score:.6f}',fontsize=10)
    if selected.empty:
        axes[0,0].text(.5,.5,'No matching samples',ha='center',transform=axes[0,0].transAxes)
    scope = scope_label or 'local dataset'
    fig.suptitle(f'{name} — {kind} samples ({scope})')
    save(fig,f'{name}_{kind}_samples.png')
    selected.drop(columns=['distance'],errors='ignore').to_csv(RESULTS/f'{name}_{kind}_samples.csv',index=False)


def figures(predictions,metrics,scope_label=''):
    FIGURES.mkdir(exist_ok=True)
    for name in DATASETS:
        g=predictions[predictions.dataset==name]
        if g.empty:continue
        display_name = f'{name} — {scope_label}' if scope_label else name
        y=g.true_label;s=g.fake_score
        fig,ax=plt.subplots(figsize=(5,4.5))
        ConfusionMatrixDisplay(confusion_matrix(y,g.predicted_label,labels=[0,1]),display_labels=['Real','Fake']).plot(ax=ax,colorbar=False,cmap='Blues',values_format='d')
        ax.set_title(f'{display_name} — threshold > 0.5')
        save(fig,f'{name}_confusion_matrix.png')
        if y.nunique()==2:
            fpr,tpr,_=roc_curve(y,s)
            fig,ax=plt.subplots(figsize=(5.5,4.5))
            ax.plot(fpr,tpr,label=f'AUC = {roc_auc_score(y,s):.5f}')
            ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='False Positive Rate',ylabel='True Positive Rate',title=f'{display_name} — ROC (Fake positive)')
            ax.legend();save(fig,f'{name}_roc_curve.png')
            precision,recall,_=precision_recall_curve(y,s)
            fig,ax=plt.subplots(figsize=(5.5,4.5))
            ax.step(recall,precision,where='post',label=f'AP = {average_precision_score(y,s):.5f}')
            ax.axhline(y.mean(),ls='--',color='gray',label=f'Fake prevalence = {y.mean():.3f}')
            ax.set(xlabel='Recall',ylabel='Precision',title=f'{display_name} — Precision–Recall',ylim=(-.02,1.02));ax.legend()
            save(fig,f'{name}_pr_curve.png')
        fig,ax=plt.subplots(figsize=(6,4))
        for label,title,color in [(0,'Real','tab:blue'),(1,'Fake','tab:orange')]:
            ax.hist(s[y==label],bins=np.linspace(0,1,51),alpha=.55,label=f'{title} (n={(y==label).sum()})',color=color)
        ax.axvline(.5,color='black',ls='--');ax.set(xlabel='Fake score (sigmoid)',ylabel='Images',title=f'{display_name} — Fake score distribution');ax.legend()
        save(fig,f'{name}_fake_score_distribution.png')
        for kind in ['correct','incorrect','near_threshold']:gallery(g,name,kind,scope_label)
    fig,ax=plt.subplots(figsize=(8,4.5))
    x=np.arange(len(metrics));width=.24
    for i,col in enumerate(['Accuracy','AP','F1']):ax.bar(x+(i-1)*width,metrics[col]*100,width,label=col)
    title = f'NPR original checkpoint — {scope_label}' if scope_label else 'NPR original checkpoint — full valid local test sets'
    ax.set_xticks(x,metrics.Dataset);ax.set(ylabel='Percent (%)',title=title,ylim=(0,110));ax.legend(ncols=3)
    save(fig,'baseline_comparison.png')


def model_visuals(detector,manifest):
    FIGURES.mkdir(exist_ok=True)
    row=manifest[manifest.eligible].iloc[0]
    im=read_rgb(ROOT/row.image_path);x=detector.preprocess(im).unsqueeze(0)
    import torch
    with torch.inference_mode():residual=(x-detector.model.interpolate(x,.5))[0].numpy()
    fig,axes=plt.subplots(1,4,figsize=(14,4))
    axes[0].imshow(im);axes[0].set_title('Original RGB')
    scale=max(float(np.percentile(np.abs(residual),99)),1e-8)
    for i,ch in enumerate('RGB'):
        axes[i+1].imshow(residual[i],cmap='seismic',vmin=-scale,vmax=scale);axes[i+1].set_title(f'NPR {ch} (normalized)')
    for ax in axes:ax.axis('off')
    fig.suptitle('Visualization only: residual is NOT fed back into model')
    save(fig,'npr_visualization.png')
    fig,ax=plt.subplots(figsize=(14,3))
    labels=['RGB\n256×256','Normalize\nImageNet','NPR × 2/3\ninside forward','Conv/BN/ReLU\nMaxPool','Bottleneck\n3 + 4 blocks','AvgPool\nLinear 512→1','Sigmoid\nFake > 0.5']
    for i,label in enumerate(labels):
        ax.text(i,.5,label,ha='center',va='center',bbox=dict(boxstyle='round,pad=.5',fc='#e9f2ff',ec='#4274a7'),fontsize=9)
        if i<len(labels)-1:ax.annotate('',xy=(i+.63,.5),xytext=(i+.38,.5),arrowprops=dict(arrowstyle='->'))
    ax.set(xlim=(-.6,6.6),ylim=(0,1));ax.axis('off');ax.set_title(f'Original custom NPR network — {detector.parameter_count:,} parameters')
    save(fig,'npr_architecture.png')


def paper_comparison(metrics):
    references={'progan':(99.8,100.,'Table 1'),'stylegan2':(97.3,100.,'Table 1'),'sdv1':(97.4,99.8,'Table 3')}
    rows=[]
    for r in metrics.itertuples():
        acc,ap,table=references[r.Dataset]
        reason='Local source archive and exact test split not verified; original NPR.pth matches repository, protocol matches local author source; environment differs.'
        if r.Dataset=='sdv1':reason+=' Generator identity and sdv1_new equivalence unverified; local ImageNet-only source is imbalanced (1:2).'
        rows.append(dict(Dataset=r.Dataset,Paper_Accuracy=acc,Our_Accuracy=r.Accuracy*100,Paper_AP=ap,Our_AP=r.AP*100,
                         Accuracy_Difference_pp=r.Accuracy*100-acc,AP_Difference_pp=r.AP*100-ap,
                         Comparable='No',Reason=reason,Paper_Table=table,Source=PAPER_URL,
                         Paper_version='arXiv v2 (2023-12-20); CVPR final PDF not retrieved',
                         CVPR_final_verification='Not Verified',Full_local_evaluation=bool(r.Complete)))
    frame=pd.DataFrame(rows)
    frame.to_csv(RESULTS/'paper_comparison.csv',index=False)
    text=('# Đối chiếu paper NPR\n\n'+table_md(frame[['Dataset','Paper_Accuracy','Our_Accuracy','Paper_AP','Our_AP','Accuracy_Difference_pp','AP_Difference_pp','Comparable']])+
          '\n\nĐơn vị: phần trăm; chênh lệch là điểm phần trăm, không phải phần trăm tương đối. '
          'Đây là chênh lệch mô tả, không phải bằng chứng tái lập benchmark.\n\n'
          f'Nguồn số liệu đã đọc: [bản tác giả arXiv v2, trang 6, Tables 1 và 3]({PAPER_URL}). '
          'Table 1: NPR ProGAN Acc/AP 99.8/100.0; StyleGAN2 97.3/100.0. '
          'Table 3: NPR Stable Diffusion v1 97.4/99.8. PDF CVPR cuối cùng không truy cập được qua công cụ trong phiên; '
          'không tuyên bố đã xác minh bản CVPR cuối cùng.\n\n'
          'Checkpoint NPR.pth được xác nhận trùng Git blob của repository local; không đổi sang model_epoch_last_3090.pth. '
          'Tiền xử lý khớp test.py local cho ForenSynths/DiffusionForensics. '
          'Chưa có manifest chính thức để so sánh toàn bộ test split/ảnh và chưa xác minh nguồn archive local; '
          'mọi hàng Comparable=No. Chạy toàn bộ ảnh local không đồng nghĩa đã chạy toàn bộ benchmark paper.\n\n'
          'NPR README chỉ rõ sdv1_new trong DiffusionForensics. DIRE gốc tổ chức images/recons/dire theo split và domain; '
          'NPR sắp xếp theo generator/nhãn. Các tên này không chứng minh hai tập local tương đương. '
          'Tham khảo [NPR repository](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) và '
          '[DIRE dataset layout](https://github.com/ZhendongWang6/DIRE#diffusionforensics-dataset).\n\n'+
          '\n\n'.join(f'**{r.Dataset}:** {r.Reason}' for r in frame.itertuples()))
    (RESULTS/'paper_comparison.md').write_text(text,encoding='utf-8')
    return frame


def analysis(predictions,metrics):
    category=[]
    merged=predictions.merge(pd.read_csv(RESULTS/'data_manifest.csv')[['image_path','category','width','height','format']],on='image_path',validate='one_to_one')
    for (name,cat),g in merged.groupby(['dataset','category']):
        category.append(dict(dataset=name,category=cat,total=len(g),errors=int((g.true_label!=g.predicted_label).sum()),accuracy=float((g.true_label==g.predicted_label).mean())))
    bycat=pd.DataFrame(category);bycat.to_csv(RESULTS/'category_metrics.csv',index=False)
    errors=merged[merged.true_label!=merged.predicted_label].copy()
    errors['distance_to_threshold']=(errors.fake_score-.5).abs()
    pixel_stats=[]
    for row in errors.itertuples():
        rgb=np.asarray(read_rgb(ROOT/row.image_path).resize((256,256),resample=Image.Resampling.BILINEAR),dtype=np.float32)/255
        channel_spread=rgb.max(axis=2)-rgb.min(axis=2)
        pixel_stats.append(dict(image_path=row.image_path,mean_channel_spread=float(channel_spread.mean()),
                                near_grayscale=bool(channel_spread.mean()<1/255),
                                mean_brightness=float(rgb.mean()),pixel_std=float(rgb.std())))
    if pixel_stats:
        errors=errors.merge(pd.DataFrame(pixel_stats),on='image_path',validate='one_to_one')
    errors.sort_values('distance_to_threshold',ascending=False).to_csv(RESULTS/'error_analysis.csv',index=False)
    best=metrics.loc[metrics.Accuracy.idxmax()];worst=metrics.loc[metrics.Accuracy.idxmin()]
    text='# Phân tích lỗi trên dữ liệu thực tế\n\n'+table_md(metrics)
    text+=f'\n\nTrong ba nguồn local, Accuracy cao nhất: {best.Dataset} ({best.Accuracy:.4%}); thấp nhất: {worst.Dataset} ({worst.Accuracy:.4%}). '
    gan=metrics[metrics.Dataset.isin(['progan','stylegan2'])].Accuracy.mean();sd=metrics.loc[metrics.Dataset=='sdv1','Accuracy'].iloc[0]
    text+=f'Trung bình Accuracy hai nguồn GAN (không trọng số): {gan:.4%}; sdv1 local: {sd:.4%}; chênh lệch sdv1 − GAN: {(sd-gan)*100:.4f} điểm %. '
    text+=('Đây chỉ là quan sát trên các tập local khác nội dung và tỷ lệ lớp; không chứng minh tác động nhân quả của GAN/Diffusion, '
           'đặc biệt danh tính nguồn sdv1 chưa được xác minh.\n\n')
    for r in metrics.itertuples():
        text+=f'- {r.Dataset}: FP={r.FP}, FN={r.FN}; Real={r.Real}, Fake={r.Fake}; Recall Fake={r.Recall:.4%}.\n'
        if r.FN>r.FP:
            text+=f'  Trong các lỗi của nguồn này, bỏ sót Fake (FN) chiếm {r.FN/(r.FN+r.FP):.2%}; cần xem false negatives khi nghiên cứu cải tiến.\n'
    text+='\n## Theo danh mục\n\n'+table_md(bycat)
    quality_path=RESULTS/'image_quality_summary.csv'
    if quality_path.exists():
        quality=pd.read_csv(quality_path)
        text+='\n\n## Ảnh đơn sắc và lỗi lặp lại\n\n'+table_md(quality)
        text+=('\n\nCác nhóm ảnh đơn sắc có thể lặp cùng một nội dung và gây nhiều lỗi đếm theo đường dẫn. '
               'Số lỗi trong bảng không đồng nghĩa có bấy nhiêu tình huống thị giác độc lập. Kết quả chính vẫn tính mọi đường dẫn decode hợp lệ; '
               'xem image_quality_audit.md và image_quality_flags.csv để kiểm tra thủ công. '
               'Không coi sai sót trên ảnh đen là bằng chứng riêng về điểm yếu trên ảnh diffusion chân thực.\n')
    if pixel_stats:
        text+='\n\n## Quan sát đặc điểm các ảnh sai\n\n'
        for name,g in errors.groupby('dataset'):
            gray=int(g.near_grayscale.sum())
            text+=f'- {name}: {gray}/{len(g)} ảnh sai có độ lệch trung bình giữa kênh RGB < 1/255 sau resize (gần ảnh xám).\n'
        text+=('\nĐây là thống kê trong nhóm dự đoán sai, chưa phải tỷ lệ lỗi của toàn bộ ảnh xám. '
               'Không suy ra ảnh xám gây lỗi khi chưa có nhóm đối chứng đầy đủ. Xem gallery và từng dòng CSV để kiểm tra trực tiếp.\n')
    text+=('\n\n## Gợi ý cho Người 2\n\n'
           'Xem error_analysis.csv và gallery incorrect để kiểm tra các lỗi tự tin cao; near_threshold để xem mẫu không chắc chắn. '
           'Số lỗi theo danh mục ở trên là bằng chứng trực tiếp; không suy diễn đặc điểm thị giác gây lỗi chỉ từ tên file. '
           'Kiểm tra robustness với resize/JPEG hoặc phân tích từng domain là hướng nghiên cứu tiếp theo, chưa được đo ở baseline này. '
           'Không chọn threshold từ tập test để báo cáo cải tiến; dùng tập validation riêng và báo cáo cùng manifest test. '
           'Do tỷ lệ Fake của sdv1 là 2/3, AP cần diễn giải cùng prevalence; không gộp các nguồn rồi tuyên bố generalization. '
           'Kiểm tra duplicate/leakage và xác minh archive trước khi kết luận tái lập paper.\n')
    (RESULTS/'error_analysis.md').write_text(text,encoding='utf-8')
    return text


def duplicate_summary():
    manifest=pd.read_csv(RESULTS/'data_manifest.csv')
    rows=[]
    for name,g in manifest.groupby('dataset',sort=False):
        within=g[g.sha256.ne('') & g.duplicated('sha256',keep=False)]
        all_cross=manifest.groupby('sha256').dataset.nunique()
        cross_hash=set(all_cross[all_cross>1].index)
        rows.append(dict(Dataset=name,WithinDatasetDuplicateFiles=len(within),
                         WithinDatasetDuplicateGroups=within.sha256.nunique(),
                         WithinDatasetExtraCopies=len(within)-within.sha256.nunique(),
                         CrossDatasetDuplicateFiles=int(g.sha256.isin(cross_hash).sum())))
    summary=pd.DataFrame(rows)
    summary.to_csv(RESULTS/'duplicate_summary.csv',index=False)
    path=RESULTS/'data_audit.md'
    text=path.read_text(encoding='utf-8').split('\n\n## Chi tiết trùng lặp')[0]
    dimension_rows=[]
    for name,g in manifest.groupby('dataset',sort=False):
        dimensions=g.groupby(['width','height']).size().sort_values(ascending=False)
        dimension_rows.append(dict(Dataset=name,Distinct_sizes=len(dimensions),
                                   Width_range=f'{int(g.width.min())}–{int(g.width.max())}',
                                   Height_range=f'{int(g.height.min())}–{int(g.height.max())}',
                                   Top_dimensions='; '.join(f'{int(w)}×{int(h)}: {n}' for (w,h),n in dimensions.head(8).items())))
    marker='\n\n## Kích thước ảnh'
    end_marker='\n\nNhãn chỉ lấy'
    if marker in text and end_marker in text:
        start=text.index(marker);end=text.index(end_marker,start)
        text=(text[:start]+marker+'\n\n'+table_md(pd.DataFrame(dimension_rows))+
              '\n\nPhân bố kích thước đầy đủ trong data_audit.csv (Dimensions), từng ảnh trong data_manifest.csv.'+text[end:])
    text+='\n\n## Chi tiết trùng lặp\n\n'+table_md(summary)
    text+='\n\nDuplicateFiles trong bảng audit chính tính các file có bản trùng ở bất kỳ nguồn nào. Bảng này tách trùng nội bộ và trùng giữa nguồn; ExtraCopies đếm số bản dư ngoài một bản cho mỗi hash. Các nguồn có ảnh trùng không độc lập hoàn toàn.\n'
    provenance_path=RESULTS/'provenance_header_audit.json'
    if provenance_path.exists():
        provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
        inspected=sum(r['images_inspected'] for r in provenance['summary'])
        candidates=len(provenance['potential_generation_metadata'])
        text+=(f'\nĐã kiểm tra header/EXIF của {inspected} ảnh; có {candidates} metadata ứng viên liên quan generator/prompt/checkpoint. '
               'Chi tiết ở provenance_header_audit.json. Không có metadata xác minh generator trong lượt này; '
               'tên file sdv1 gồm hậu tố real/ILSVRC và fake không đủ xác minh nguồn sinh ảnh.\n')
    quality_path=RESULTS/'image_quality_summary.csv'
    if quality_path.exists():
        text+='\n\n### Cờ chất lượng nội dung\n\n'+table_md(pd.read_csv(quality_path))
        text+=('\n\nValid/Invalid trong bảng audit là khả năng verify/decode ảnh. Ảnh đơn sắc đọc được vẫn thuộc baseline chính '
               'nhưng cần kiểm tra thủ công về chất lượng nội dung; xem image_quality_audit.md và image_quality_flags.csv. '
               'Không sửa, đổi nhãn hoặc xóa các ảnh này.\n')
    path.write_text(text,encoding='utf-8')
    return summary


def generate_reports(detector=None):
    predictions=pd.read_csv(RESULTS/'per_image_predictions.csv')
    metrics=pd.read_csv(RESULTS/'baseline_metrics.csv')
    figures(predictions,metrics)
    duplicate_summary()
    paper_comparison(metrics)
    analysis(predictions,metrics)
    if detector is not None:model_visuals(detector,pd.read_csv(RESULTS/'data_manifest.csv'))
    return metrics


if __name__=='__main__':
    from inference import load_model
    generate_reports(load_model())
