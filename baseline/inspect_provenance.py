"""Inspect existing image headers for provenance evidence; never infer model from pixels."""
from collections import Counter
import json
import re
import pandas as pd
from PIL import Image
from inference import ROOT
from evaluation import RESULTS, write_json

manifest=pd.read_csv(RESULTS/'data_manifest.csv')
summaries=[]
evidence=[]
for name,g in manifest.groupby('dataset',sort=False):
    info_keys=Counter();exif_tags=Counter();formats=Counter();patterns=Counter()
    for row in g.itertuples():
        with Image.open(ROOT/row.image_path) as im:
            info_keys.update(im.info.keys())
            exif=im.getexif()
            exif_tags.update(str(k) for k in exif.keys())
            formats[im.format]+=1
            for key,value in im.info.items():
                if isinstance(value,str) and re.search(r'stable.?diffusion|sd.?v1|model.?hash|checkpoint|parameters|prompt|generator',key+' '+value,re.I):
                    evidence.append(dict(dataset=name,image_path=row.image_path,key=key,value=value[:4000]))
        patterns[re.sub(r'\d+','<number>',ROOT.joinpath(row.image_path).name)]+=1
    summaries.append(dict(dataset=name,images_inspected=len(g),image_info_keys=dict(info_keys),exif_tags=dict(exif_tags),formats=dict(formats),filename_patterns=dict(patterns.most_common(20))))
write_json(RESULTS/'provenance_header_audit.json',dict(summary=summaries,potential_generation_metadata=evidence,
    interpretation='Image metadata can be missing or edited. No metadata alone proves benchmark split. Inspect candidate metadata before claiming generator identity.'))
print(json.dumps(dict(summary=summaries,generation_metadata_candidates=len(evidence)),ensure_ascii=False,indent=2))
