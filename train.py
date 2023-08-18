import os
from pycocotools.coco import COCO
import numpy as np

!git clone https://github.com/facebookresearch/detr.git

!python main.py \
  --dataset_file "custom" \
  --coco_path "/data/custom/" \
  --output_dir "outputs" \
  --resume "model/detr-r50_no-class-head.pth" \
  --num_classes $2 \
  --epochs 10
     

"""
data_dir = './data'
ann_type = 'val2017'
ann_file = os.path.join(
    data_dir,
    f'anns_trainval2017/annotations/instances_{ann_type}.json'
)

CLASSES = ['cat', 'banana']

coco = COCO(ann_file)

# get category and image IDs for our classes
catIds = coco.getCatIds(catNms=CLASSES);
imgIds = coco.getImgIds(catIds=catIds );

img_id = imgIds[np.random.randint(0,len(imgIds))]

"""
