"""
This file downloads a small portion of  the 'cat' and 'banana' classes
from the COCO dataset for fine tuning of the DETE model
"""
import os
import torch
import urllib.request
import zipfile

#create data directory
os.makedirs('data', exist_ok=True)

ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
ann_path = 'data/anns_trainval2017'

#download annotations zip
if os.path.isfile(f"{ann_path}.zip"):
    print(f"Annotations found at {ann_path}.zip")
else:
    print(f"Downloading annotations to {ann_path}.zip...")
    urllib.request.urlretrieve(ann_url, f"{ann_path}.zip")

#extract zip
if os.path.isdir(f"{ann_path}"):
    print(f"Extracted annotations found at {ann_path}")
else:
    print(f"Extracting annotations to {ann_path}")
    #unzip file
    with zipfile.ZipFile(f"{ann_path}.zip", 'r') as zip_ref:
        zip_ref.extractall(f"{ann_path}")

#download pretrained weights
os.makedirs('model', exist_ok=True)

# Get pretrained weights
checkpoint = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
            map_location='cpu',
            check_hash=True)

# Remove class weights
del checkpoint["model"]["class_embed.weight"]
del checkpoint["model"]["class_embed.bias"]

# Save
torch.save(checkpoint,
           'detr-r50_no-class-head.pth')


# instantiate COCO specifying the annotations json path
#coco_train = COCO(f"{ann_path}/annotations/instances_train2017.json")
coco_val = COCO(f"{ann_path}/annotations/instances_val2017.json")

# Specify a list of category names of interest
catIds = coco_val.getCatIds(catNms=['cat', 'banana'])
print(catIds)

# Get the corresponding image ids and images using loadImgs
#imgIds = coco.getImgIds(catIds=catIds)
#images = coco.loadImgs(imgIds)

