import os
from pycocotools.coco import COCO
import requests
import torch
import urllib.request
import zipfile

"""
This file downloads the 2017 COCO train/val annotaions file and
a few images from the classes 'cat' and 'banana'. It also downloads
the DETR model.
"""

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

def get_images(images, save_dir)->None:
    # Save the images into a local folder
    for im in images:
        img_data = requests.get(im['coco_url']).content
        with open(save_dir + im['file_name'], 'wb') as handler:
            handler.write(img_data)
    
# instantiate COCO specifying the annotations json path
# since we only need a few images for fine tuning, we can just use the val
# annotations file
split_type = 'val2017'
coco = COCO(os.path.join(ann_path, f"annotations/instances_{split_type}.json"))

# Get id values for specified classes
CLASSES = ['cat', 'banana']
cat_catIds = coco.getCatIds(catNms=CLASSES[0])
banana_catIds = coco.getCatIds(catNms=CLASSES[1])

# Get img ids and load into memory
num_samples = 100
cat_imgIds = coco.getImgIds(catIds=cat_catIds)
cat_images = coco.loadImgs(cat_imgIds)[:num_samples]
banana_imgIds = coco.getImgIds(catIds=banana_catIds)
banana_images = coco.loadImgs(banana_imgIds)[:num_samples]

print(f"Num cat images: {len(cat_images)}")
print(f"Num banana images: {len(banana_images)}")

get_images(cat_images, data_dir)
get_images(banana_image, data_dir)

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
           'model/detr-r50_no-class-head.pth')

