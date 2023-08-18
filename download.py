import os
import json
from pycocotools.coco import COCO
import requests
import torch
import urllib.request
import zipfile
from tqdm import tqdm

"""
This file downloads the 2017 COCO train/val annotaions file and
a few images from the classes 'cat' and 'banana'. It also downloads
the DETR model.
"""
def get_archive(url:str, save_path:str)->None:
    """
    Downloads the archive pointed to by the url

    Arguments:
        url: str, url to send request
        save_path: str, directory path to save zip file

    Returns:
        None
    """
    if os.path.isfile(f"{save_path}.zip"):
        print(f"Annotations found at {save_path}.zip")
    else:
        print(f"Downloading annotations to {save_path}.zip...")
        archive = requests.get(ann_url, f"{save_path}.zip").content
        with open(f"{save_path}.zip", 'wb') as handler:
            handler.write(archive)

def extract_json(archive_path:str, split_type:str)->None:
    """
    Extracts one split member from the zip file 

    Arguments:
        archive_path: str, path to coco zip file
        split_type: str, type of annotation file (train2017, val2017)

    Returns:
        None
    """
    ann_file = f"annotations/instances_{split_type}.json"
    ann_path = os.path.join(archive_path, ann_file)
    
    if os.path.isfile(ann_path):
        print(f"Extracted annotations found at {ann_path}")
    else:
        print(f"Extracting annotations to {ann_path}")
        with zipfile.ZipFile(f"{archive_path}.zip", 'r') as zip_ref:
            zip_ref.extract(ann_file, f"{archive_path}")

def get_images(imgs:list, coco_dir:str, split:str)->None:
    """
    Downloads images to local folder

    Arguments:
        imgs: list, list of images in coco format
        coco_dir: str, path to location where coco annotations are stored
        split_type: str, type of annotation file (train2017, val2017)

    Returns:
        None
    """
    img_dir = os.path.join(coco_dir, split)
    os.makedirs(img_dir, exist_ok=True)
    for im in tqdm(imgs, desc=f"Downloading {split} images"):
        img_name = os.path.join(img_dir, im['file_name'])
        if not os.path.isfile(img_name):
            # request image 
            img_data = requests.get(im['coco_url']).content
            #write image
            with open(img_name, 'wb') as handler:
                handler.write(img_data)

def get_model(url:str)->None:
    """
    Downloads model, removes class weights and saves locally

    Arguments:
        url: str, HTTP to request model

    Returns:
        None
    """

    # get pretrained weights
    checkpoint = torch.hub.load_state_dict_from_url(
        url=url,
        map_location='cpu',
        check_hash=True)

    # remove class weights
    del checkpoint["model"]["class_embed.weight"]
    del checkpoint["model"]["class_embed.bias"]

    # save
    torch.save(checkpoint,
               'model/detr-r50_no-class-head.pth')

def split2index(split):
    """
    Converts a split of percentages to list indexes

    Arguments:
        split: list, list of split percentages

    Returns:
        split_i: list, list of split end index
    """
    split_i = [int(i*num_samples) for i in split]
    split_i[1] = split_i[1] + split_i[0]
    split_i[2] = split_i[2] + split_i[1]
    return split_i

def clean_up():
    """
    Deletes files and directories no longer needed
    """
    os.delete('data/anns_trainval2017')
    os.delete('data/anns_trainval2017.zip')
    
if __name__ == "__main__":

    CLASSES = ['cat', 'banana']
    coco_path = 'data/anns_trainval2017'
    custom_path = 'data/custom' #link to this path for training
    ann_type = 'val2017' # just using val since we only need a small amount of data
    num_samples = 100 
    split = (0.7, 0.15, 0.15)

    #make data directory
    os.makedirs(custom_path, exist_ok=True)
    
    # download coco2017 annotaions archive
    ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    get_archive(ann_url, coco_path)
    
    # extract zip
    extract_json(coco_path, ann_type)

    # instantiate COCO specifying the annotations json path
    coco = COCO(os.path.join(coco_path, f"annotations/instances_{ann_type}.json"))
    cats = coco.loadCats(coco.getCatIds(catNms=CLASSES))
    
    # create dict of custom COCO format annotations for train, val, and test
    custom_anns = [{
        "info": coco.dataset["info"],
        "licenses": coco.dataset["licenses"],
        "categories": cats,
        "images": [],
        "annotations": []
    } for _ in range(3)]

    # convert split percentages to list indicies 
    split_i = split2index(split)

    # gather images and annotations from annotation json for specified classes and
    # number of samples

    print(coco.dataset["annotations"][0])
    for cat in cats:
        # get images
        imgIds = coco.getImgIds(catIds=cat["id"])[:100]
        images = coco.loadImgs(imgIds)
        # save to cooresponding dict
        custom_anns[0]["images"].extend( images[:split_i[0]] )
        custom_anns[1]["images"].extend( images[split_i[0]:split_i[1]] )
        custom_anns[2]["images"].extend( images[split_i[1]:] )

        # get cooresponding annotations
        annotations = [coco.loadAnns(coco.getAnnIds(imgIds=im["id"]))[0] for im in images]
        custom_anns[0]["annotations"].extend( annotations[:split_i[0]] )
        custom_anns[1]["annotations"].extend( annotations[split_i[0]:split_i[1]] )
        custom_anns[2]["annotations"].extend( annotations[split_i[1]:] )

    print(f"{len(custom_anns[0]['annotations'])} training samples")
    print(f"{len(custom_anns[1]['annotations'])} validation samples")
    print(f"{len(custom_anns[2]['annotations'])} test samples")

    # download images
    get_images(custom_anns[0]["images"], custom_path, 'train2017')
    get_images(custom_anns[1]["images"], custom_path, 'val2017')
    get_images(custom_anns[2]["images"], custom_path, 'test2017')

    # write annotations to file
    print("Writing annotations...")
    os.makedirs(os.path.join(custom_path, 'annotations'), exist_ok=True)
    with open(os.path.join(custom_path, 'annotations/custom_train.json'), 'w') as outfile:
        json.dump(custom_anns[0], outfile)
    with open(os.path.join(custom_path, 'annotations/custom_val.json'), 'w') as outfile:
        json.dump(custom_anns[1], outfile)
    with open(os.path.join(custom_path, 'annotations/custom_test.json'), 'w') as outfile:
        json.dump(custom_anns[2], outfile)

    # download pretrained weights
    print("Downloading model...")
    os.makedirs('model', exist_ok=True)
    model_url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
    get_model(model_url)

    #clean_up()
