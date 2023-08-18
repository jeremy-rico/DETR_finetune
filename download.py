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
        return
    
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
        return

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

def get_model(url:str, save_dir:str)->None:
    """
    Downloads model, removes class weights and saves locally

    Arguments:
        url: str, HTTP to request model

    Returns:
        None
    """
    model_path = os.path.join(save_dir, "detr-r50_no-class-head.pth")
    if os.path.isfile(model_path):
        print(f"Cached model found at {model_path}")
        return

    print("Downloading model...")
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
               model_path)

def split2index(split:list, num_samples:int)->list:
    """
    Converts a split of percentages to list indexes

    Arguments:
        split: list, list of split percentages

    Returns:
        split_i: list, list of split end index
    """
    if sum(split) > 1.0:
        raise ValueError("Please use valid split values")
            
    split_i = [int(i*num_samples) for i in split]
    split_i[1] = split_i[1] + split_i[0]
    split_i[2] = split_i[2] + split_i[1]
    return split_i

def clean_up()->None:
    """
    Deletes files and directories no longer needed
    """
    os.delete('data/anns_trainval2017')
    os.delete('data/anns_trainval2017.zip')
    
if __name__ == "__main__":

    # model/trainins variables
    CLASSES = ['cat', 'banana']
    num_samples = 100 # num of total samples to use per class
    split = (0.7, 0.15, 0.15) # train, val, test split

    # data path variables
    coco_path = 'data/anns_trainval2017'
    custom_path = 'data/custom' #link to this path for training

    #make data directory
    os.makedirs(custom_path, exist_ok=True)
    
    # download coco2017 annotaions archive
    ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    get_archive(ann_url, coco_path)

    #extract member from zip 
    ann_type = 'val2017' # just using val since we only need a small amount of data
    extract_json(coco_path, ann_type)

    # instantiate COCO specifying the annotations json path
    coco = COCO(os.path.join(coco_path, f"annotations/instances_{ann_type}.json"))
    
    # get category ids for our classes
    cats = coco.loadCats(coco.getCatIds(catNms=CLASSES))
    
    # create dict of custom COCO format annotations for train, val, and test
    custom_anns = {keyword: {
        "info": coco.dataset["info"],
        "licenses": coco.dataset["licenses"],
        "categories": cats,
        "images": [],
        "annotations": []
    } for keyword in ['train', 'test', 'val'] }

    # convert split percentages to list indicies 
    split_i = split2index(split, num_samples)

    # gather images and annotations from annotation json for specified classes and
    # number of samples
    for i, cat in enumerate(cats):
        # get images
        imgIds = coco.getImgIds(catIds=cat["id"])[:100]
        images = coco.loadImgs(imgIds)

        # put images in cooresponding dict
        custom_anns["train"]["images"].extend( images[:split_i[0]] )
        custom_anns["val"]["images"].extend( images[split_i[0]:split_i[1]] )
        custom_anns["test"]["images"].extend( images[split_i[1]:] )

        # get all annotations for each image for our categories
        for keyword in ['train', 'test', 'val']:
            for im in custom_anns[keyword]["images"]:
                anns = coco.loadAnns(coco.getAnnIds(imgIds=im["id"], catIds=cat["id"]))
                for ann in anns:
                    #change cat id
                    ann["category_id"] = i
                    custom_anns[keyword]["annotations"].append(ann)

        for k in custom_anns.keys():
            custom_anns[k]["categories"][i]["id"] = i

    print("Preparing custom dataset...")

    # make annotations directory
    os.makedirs(os.path.join(custom_path, 'annotations'), exist_ok=True)

    for keyword in ['train', 'val', 'test']:
        # download images
        get_images(custom_anns[keyword]["images"], custom_path, f"{keyword}2017")

        # write annotations to file
        with open(os.path.join(custom_path, f"annotations/custom_{keyword}.json"), 'w') as outfile:
            json.dump(custom_anns[keyword], outfile)

    test_ann_file = os.path.join(custom_path, f"annotations/custom_val.json")
    coco = COCO(test_ann_file)
    
    # download pretrained weights
    os.makedirs('model', exist_ok=True)
    model_url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
    get_model(model_url, 'model')

    #clean_up()
