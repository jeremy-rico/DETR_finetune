import os
import json
from pycocotools.coco import COCO
import requests
import torch
import urllib.request
import zipfile
from tqdm import tqdm
import argparse
from pathlib import Path

"""
This file downloads the 2017 COCO train/val annotaions file and
a few images from the classes 'cat' and 'banana'. It also downloads
the DETR model.
"""
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--classes', nargs=2, default=['cat', 'banana'], type=str)
    parser.add_argument('--num_samples', default=100, type=int)
    parser.add_argument('--split', default=(0.7,0.15,0.15), type=tuple)
    parser.add_argument('--coco_type', default='trainval2017', type=str)
    parser.add_argument('--out_dir', default='data/', type=str)

    return parser

def get_archive(url:str, out_dir:Path, coco_type:str)->Path:
    """
    Downloads the archive pointed to by the url

    Arguments:
        url: str, url to send request
        out_dir: pathlib.Path, directory path to save zip file
        coco_type: str, coco annotation type

    Returns:
        archive_path: Path, path to downloaded zip
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"annotations_{coco_type}.zip"
    if archive_path.exists():
        print(f"Annotations found at {archive_path}")
        return archive_path
    
    print(f"Downloading annotations to {archive_path}...")
    archive = requests.get(url, str(archive_path)).content
    with open(archive_path, 'wb') as handler:
        handler.write(archive)
        
    return archive_path

def extract_json(out_dir:Path, archive_path:Path, split_type:str)->None:
    """
    Extracts one split member from the zip file 

    Arguments:
        out_dir: pathlib.Path path to save extracted annotations
        archive_path: pathlib.Path path to coco zip file
        split_type: str, type of annotation file (train2017, val2017)

    Returns:
        None
    """
    ann_path = out_dir / archive_path.stem
    ann_file = f'annotations/instances_{split_type}.json'
    if ann_path.exists():
        print(f"Extracted annotations found at {ann_path}")
        return

    print(f"Extracting annotations to {ann_path}")
    with zipfile.ZipFile(str(archive_path), 'r') as zip_ref:
        zip_ref.extract(ann_file, ann_path)

def get_images(imgs:list, coco_dir:Path, split:str)->None:
    """
    Downloads images to local folder

    Arguments:
        imgs: list, list of images in coco format
        coco_dir: pathlib.Path, path to location where coco annotations are stored
        split_type: str, type of annotation file (train2017, val2017)

    Returns:
        None
    """
    img_dir = coco_dir / split
    img_dir.mkdir(parents=True, exist_ok=True)
    for im in tqdm(imgs, desc=f"Downloading {split} images"):
        img_name = img_dir /  im['file_name']
        if not img_name.exists():
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
    model_path = save_dir / "detr-r50_no-class-head.pth"
    if model_path.exists():
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

def main(args):
    # data path variables
    #coco_path = Path(args.out_dir) / f'annotations_{args.coco_type}'
    coco_path = Path(args.out_dir)
    custom_path = Path(args.out_dir) / 'custom' #link to this path for training

    # download coco2017 annotaions archive
    ann_url = f'http://images.cocodataset.org/annotations/annotations_{args.coco_type}.zip'
    archive_path = get_archive(ann_url, coco_path, args.coco_type)

    # extract member from zip
    # just using val since we only need a small amount of data
    ann_type = 'val2017' if args.num_samples < 150 else 'train2017' 
    extract_json(coco_path, archive_path, ann_type)

    # instantiate COCO specifying the annotations json path
    ann_file = coco_path / f"annotations_{args.coco_type}/annotations/instances_{ann_type}.json"
    coco = COCO(ann_file)
    
    # get category ids for our classes
    cats = coco.loadCats(coco.getCatIds(catNms=args.classes))
    
    # create dict of custom COCO format annotations for train, val, and test
    custom_anns = {keyword: {
        "info": coco.dataset["info"],
        "licenses": coco.dataset["licenses"],
        "categories": cats,
        "images": [],
        "annotations": []
    } for keyword in ['train', 'test', 'val'] }

    # convert split percentages to list indicies 
    split_i = split2index(args.split, args.num_samples)

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
    custom_ann_path = custom_path / 'annotations'
    custom_ann_path.mkdir(parents=True, exist_ok=True)

    for keyword in ['train', 'val', 'test']:
        # download images
        get_images(custom_anns[keyword]["images"], custom_path, f"{keyword}2017")

        # write annotations to file
        with open(custom_ann_path / f"custom_{keyword}.json", 'w') as outfile:
            json.dump(custom_anns[keyword], outfile)

    # download pretrained weights
    model_path = Path('model')
    model_path.mkdir(parents='True', exist_ok=True)
    model_url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
    get_model(model_url, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('COCO and DETR download script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    main(args)
