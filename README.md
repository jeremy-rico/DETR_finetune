DETR Finetuning

This script creates a custom COCO dataset of two classes and fintunes the DETR model on them.

It also instruments the code to create a per class f1 score for inference images.

To run

```shell
python3 -m venv DETR
python3 -m pip install requirements.txt
python3 download.py
```

Training is handled by a finetuning fork of the DETR repo by woctezuma. It can be found here:
https://github.com/woctezuma/detr.git

clone the repo and run main.py to train on the data created by download.py. It should look something like this:
```shell
python3 main.py \
  --dataset_file "custom" \
  --coco_path path/to/coco/dataset \
  --output_dir path/to/save/directory \
  --resume path/to/model/checkpoint \
  --num_classes %num_classes \
  --epochs 50
```

Once you have your model checkpoint, you can run
```shell
python3 infer.py
```

To get the per class f1 values
