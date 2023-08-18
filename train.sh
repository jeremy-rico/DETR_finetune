DATASET_FILE="custom"
COCO_PATH="/data/custom/"
OUT_DIR="outputs"
WEIGHTS="model/detr-r50_no-class-head.pth"
NUM_CLASSES=2
NUM_EPOCHS=10

!git clone https://github.com/facebookresearch/detr.git

!python main.py \
--dataset_file $DATASET_FILE \
  --coco_path $COCO_PATH \
  --output_dir $OUT_DIR \
  --resume $WEIGHTS \
  --num_classes $NUM_CLASSES \
  --epochs $NUM_EPOCHS
