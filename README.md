DETR Finetuning

This script creates a custom COCO dataset of two classes and fintunes the DETR model on them.

It also instruments the code to create a per class f1 score for inference images.

To run

# python3 -m venv DETR
# python3 -m pip install requirements.txt

Its is recommended to run the notebook on a cloud gpu service such as google colab. 