import torch
from pathlib import Path
from pycocotools.coco import COCO
from utils import run_infer, f_beta

def main():
    num_classes = 2
    model = torch.hub.load('facebookresearch/detr',
                           'detr_resnet50',
                           pretrained=False,
                           num_classes=num_classes)

    out_dir = Path('model')
    checkpoint = torch.load(out_dir / 'checkpoint.pth',
                            map_location='cpu')

    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    dataDir = Path('data/custom')
    test_annFile = dataDir / f'annotations/custom_test.json'
    test_coco = COCO(test_annFile)

    outputs, targets = run_infer(test_coco, model, dataDir)

    per_cls_f1 = f_beta(test_coco, outputs, targets, beta=1)

    for cls, f1 in per_cls_f1.items():
        print(f"Class name: {cls:>6}, f1-score: {f1:.3}")

if __name__ == "__main__":
    main()
