import torch
from tqdm import tqdm
from PIL import Image
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T
from pathlib import Path
import torch.nn.functional as F
from pycocotools.coco import COCO
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_area

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

class HungarianMatcher(torch.nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best
    predictions, while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the
                        matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box
                       coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the
                       matching
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the
                                classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted
                               box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is
                     a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number
                           of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                         box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        i_out = [(torch.as_tensor(i, dtype=torch.int64),
                  torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        #print(f"GIOU???: {i_out[0][-1]}")
        return {'indices': i_out, 'giou': cost_giou.view(bs, num_queries, -1).cpu()}


    def build_matcher(args):
        return HungarianMatcher(cost_class=args.set_cost_class,
                                cost_bbox=args.set_cost_bbox,
                                cost_giou=args.set_cost_giou)

def format_labels(anns):
    target = {
        'labels': [],
        'boxes': []
    }
    for ann in anns:
        target['labels'].append(ann['category_id'])
        target['boxes'].append(ann['bbox'])

    for k in target.keys():
        target[k] = torch.tensor(target[k])

    return target
            
# runs inference on all test images in coco file
def run_infer(coco, model, dataDir):
    outputs = {'pred_logits': torch.Tensor([]),
               'pred_boxes': torch.Tensor([])}
    targets = []
    
    imgIds = coco.getImgIds() #get all test image ids
    imgIds = imgIds[:5]
    for imgId in tqdm(imgIds, desc='Running inference...'): 
        # get image
        test_img = coco.loadImgs(imgId)[0]
        img_name = dataDir / 'test2017' / test_img['file_name']
        im = Image.open(img_name)
        img = transform(im).unsqueeze(0)
        
        # format labels
        annIds = coco.getAnnIds(imgIds=imgId)
        anns = coco.loadAnns(annIds)

        # infer with model
        img_outputs = model(img)
        for k in outputs.keys():
            outputs[k] = torch.cat( (outputs[k], img_outputs[k]) )

        targets.append(format_labels(anns))
        
    return outputs, targets

def get_f1(coco, outputs, targets, giou_thresh=0.5, conf_thresh=0.7):
    #load cat ids
    cats = coco.loadCats(coco.getCatIds())
    
    # class based conf matrix holder, index is the category id
    cls_conf_mat = [{'tp':0, 'fp':0, 'fn':0} for cat in cats] #tn not applicable
    
    #perform matching
    # I modified this class to also return giou for every pair
    matcher = HungarianMatcher()
    matcher_dict = matcher.forward(outputs, targets)
    pairs = matcher_dict['indices']
    giou = matcher_dict['giou'] #giou for pair

    # get giou for every pair for conf matrix
    for i, pair in enumerate(pairs):
        # get all preds of high conf for fp
        probas_all = outputs['pred_logits'].softmax(-1)[i, :, :-1]
        keep = probas_all.max(-1).values > conf_thresh
        probas_all_set = set(probas_all[keep])

        for j in range(len(pair[0])): #num matched per image
            index_i = pair[0][j]
            index_j = pair[1][j]

            probas_matched = outputs['pred_logits'][i][index_i].softmax(-1)
            if probas_matched[:len(cats)] in probas_all_set:
                probas_all_set.remove(probas_matched[:len(cats)])
            pred_id = torch.argmax(probas_matched)
            true_id = targets[i]['labels'][index_j]
            pair_giou = giou[i][index_i][j]

            # correct class and box = true positive
            print(f"Pred: {pred_id}, True: {true_id}, giou: {pair_giou}")
            if pred_id == len(cats): #no_object prediction
                cls_conf_mat[true_id]['fn'] += 1
            elif pair_giou >= giou_thresh:
                if pred_id == true_id:
                    cls_conf_mat[pred_id]['tp'] += 1
                    print('TP!')
                else:
                    cls_conf_mat[pred_id]['fp'] += 1 
                    cls_conf_mat[true_id]['fn'] += 1
                    print('Misclass!')
            else:
                cls_conf_mat[true_id]['fn'] += 1
                print('FN!')
                
        for probs in probas_all_set:
            cls_conf_mat[probs.argmax()]['fp'] += 1
            print('FP!')

    for i, cls in enumerate(cls_conf_mat):
        prec = cls['tp'] / (cls['tp']+cls['fp'])
        recall = cls['tp'] / (cls['tp']+cls['fn'])
        cls_conf_mat[i] = 2*prec*recall / (prec*recall) if prec+recall > 0 else 0

    f1_holder = {cat['name']: cls[cat['id']] for cat in cats}
    print(f1_holder)
        

def main():
    num_classes = 2
    model = torch.hub.load('facebookresearch/detr',
                           'detr_resnet50',
                           pretrained=False,
                           num_classes=num_classes)

    out_dir = Path('model')
    checkpoint = torch.load(out_dir / 'checkpoint.pth',
                            map_location='cpu')

    model.load_state_dict(checkpoint['model'],
                          strict=False)
    model.eval()

    dataDir = Path('data/custom')
    test_annFile = dataDir / f'annotations/custom_test.json'
    test_coco = COCO(test_annFile)

    outputs, targets = run_infer(test_coco, model, dataDir)

    f1 = get_f1(test_coco, outputs, targets)

if __name__ == "__main__":
    main()
