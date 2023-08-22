import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import torch.nn.functional as F
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
    """
    Stolen from https://github.com/facebookresearch/detr
    See description and input formatting there
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
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

        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        
    def build_matcher(args):
        return HungarianMatcher(cost_class=args.set_cost_class,
                                cost_bbox=args.set_cost_bbox,
                                cost_giou=args.set_cost_giou)


def format_labels(coco_img, anns):
    """
    Formats COCO labels to be passed into the Hungarian Matcher

    Arguments:
        coco_img:
    
    """
    target = {
        'labels': [],
        'boxes': []
    }
    for ann in anns:
        # convert coco format to cx, cy, w, h as ratio
        cx = (ann['bbox'][0]+0.5*ann['bbox'][2]) / coco_img['width']
        cy = (ann['bbox'][1]+0.5*ann['bbox'][3]) / coco_img['height']
        w = ann['bbox'][2] / coco_img['width']
        h = ann['bbox'][3] / coco_img['height']
        new_box = [cx, cy, w, h]

        target['labels'].append(ann['category_id'])
        target['boxes'].append(new_box)

    for k in target.keys():
        target[k] = torch.tensor(target[k])

    return target
            
# runs inference on all test images in coco file
def run_infer(coco, model, dataDir):
    """
    Runs model inference on ALL images in a coco object

    Arguments:
        coco: coco data
        model: torch model
        dataDir: directory where images are held

    Returns:
        outputs: a dictionary with the following key value pairs
            'pred_logits' a tensor of dim [batch_size, num_queries, num_classes]
            'pred_boxes' a tensor of dim [batch_size, num_queries, 4]
        targets: a list of dictionaries where len(targets) = batch size each element has
                 the following keys:
            'labels': tensor of dim [batch size] containing class labels
            'boxes': tensor of dim [batch size, 4] containing formatted boxes (see format_labels)
    """
    # instanciate data holders
    outputs = {'pred_logits': torch.Tensor([]),
               'pred_boxes': torch.Tensor([])}
    targets = []

    # load all image IDs
    imgIds = coco.getImgIds()
    
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

        # concat model outputs to holder
        for k in outputs.keys():
            outputs[k] = torch.cat( (outputs[k], img_outputs[k]) )

        # append formated labels
        targets.append(format_labels(test_img, anns))
        
    return outputs, targets

"""
def box_cxcywh_to_mxmywh(coco_img, box):
    cx, cy, w, h = box.unbind(1)
    img_width, img_height = (coco_img['width'], coco_img['height'])
    xmin = (cx.item()-0.5*w.item())*img_width
    ymin = (cy.item()-0.5*h.item())*img_height
    w = w.item()*img_width
    h = h.item()*img_height
    return [xmin, ymin, w, h]
"""

#import matplotlib.patches as patches ###DEBUG ###
def f_beta(coco, outputs, targets, giou_thresh=0.5, conf_thresh=0.7, beta=1):
    """
    Processes output of run_infer to create a per class f1 score

    Arguments:
        coco: coco dataset
        outputs: outputs of model as a dict (see run_infer)
        targets: list of target dicts (see run_infer)
        giou_thresh: giou threshold needed to be considered true positive
        conf_thresh: confidence threshold needed to be considered for false positive

    Returns:
       a dict where k, v pairs are {class name: f1-score}
    """
    #load cat ids
    cats = coco.loadCats(coco.getCatIds())

    """
    ### DEBUG ###
    imgIds = coco.getImgIds()
    dataDir = Path('data/custom/')
    inferDir = dataDir / 'infer'
    inferDir.mkdir(parents=True, exist_ok=True)
    """
    ### END DEBUG ###
    
    # class based conf matrix holder, index is the category id
    cls_conf_mat = [{'tp':0, 'fp':0, 'fn':0} for cat in cats] #tn not applicable
    
    #perform matching
    # I modified this class to also return giou for every pair
    matcher = HungarianMatcher()
    pairs = matcher.forward(outputs, targets)

    # get giou for every pair for conf matrix
    for i, pair in enumerate(pairs):
        """
        ### DEBUG ###
        test_img = coco.loadImgs(imgIds[i])[0]
        img_name = dataDir / 'test2017' / test_img['file_name']
        im = Image.open(img_name)

        fig, ax = plt.subplots()
        ax.imshow(im)
        ### END DEBUG ###
        """
        # get all preds of high conf for fp
        probas_all = outputs['pred_logits'].softmax(-1)[i, :, :-1]
        keep = probas_all.max(-1).values > conf_thresh
        probas_all_set = set(probas_all[keep])

        for j in range(len(pair[0])): #num matched per image
            index_i = pair[0][j] #index of prediction
            index_j = pair[1][j] #index of matched annotation

            probas_matched = outputs['pred_logits'][i][index_i].softmax(-1)
            pred_id = torch.argmax(probas_matched)
            true_id = targets[i]['labels'][index_j]

            pred_box = outputs['pred_boxes'][i][index_i].view(1, -1)
            true_box = targets[i]['boxes'][index_j].view(1, -1)

            giou = generalized_box_iou(box_cxcywh_to_xyxy(pred_box),
                                       box_cxcywh_to_xyxy(true_box) )

            """
            pred = box_cxcywh_to_mxmywh(test_img, pred_box)
            true = box_cxcywh_to_mxmywh(test_img, true_box)
            
            #print(giou)
            #continue

            pred_rect = patches.Rectangle(
                (pred[0], pred[1]),
                pred[2], pred[3],
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            true_rect = patches.Rectangle(
                (true[0], true[1]),
                true[2], true[3],
                linewidth = 1,
                edgecolor = 'g',
                facecolor='none'
            )
             
            ax.add_patch(pred_rect)
            ax.add_patch(true_rect)

            plt.text(true[0], true[1], f"True: {true_id}", color='g')
            plt.text(pred[0], pred[1], f"Pred: {pred_id}", color='r')
            plt.text(j*200, 0, f"GIOU: {giou.item():.3}")

            ### END DEBUG ###
            """
            # remove matched boxes from all high conf (left overs will be FP)
            if probas_matched[:len(cats)] in probas_all_set:
                probas_all_set.remove(probas_matched[:len(cats)])

            # correct class and box = true positive
            if pred_id == len(cats): #no_object prediction
                cls_conf_mat[true_id]['fn'] += 1
            elif giou >= giou_thresh:
                if pred_id == true_id:
                    cls_conf_mat[pred_id]['tp'] += 1
                else:
                    cls_conf_mat[pred_id]['fp'] += 1 #misclass
            else:
                cls_conf_mat[true_id]['fn'] += 1

        # any predictions not matched to a annotations are fp for the predicted class
        for probs in probas_all_set:
            cls_conf_mat[probs.argmax()]['fp'] += 1

        #plt.savefig(inferDir/test_img['file_name']) ###DEBUG ###
        #plt.close()

    #calculate per class f1-score
    for i, cls in enumerate(cls_conf_mat):
        prec = cls['tp'] / (cls['tp']+cls['fp']) if cls['tp']+cls['fp'] > 0 else 0
        recall = cls['tp'] / (cls['tp']+cls['fn']) if cls['tp']+cls['fn'] > 0 else 0
        cls_conf_mat[i] = ( (2 * beta**2) * prec * recall ) / ( (beta**2 * prec) + recall ) if prec+recall > 0 else 0

    return {cat['name']: cls_conf_mat[cat['id']] for cat in cats}
