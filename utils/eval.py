from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss,L1Loss
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor
from .losses import GIoULoss, CrossEntropyLoss, L1Loss
from .util import multi_apply



def calculate_em(list1, list2):
    assert len(list1) == len(list2)
    return sum(1 for x, y in zip(list1, list2) if x == y) / len(list1)

def evaluate_text(preds, gts):

    p = precision_score(gts, preds, average='binary')
    r = recall_score(gts, preds, average='binary')
    f1 = f1_score(gts, preds, average='binary')
    acc = accuracy_score(gts, preds)
    EM = calculate_em(preds, gts)

    return acc, f1, p, r, EM

def batch_post_process(labels, labels_pred):

    # labels: ['[['food', 'O'],['shortages', 'O'],['in', 'O']]',...]        batch['label'] in our case
    # labels_pred: ['food shortages', ...]
    labels = [eval(label) for label in labels]
    preds = []
    gts = []
    ems = []
    accs = []
    

    for lab, lab_pred in zip(labels, labels_pred):
        lab_pred = lab_pred.split(' ')
        # t5 encode '<' as unk and decode as '' that might disrupt calculation
        for i,_ in enumerate(lab_pred):
            if 'user' in lab_pred[i]:
                lab_pred[i] = '<user>'
        pred = [a[0] in lab_pred for a in lab]
        gt = [a[1] != 'O' for a in lab]
        exact_match = [a == b for (a, b) in zip(pred, gt)]
        ems.append(sum(exact_match) // len(exact_match))
        accs += exact_match
        preds += pred
        gts += gt
    
    batch_acc = np.mean(accs)
    batch_EM = np.mean(ems)
    return preds, gts, batch_acc, batch_EM


def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.

    else:
        # correct ap caculation
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_eval(pred, target, threshold=0.5, use_07_metric=False, ):
    '''
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    '''
    # print(pred)
    if len(pred) == 0:
        return -1
    image_ids = [x[0] for x in pred]
    confidence = np.array([float(x[3]) for x in pred])
    BB = np.array([[x[1][0], x[1][1], x[2][0], x[2][1]] for x in pred])
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    # sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    # print(BB)
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    npos = 0.
    for key1 in target:
        npos += len(target[key1])
    # print(npos)
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d, image_id in enumerate(image_ids):
        bb = BB[d]
        if image_id in target:
            temp = target[image_id]
            # print(temp)
            BBGT = [[item[0][0], item[0][1], item[1][0], item[1][1]] for item in temp]
            # print(BBGT)
            for bbgt in BBGT:
                # compute overlaps
                # intersection
                ixmin = np.maximum(bbgt[0], bb[0])
                iymin = np.maximum(bbgt[1], bb[1])
                ixmax = np.minimum(bbgt[2], bb[2])
                iymax = np.minimum(bbgt[3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                union = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (bbgt[2] - bbgt[0] + 1.) * (
                        bbgt[3] - bbgt[1] + 1.) - inters
                # if union == 0:
                #     print(bb, bbgt)

                overlaps = inters / union
                if overlaps > threshold:
                    tp[d] = 1
                    BBGT.remove(bbgt)
                    if len(BBGT) == 0:
                        del target[image_id]
                    break
            fp[d] = 1 - tp[d]
        else:
            fp[d] = 1
    # print(tp)
    # print(fp)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    # print(tp)
    # print(fp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # print(rec,prec)
    ap = voc_ap(rec, prec, use_07_metric)
    return ap

def eval_ap_batch(pred_boxes,target_boxes):
    obj_preds = []
    obj_targets = {}
    for (i, (boxes,bboxes)) in tqdm(enumerate(zip(pred_boxes,target_boxes))):
        # emissions, img_output = model()  # seq_len * bs * labels

        # 处理一个sample里的多个box
        targets_img = []            # xywh
        for j in range(len(bboxes)):
            box = bboxes[j]
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            targets_img.append([(x1, y1), (x2, y2), 1.0])
        # print(targets_img)
        if targets_img == [[(0, 0), (0, 0), 1.0]]:
            targets_img = []
        # return targets_img
        
        # pred_boxes处理        xyxy

        preds_img = []
        for j in range(len(boxes)):
            box = boxes[j]
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            # preds_img.append([(x1, y1), (x2, y2), pred_logits])
            preds_img.append([(x1, y1), (x2, y2), 1])

        # return preds_img
        for item in preds_img:
            item.insert(0, i)
            obj_preds.append(item)
        # return obj_preds

        for item in targets_img:
            if i not in obj_targets.keys():
                obj_targets[i] = []
            obj_targets[i].append(item)

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    aps = list()
    for th in thresholds:
        a = obj_preds.copy()
        b = obj_targets.copy()
        aps.append(voc_eval(a, b, th))

    ap = np.mean(aps)
    ap50 = aps[0]
    ap75 = aps[5]

    return ap, ap50, ap75

def giou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows))
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])

    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])

    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])

    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_area = outer[:, 0] * outer[:, 1]
    union = area1+area2-inter_area
    closure = outer_area

    ious = inter_area / union - (closure - union) / closure
    ious = torch.clamp(ious,min=-1.0,max = 1.0)
    if exchange:
        ious = ious.T
    return torch.sum(1-ious)

def giou_losses(img_out,labels):
    losses = 0
    for img_o, label in zip(img_out,labels):
        losses = losses + giou(img_o, label)
    return losses

def loss_by_feat_single(cls_scores: Tensor, bbox_preds: Tensor,text_masks: Tensor,
                        box_labels: Tensor, target_boxes: Tensor,
                        Loss_cls =  CrossEntropyLoss(#bg_cls_weight=0.1,
                                                    use_sigmoid=False,
                                                    use_mask=False,
                                                    loss_weight=1.0,
                                                    class_weight=None),
                        Loss_bbox = L1Loss(loss_weight=5.0),              
                        Loss_iou = GIoULoss(loss_weight=2.0),
                        sync_cls_avg_factor: bool = False) -> Tuple[Tensor]:
    """Loss function for outputs from a single decoder layer of a single
    feature level.
    cls_scores(img_out.logits): [bs, num_q, 512]
    bbox_preds(img_out.pred_boxes): [bs, num_q, 4]
    target_boxes: [bs, 10, 4]
    text_masks: [bs, 512]
    box_labels: [bs, 10]

    Args:
        cls_scores (Tensor): Box score logits from a single decoder layer
            for all images, has shape (bs, num_queries, cls_out_channels).
            [bs, 2, 512]
        bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
            for all images, with normalized coordinate (cx, cy, w, h) and
            shape (bs, num_queries, 4).
            [bs, 2, 4]
        batch_gt_instances (list[:obj:`InstanceData`]): Batch of
            gt_instance. It usually includes ``bboxes`` and ``labels``
            attributes.
        batch_img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.

    Returns:
        Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
        `loss_iou`.
    """
    num_imgs = cls_scores.size(0)
    cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
    bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

    if bbox_preds.size()[1] > 1:                             # num_q > 1时处理
        pred_select = []
        for i,label in enumerate(box_labels-1):         # 把[0,1,2]转换成[-1,0,1], 因为bbox里0是第一类，1是第二类
            t = bbox_preds[i,label[0].long(),:]         # 在这里选择label对应的预测框
            pred_select.append(t)
        bbox_preds = torch.stack(pred_select)

    num_total_pos = int(box_labels.sum())
    num_total_neg = torch.sum(box_labels.eq(0)).item()
    
    # label_weights = torch.stack(label_weights_list, 0)
    bbox_targets = target_boxes
    # bbox_weights = torch.cat(bbox_weights_list, 0)

    # not sure!!!
    bbox_weights = None 
    label_weights = None

    # ===== this change =====
    # Loss is not computed for the padded regions of the text.
    assert (text_masks.dim() == 2)
    text_masks1 = text_masks.new_zeros(
        (text_masks.size(0), text_masks.size(1))).to(text_masks.device)
    text_masks1[:, :text_masks.size(1)] = text_masks
    text_mask = (text_masks1 > 0).unsqueeze(1)
    
    text_mask1 = text_mask.repeat(1, cls_scores.size(1), 1)
    cls_scores = torch.masked_select(cls_scores, text_mask1).contiguous()

    padded_box_labels = torch.zeros([box_labels.size(0),text_mask.size(2)]).to(text_mask.device)
    padded_box_labels[:,:box_labels.size(1)] = box_labels
    labels = torch.masked_select(padded_box_labels,torch.squeeze(text_mask, dim=1))

    cls_scores = cls_scores.view(-1,labels.size(0))

    # label_weights = label_weights[...,
    #                                 None].repeat(1, 1, text_mask.size(-1))
    # label_weights = torch.masked_select(label_weights, text_mask)

    # classification loss
    bg_cls_weight = 1e-3
    cls_avg_factor = num_total_pos * 1.0 + \
        num_total_neg * bg_cls_weight
    
    cls_avg_factor = max(cls_avg_factor, 1)
    
    cls_scores = cls_scores.unsqueeze(dim=0)
    labels = labels.unsqueeze(dim=0).long()

    # num_q >1时
    if bbox_preds.size()[1] > 1:                             # num_q > 1时处理
        labels = labels>1
        labels = labels.long()

    loss_cls = Loss_cls(
        cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

    # Compute the average number of gt boxes across all gpus, for
    # normalization purposes
    num_total_pos = loss_cls.new_tensor([num_total_pos])
    num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

    # DETR regress the relative position of boxes (cxcywh) in the image,
    # thus the learning target is normalized by the image size. So here
    # we need to re-scale them for calculating IoU loss
    # 我们在外面resize好了，不用再resize
    # bbox_preds = bbox_preds.reshape(-1, 4)
    bboxes = bbox_preds     # bs, 4
    bboxes_gt = target_boxes    # bs, 1, 4

    # pad out boxes
    if bboxes.dim() == 2:
        bboxes = torch.unsqueeze(bboxes,dim=1)
    pad_boxes = torch.zeros(bboxes_gt.size()).to(bboxes.device) # bs, 10, 4
    pad_boxes[:,:bboxes.size(1),:] = bboxes
    bboxes = pad_boxes
    
    # regression IoU loss, defaultly GIoU loss
    mask = bboxes_gt>0
    bbox_weights = torch.where(mask, torch.tensor(1), torch.tensor(0)).float().to(bboxes_gt.device)
    loss_iou = Loss_iou(
        bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

    bbox_preds = bboxes
    bbox_targets = bboxes_gt
    # regression L1 loss
    loss_bbox = Loss_bbox(
        bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
    return loss_cls, loss_bbox, loss_iou


def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def bbox_cxcywh_to_xyxy(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


def bbox_xyxy_to_cxcywh(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)

def get_targets(cls_scores_list: List[Tensor],
                    bbox_preds_list: List[Tensor],
                    batch_gt_instances: List,
                    batch_img_metas: List[dict]) -> tuple:
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image, has shape [num_queries,
                cls_out_channels].      out.logits
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_queries, 4].      out.bboxes
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(get_targets_single,
                                      cls_scores_list, bbox_preds_list,
                                      batch_gt_instances, batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

def get_targets_single(cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances,#: InstanceData,
                            img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        InstanceData = None
        self = None
        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        gt_bboxes = gt_instances.bboxes

        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # Major changes. The labels are 0-1 binary labels for each bbox
        # and text tokens.
        max_text_len = 20
        labels = gt_bboxes.new_full((num_bboxes, max_text_len),
                                    0,
                                    dtype=torch.float32)
        labels[pos_inds] = gt_instances.positive_maps[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)