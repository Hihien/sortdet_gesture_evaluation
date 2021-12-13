import numpy as np


def _as_nparray(*args):
    return tuple(np.array(arg) for arg in args)


def tube_accuracy(pred, gt):
    acc = 0.
    for frame_id in range(max(pred['start'], gt['start']), min(pred['end'], gt['end']) + 1):
        bbox_pred = pred['frames'][frame_id]
        bbox_gt = gt['bbox']
        if bbox_pred == None or len(bbox_pred) == 0:
            acc += 0
        else:
            acc += accuracy(bbox_gt, bbox_pred)
    return acc / (max(pred['end'], gt['end']) - min(pred['start'], gt['start']) + 1)


def edit_score(pred, gt, norm=True):
    score = 0.
    sequence_pred = []
    sequence_gt = []
    for frame_pred_id in range(pred['start'], pred['end'] + 1, 1):
        # print(type(sequence_pred))
        sequence_pred.append(frame_pred_id)
        sequence_preds = np.asarray(sequence_pred)

    for frame_gt_id in range(gt['start'], gt['end'] + 1, 1):
        sequence_gt.append(frame_gt_id)
        sequence_gts = np.asarray(sequence_gt)

    return _levenstein(sequence_preds, sequence_gts, norm)


def f1_overlap(pred, gt):
    f1 = 0.
    for frame_id in range(max(pred['start'], gt['start']), min(pred['end'], gt['end']) + 1):
        bbox_pred = pred['frames'][frame_id]
        bbox_gt = gt['bbox']
        if bbox_pred == None or len(bbox_pred) == 0:
            f1 += 0
        else:
            f1 += overlap_(bbox_gt, bbox_pred)
    return f1 / (max(pred['end'], gt['end']) - min(pred['start'], gt['start']) + 1)


def temporal_IoU(pred, gt, eps=1e-6):
    frame_start = max(pred['start'], gt['start'])
    frame_end = min(pred['end'], gt['end'])
    overlap = max(frame_end - frame_start, 0)
    union = (pred['end'] - pred['start']) + (gt['end'] - gt['start']) - overlap
    union = max(union, eps)
    temporal_IoU = overlap / union
    return temporal_IoU


# --------help_function--------
def overlap_(bboxes1, bboxes2, eps=1e-6, threshold=0.3):
    bboxes1, bboxes2 = _as_nparray(bboxes1, bboxes2)
    if len(bboxes1.shape) == 1:
        bboxes1 = bboxes1[np.newaxis]
    if len(bboxes2.shape) == 1:
        bboxes2 = bboxes2[np.newaxis]

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)

    n_true = bboxes1.shape[0]
    n_pred = bboxes2.shape[0]

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    TP = np.zeros(1, np.float)
    FP = np.zeros(1, np.float)
    true_used = np.zeros(n_true, np.float)

    for j in range(n_pred):
        x_start = np.maximum(bboxes2[j, 0], bboxes1[:, 0])
        y_start = np.maximum(bboxes2[j, 1], bboxes1[:, 1])
        x_end = np.minimum(bboxes2[j, 2], bboxes1[:, 2])
        y_end = np.minimum(bboxes2[j, 3], bboxes1[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(y_end - y_start, 0)
        # union = area2[j] + area1 - overlap
        # union = np.maximum(union, eps)
        IoU = overlap / area2[j]

        idx = IoU.argmax()

        if IoU[idx] >= threshold and not true_used[idx]:
            TP[-1] += 1
            true_used[idx] = 1
        else:
            FP[-1] += 1

    TP = TP.sum()
    FP = FP.sum()
    FN = n_true - true_used.sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    F1 = np.nan_to_num(F1)

    return F1


def accuracy(bboxes1, bboxes2, eps=1e-6, iou_thres=0.3):
    bboxes1, bboxes2 = _as_nparray(bboxes1, bboxes2)
    if len(bboxes1.shape) == 1:
        bboxes1 = bboxes1[np.newaxis]
    if len(bboxes2.shape) == 1:
        bboxes2 = bboxes2[np.newaxis]
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)

    n_true = bboxes1.shape[0]
    n_pred = bboxes2.shape[0]

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    tp = 0
    for j in range(n_pred):
        x_start = np.maximum(bboxes2[j, 0], bboxes1[:, 0])
        y_start = np.maximum(bboxes2[j, 1], bboxes1[:, 1])
        x_end = np.minimum(bboxes2[j, 2], bboxes1[:, 2])
        y_end = np.minimum(bboxes2[j, 3], bboxes1[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(y_end - y_start, 0)
        # union = area2[j] + area1 - overlap
        # union = np.maximum(union, eps)
        IoU = overlap / area2[j]

        idx = IoU.argmax()
        if IoU[idx] > iou_thres:
            tp += 1

    return tp / n_true


def _levenstein(pred, gt, norm=True):
    n_rows = len(pred)
    n_cols = len(gt)
    D = np.zeros([n_rows + 1, n_cols + 1], np.float)
    for i in range(n_rows + 1):
        D[i, 0] = i
    for i in range(n_cols + 1):
        D[0, i] = i
    for j in range(1, n_cols + 1):
        for i in range(1, n_rows + 1):
            if gt[j - 1] == pred[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(n_rows, n_cols)) * 100
    else:
        score = D[-1, -1]
    return score


def stt_iou(pred, gt, eps=1e-6):
    frame_start = max(pred['start'], gt['start'])
    frame_end = min(pred['end'], gt['end'])
    if frame_start > frame_end:
        return 0

    tube_intersection = 0
    tube_union = eps
    for frame_id in range(min(pred['start'], gt['start']), max(pred['end'], gt['end'])):
        bbox_pred = bbox_gt = None
        if frame_id in pred['frames'].keys() and pred['frames'][frame_id] is not None:
            bbox_pred = pred['frames'][frame_id]
        if frame_id in gt['frames'].keys() and gt['frames'][frame_id] is not None:
            bbox_gt = gt['frames'][frame_id]

        area_pred = (bbox_pred[2] - bbox_pred[0]) * (bbox_pred[3] - bbox_pred[1]) if bbox_pred is not None else 0
        area_gt = (bbox_gt[2] - bbox_gt[0]) * (bbox_gt[3] - bbox_gt[1]) if bbox_gt is not None else 0

        if bbox_pred is not None and bbox_gt is not None:
            x_start = max(bbox_pred[0], bbox_gt[0])
            y_start = max(bbox_pred[1], bbox_gt[1])
            x_end = min(bbox_pred[2], bbox_gt[2])
            y_end = min(bbox_pred[3], bbox_gt[3])
            intersection = max(x_end - x_start, 0) * max(y_end - y_start, 0)
        else:
            intersection = 0
        union = area_pred + area_gt - intersection

        tube_intersection += intersection
        tube_union += union
    return tube_intersection / tube_union


def sts_iou(pred, gt, eps=1e-6):
    frame_start = max(pred['start'], gt['start'])
    frame_end = min(pred['end'], gt['end'])
    if frame_start > frame_end:
        return 0, 0

    tube_intersection = spatial_intersection = 0
    tube_union = spatial_union = eps
    for frame_id in range(min(pred['start'], gt['start']), max(pred['end'], gt['end'])):
        bbox_pred = bbox_gt = None
        if frame_id in pred['frames'].keys() and pred['frames'][frame_id] is not None:
            bbox_pred = pred['frames'][frame_id]
        if frame_id in gt['frames'].keys() and gt['frames'][frame_id] is not None:
            bbox_gt = gt['frames'][frame_id]
        if bbox_pred is None or bbox_gt is None:
            continue

        area_pred = (bbox_pred[2] - bbox_pred[0]) * (bbox_pred[3] - bbox_pred[1]) if bbox_pred is not None else 0
        area_gt = (bbox_gt[2] - bbox_gt[0]) * (bbox_gt[3] - bbox_gt[1]) if bbox_gt is not None else 0

        if bbox_pred is not None and bbox_gt is not None:
            x_start = max(bbox_pred[0], bbox_gt[0])
            y_start = max(bbox_pred[1], bbox_gt[1])
            x_end = min(bbox_pred[2], bbox_gt[2])
            y_end = min(bbox_pred[3], bbox_gt[3])
            intersection = max(x_end - x_start, 0) * max(y_end - y_start, 0)
        else:
            intersection = 0
        union = area_pred + area_gt - intersection

        tube_intersection += intersection
        tube_union += union
        if area_pred and area_gt:
            spatial_intersection += intersection
            spatial_union += union
    spatio_temporal_iou = tube_intersection / tube_union
    spatial_iou = spatial_intersection / spatial_union
    temporal_iou = temporal_IoU(pred, gt, eps)
    return spatio_temporal_iou, spatial_iou, temporal_iou
