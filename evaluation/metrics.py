import numpy as np

def _as_nparray(*args):
    return tuple(np.array(arg) for arg in args)

def tube_accuracy(pred, gt):
    acc = 0.
    for frame_id in range(max(pred['start'], gt['start']), min(pred['end'], gt['end']) + 1):
        bbox_pred = pred['frames'][frame_id]
        bbox_gt = gt['bbox']
        # acc += accuracy(bbox_pred, bbox_gt)
        acc += accuracy(bbox_gt, bbox_pred)
    return acc / (max(pred['end'], gt['end']) - min(pred['start'], gt['start']) + 1)

def f1_overlap(pred, gt):
    f1 = 0.
    for frame_id in range(max(pred['start'], gt['start']), min(pred['end'], gt['end']) + 1):
        bbox_pred = pred['frames'][frame_id]
        bbox_gt = gt['bbox']
        # f1 += overlap_(bbox_pred, bbox_gt)
        f1 += overlap_(bbox_gt, bbox_pred)
    return f1 / (max(pred['end'], gt['end']) - min(pred['start'], gt['start']) + 1)

def temporal_IoU(pred, gt, eps = 1e-6):
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
        union = area2[j] + area1 - overlap
        union = np.maximum(union, eps)
        IoU = overlap / union

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
        union = area2[j] + area1 - overlap
        union = np.maximum(union, eps)
        IoU = overlap / union

        idx = IoU.argmax()
        if IoU[idx] > iou_thres:
            tp += 1

    return tp / n_true
