import json
from collections.abc import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from evaluation.object_detection_metrics import *


def load_coco(coco_file):
    with open(coco_file) as f:
        return json.load(f)


def coco_annotation_to_bbox(coco_annotation, default=np.empty((0, 5), dtype=np.float32)):
    if isinstance(coco_annotation, Sequence):
        if len(coco_annotation):
            return np.stack([coco_annotation_to_bbox(_) for _ in coco_annotation])
        else:
            return default
    else:
        ret = coco_annotation['bbox']
        if 'score' in coco_annotation.keys():
            ret.append(coco_annotation['score'])
        ret.append(coco_annotation['category_id'] - 1)
        return np.array(ret, dtype=np.float32)


def xywh2xyxy(bbox):
    xy = bbox[:, :2].copy()
    wh = bbox[:, 2:4].copy()
    bbox[:, :2] = xy - wh / 2
    bbox[:, 2:4] = xy + wh / 2
    return bbox


def main():
    gt_file = 'D:/code/benefactor/phuongdung/ground_truth/ground_truth.json'
    det_file = 'D:/code/benefactor/phuongdung/detections_nhat/out_nhat.json'

    gt_coco = load_coco(gt_file)
    det_coco = load_coco(det_file)

    classes = [_['name'] for _ in gt_coco['categories']]
    iouv = np.linspace(0.5, 0.95, 10)
    cm = ConfusionMatrix(nc=len(classes))

    stats = []
    pbar = tqdm(sorted(gt_coco['images'], key=lambda _: _['id']))
    for img in pbar:
        img_id = img['id']
        gts = xywh2xyxy(coco_annotation_to_bbox(
            [_ for _ in gt_coco['annotations'] if _['image_id'] == img_id],
            default=np.empty((0, 5), dtype=np.float32)
        ))
        dets = xywh2xyxy(coco_annotation_to_bbox(
            [_ for _ in det_coco['annotations'] if _['image_id'] == img_id],
            default=np.empty((0, 6), dtype=np.float32)
        ))
        pbar.set_description(f'[{img["file_name"]}] n_gts={len(gts)}, n_dets={len(dets)}')

        gt_classes = gts[:, -1].astype(np.int64) if len(gts) else np.array([])
        det_classes = dets[:, -1].astype(np.int64) if len(dets) else np.array([])
        if len(dets) == 0:
            if len(gts):
                stats.append(
                    (np.zeros((0, len(iouv)), dtype=np.bool_), np.array([]), np.array([]), gt_classes)
                )
            continue

        # if len(classes) == 1:
        #     dets[:, -2] = 0

        correct = np.zeros((len(dets), len(iouv)), dtype=np.bool_)
        if len(gts):
            detected = []

            cm.process_batch(dets, gts)
            # target boxes
            for cls in np.unique(gt_classes):
                gt_inds = np.where(gt_classes == cls)[0]  # target indices
                det_inds = np.where(det_classes == cls)[0]  # prediction indices

                # Search for detections
                if len(det_inds):
                    # Prediction to target ious
                    ious = batch_iou(dets[det_inds, :4], gts[gt_inds, :4])
                    best_ious = ious.max(1)
                    best_inds = ious.argmax(1)

                    # Append detections
                    detected_set = set()
                    for j in np.where(ious > iouv[0])[0]:
                        d = gt_inds[best_inds[j]]
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[det_inds[j]] = best_ious[j] > iouv
                            if len(detected) == len(gts):
                                break

        stats.append((correct, dets[:, -2], det_classes, gt_classes))

    # compute statistics
    n_imgs_per_class = np.array([np.sum([np.any(_[3] == cls) for _ in stats]) for cls in range(len(classes))])

    correct, confs, det_classes, gt_classes = [np.concatenate(_, axis=0) for _ in zip(*stats)]
    if correct.any():
        ap, p, r, f1 = ap_per_class(correct, confs, det_classes, gt_classes,
                                    plot=False, save_dir='.', names=classes)
        print(ap.shape, p.shape)
        ap_50 = ap[:, 0]  # AP@0.5
        ap_75 = ap[:, 5]  # AP@0.75
        ap_ = ap.mean(1)  # AP@0.5:0.95
        n_gts_per_class = np.bincount(gt_classes.astype(np.int64), minlength=len(classes))  # number of gts per class
        n_dets_per_class = np.bincount(det_classes.astype(np.int64), minlength=len(classes))  # number of dets per class
    else:
        p = r = f1 = ap_50 = ap_75 = ap_ = 0.0
        n_gts_per_class = np.zeros(1)
        n_dets_per_class = np.zeros(1)

    results = {
        'Class': ['all'],
        '#Images': [len(gt_coco['images'])],
        '#GTs': [n_gts_per_class.sum()],
        '#Dets': [n_dets_per_class.sum()],
        'P': [p.mean()],
        'R': [r.mean()],
        'F1': [f1.mean()],
        'mAP@0.5': [ap_50.mean()],
        'mAP@0.75': [ap_75.mean()],
        'mAP@0.5:0.95': [ap_.mean()],
    }
    for i in range(len(classes)):
        for col, val in zip(list(results.keys()), [
            classes, n_imgs_per_class, n_gts_per_class, n_dets_per_class,
            p, r, f1, ap_50, ap_75, ap_,
        ]):
            results[col].append(val[i])

    # Print results
    pd.options.display.float_format = '{:,.4f}'.format
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(pd.DataFrame(results).to_string(index=False))
    print()

    print('Confusion matrix')
    print(cm.matrix)


if __name__ == '__main__':
    main()
