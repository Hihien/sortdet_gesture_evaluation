import json
import numpy as np
from collections.abc import Sequence
from tqdm import tqdm
from evaluation._metrics import *


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
    det_file = 'D:/code/benefactor/phuongdung/detections_dung/out_dung.json'

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

    stats = [np.concatenate(_, 0) for _ in zip(*stats)]
    print(stats[0].shape)
    if len(stats) and stats[0].any():
        p, r, ap, f1 = ap_per_class(*stats, plot=False, save_dir='.', names=classes)
        print(ap.shape)
        ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, mf1, map50, map75, map = p.mean(), r.mean(), f1.mean(), ap50.mean(), ap75.mean(), ap.mean()
        n_gts_per_class = np.bincount(stats[3].astype(np.int64), minlength=len(classes))  # number of gts per class
        n_dets_per_class = np.bincount(stats[2].astype(np.int64), minlength=len(classes))  # number of dets per class
    else:
        p = r = f1 = ap50 = ap75 = ap = mp = mr = mf1 = map50 = map75 = map = 0.0
        n_gts_per_class = np.zeros(1)
        n_dets_per_class = np.zeros(1)

    # Print results
    print(('{:>20}' + '{:>12}' * 9).format('Class', '#Images', '#GTs', '#Dets',
                                           'P', 'R', 'F1',
                                           'mAP@.5', 'mAP@.75', 'mAP@.5:.95'))

    pf = '{:>20}' + '{:>12d}' * 3 + '{:>12.4f}' * 6  # print format
    print(pf.format('all', len(gt_coco['images']), n_gts_per_class.sum(), n_dets_per_class.sum(),
                    mp, mr, mf1, map50, map75, map))

    # Print results per class
    if len(stats):
        for i in range(len(classes)):
            print(pf.format(classes[i], n_imgs_per_class[i], n_gts_per_class[i], n_dets_per_class[i],
                            p[i], r[i], f1[i], ap50[i], ap75[i], ap[i]))

    print()
    print('Confusion matrix')
    print(cm.matrix)


if __name__ == '__main__':
    main()
