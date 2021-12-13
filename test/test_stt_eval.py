from itertools import product

import pandas as pd
import scipy.sparse as sp
import yaml
from tqdm import tqdm

from evaluation.metrics import *
from evaluation.object_detection_metrics import sts_ap_per_class


def batch_stt_iou(preds, gts, eps=1e-6):
    stt_ious = sp.csr_matrix((len(preds), len(gts)))

    pairs = product(enumerate(preds.values()), enumerate(gts.values()))
    pbar = tqdm(pairs, total=len(preds) * len(gts))
    for (pred_id, pred), (gt_id, gt) in pbar:
        frame_start = max(pred['start'], gt['start'])
        frame_end = min(pred['end'], gt['end'])
        if frame_start > frame_end:
            continue
        pbar.set_description(f'[{pred_id}-{gt_id}]')
        stt_ious[pred_id, gt_id] = stt_iou(pred, gt, eps)
    return stt_ious


def batch_sts_iou(preds, gts, eps=1e-6):
    spatio_temporal_ious = sp.csr_matrix((len(preds), len(gts)))
    spatial_ious = sp.csr_matrix((len(preds), len(gts)))
    temporal_ious = sp.csr_matrix((len(preds), len(gts)))

    pairs = product(enumerate(preds.values()), enumerate(gts.values()))
    pbar = tqdm(pairs, total=len(preds) * len(gts))
    for (pred_id, pred), (gt_id, gt) in pbar:
        frame_start = max(pred['start'], gt['start'])
        frame_end = min(pred['end'], gt['end'])
        if frame_start > frame_end:
            continue
        pbar.set_description(f'[{pred_id}-{gt_id}]')
        (spatio_temporal_ious[pred_id, gt_id],
         spatial_ious[pred_id, gt_id],
         temporal_ious[pred_id, gt_id]) = sts_iou(pred, gt, eps)
    return spatio_temporal_ious, spatial_ious, temporal_ious


def main():
    gt_files = ['../data/ground_truths.yaml']
    det_files = ['../data/output_sort_sieu_xin.yaml']
    classes = [0]

    spatial_iouv = np.linspace(0.5, 0.95, 10)
    temporal_iouv = np.linspace(0.1, 0.5, 5)

    stats = []
    for gt_file, det_file in zip(gt_files, det_files):
        gt_yaml = yaml.safe_load(open(gt_file))
        det_yaml = yaml.safe_load(open(det_file))

        gt_classes = np.zeros(len(gt_yaml))
        det_classes = np.zeros(len(det_yaml))
        correct = np.zeros((len(det_yaml), len(spatial_iouv), len(temporal_iouv)), dtype=np.bool_)
        if len(gt_yaml):
            detected = []
            # target boxes
            for cls in np.unique(gt_classes):
                gt_inds = np.where(gt_classes == cls)[0]  # target indices
                det_inds = np.where(det_classes == cls)[0]  # prediction indices

                # Search for detections
                if len(det_inds):
                    # Prediction to target ious
                    ious = batch_sts_iou({k: v for k, v in det_yaml.items() if v['category'] == cls},
                                         {k: v for k, v in gt_yaml.items() if v['category'] == cls})
                    spatio_temporal_tube_ious, spatial_ious, temporal_ious = ious
                    # best_ious = spatio_temporal_tube_ious.max(1).toarray()
                    # best_inds = spatio_temporal_tube_ious.argmax(1)
                    best_inds = np.asarray(spatio_temporal_tube_ious.argmax(1)).flatten()
                    best_spatial_ious = np.asarray(spatial_ious[np.arange(len(best_inds)), best_inds]).flatten()
                    best_temporal_ious = np.asarray(temporal_ious[np.arange(len(best_inds)), best_inds]).flatten()

                    # Append detections
                    detected_set = set()
                    for j in np.nonzero((spatial_ious > spatial_iouv[0]).multiply(temporal_ious > temporal_iouv[0]))[0]:
                        d = gt_inds[best_inds[j]]
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[det_inds[j]] = np.outer(best_spatial_ious[j] > spatial_iouv,
                                                            best_temporal_ious[j] > temporal_iouv)
                            if len(detected) == len(gt_yaml):
                                break
        # confs = [
        #     np.array([bbox[-1] if bbox is not None else 0
        #               for bbox in det['frames'].values()]).mean()
        #     for det in det_yaml.values()
        # ]
        confs = [
            np.array([bbox[-1] for bbox in det['frames'].values() if bbox is not None]).mean()
            for det in det_yaml.values()
        ]
        stats.append((correct, confs, det_classes, gt_classes))

    # compute statistics
    # n_frames_per_class = np.array([sum([sum(_[3] == cls) for _ in stats])
    #                                for cls in range(len(classes))])

    correct, confs, det_classes, gt_classes = [np.concatenate(_, axis=0) for _ in zip(*stats)]
    if correct.any():
        ap, p, r, f1 = sts_ap_per_class(correct, confs, det_classes, gt_classes,
                                        plot=False, save_dir='.', names=classes)
        ap_50_10 = ap[:, 0, 0]  # AP@0.5@0.1
        ap_75_30 = ap[:, 5, 2]  # AP@0.75@0.3
        ap__ = ap.mean((1, 2))  # AP@0.5:0.95:0.05@0.1:0.5:0.1
        n_gts_per_class = np.bincount(gt_classes.astype(np.int64), minlength=len(classes))  # number of gts per class
        n_dets_per_class = np.bincount(det_classes.astype(np.int64), minlength=len(classes))  # number of dets per class
    else:
        p = r = f1 = ap_50_10 = ap_75_30 = ap__ = 0.0
        n_gts_per_class = np.zeros(1)
        n_dets_per_class = np.zeros(1)

    results = {
        'Class': ['all'],
        '#GTs': [n_gts_per_class.sum()],
        '#Dets': [n_dets_per_class.sum()],
        'P': [p.mean()],
        'R': [r.mean()],
        'F1': [f1.mean()],
        'mAP@0.5@0.1': [ap_50_10.mean()],
        'mAP@0.75@0.3': [ap_75_30.mean()],
        'mAP@0.5:0.95@0.1:0.5': [ap__.mean()],
    }
    for i in range(len(classes)):
        for col, val in zip(list(results.keys()), [
            classes, n_gts_per_class, n_dets_per_class,
            p, r, f1, ap_50_10, ap_75_30, ap__,
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


if __name__ == '__main__':
    main()
