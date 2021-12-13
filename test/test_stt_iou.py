from itertools import product

import scipy.sparse as sp
import yaml
from tqdm import tqdm

from evaluation.metrics import *


def batch_stt_iou(preds, gts, eps=1e-6):
    # accs = sp.csr_matrix((len(preds), len(gts)))
    # temporal_ious = sp.csr_matrix((len(preds), len(gts)))
    # f1s = sp.csr_matrix((len(preds), len(gts)))
    # edits = sp.csr_matrix((len(preds), len(gts)))
    stt_ious = sp.csr_matrix((len(preds), len(gts)))

    pairs = product(enumerate(preds.values()), enumerate(gts.values()))
    pbar = tqdm(pairs, total=len(preds) * len(gts))
    for (pred_id, pred), (gt_id, gt) in pbar:
        frame_start = max(pred['start'], gt['start'])
        frame_end = min(pred['end'], gt['end'])
        if frame_start > frame_end:
            continue
        pbar.set_description(f'[{pred_id}-{gt_id}]')
        # accs[pred_id, gt_id] = tube_accuracy(pred, gt)
        # temporal_ious[pred_id, gt_id] = temporal_IoU(pred, gt)
        # f1s[pred_id, gt_id] = f1_overlap(pred, gt)
        # edits[pred_id, gt_id] = edit_score(pred, gt)
        stt_ious[pred_id, gt_id] = stt_iou(pred, gt, eps)
    return stt_ious


def main():
    gt_file = '../data/ground_truths.yaml'
    det_file = '../data/output_sort_sieu_xin.yaml'
    stt_iou_threshold = 0.1

    gt_yaml = yaml.safe_load(open(gt_file))
    det_yaml = yaml.safe_load(open(det_file))

    stt_ious = batch_stt_iou(det_yaml, gt_yaml)
    print(np.nonzero(stt_ious.max(1) > stt_iou_threshold)[0].__len__())
    exit()

    non_zero_inds = np.stack(stt_ious.nonzero()).T
    # for i, j in non_zero_inds:
    #     print(i, j, stt_ious[i, j])
    # print()
    print('number of matches:', len(non_zero_inds))

    for gt_id, gt in gt_yaml.items():
        best_stt_iou_ind = stt_ious[:, gt_id - 1].argmax()
        match_inds = np.nonzero(stt_ious[:, gt_id - 1] >= stt_iou_threshold)[0]
        match_stt_ious = stt_ious[match_inds, gt_id - 1].toarray().flatten()
        print(f'Groundtruth {gt_id} is matched with detections {match_inds + 1} with stt_iou_scores {match_stt_ious}.')

    exit()


if __name__ == '__main__':
    main()
