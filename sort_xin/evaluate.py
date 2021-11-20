import yaml
import numpy as np
from evaluation.metrics import *


def main():
    annotation_file = '../data/anno_ch01_20210331085158_of.txt'
    prediction_file = 'out_track.yaml'

    ground_truths = []
    with open(annotation_file) as f:
        annotations = np.array([_.strip().split() for _ in f.readlines()])
        annotations = annotations.astype(np.int64)
        for annotation in annotations:
            ground_truths.append({
                'start': annotation[1].item(),
                'end': annotation[2].item(),
                'bbox': annotation[3:7].tolist(),
            })
    print('Loaded ground-truths')
    # print(ground_truths)

    with open(prediction_file) as f:
        predictions = yaml.safe_load(f)
    print('Loaded predictions')
    # print(predictions)
    print(predictions[1])
    exit()
    # test thu
    pred = predictions[1]
    gt = ground_truths[9]
    print('pred:', pred)
    print('gt:', gt)
    
    for frame_id in range(max(pred['start'], gt['start']), min(pred['end'], gt['end']) + 1):
        bbox_pred = pred['frames'][frame_id]
        bbox_gt = gt['bbox']


    print('accuracy:', tube_accuracy(pred, gt))
    print('temporal_IoU: ', temporal_IoU(pred, gt))
    print('f1_overlap: ', f1_overlap(pred, gt))


if __name__ == '__main__':
    main()
