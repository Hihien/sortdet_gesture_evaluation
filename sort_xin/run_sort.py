import torch
import numpy as np
from sort import Sort
from evaluation import *


def main():
    output_file = 'out_ch01_20210331085159.txt'
    with open(output_file) as f:
        lines = [_.strip() for _ in f.readlines()]

    dets = []
    det = ''
    for line in lines:
        if line.startswith('D'):
            if len(det):
                det = torch.tensor(eval(det.replace(' ', ',')))
                dets.append(det)
            det = line[line.find(' ') + 1:]
            frame_name = line[: line.find(' ')]
        else:
            det += ' ' + line

    tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)

    # for det in dets:
    for i in range(1, len(dets), 1):
        # convert xywh to x1y1x2y2
        # xy = det[:, :2].clone()
        # wh = det[:, 2:4].clone()
        # det[:, :2] = xy - wh / 2
        # det[:, 2:4] = xy + wh / 2
        # update tracklets
        det = dets[i-1]
        matches, unmatches = tracker.update(det)

        print(matches[:, -1].int().tolist(), unmatches)


if __name__ == '__main__':
    main()
