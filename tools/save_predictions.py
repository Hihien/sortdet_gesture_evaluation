import os
import re

import numpy as np
import yaml
from tqdm import tqdm

from sort_xin.sort import Sort


def load_prediction(prediction_output_file):
    """Read Hien's retard format"""
    with open(prediction_output_file) as f:
        lines = [_.rstrip() for _ in f.readlines()]
    dets = []
    det = ''
    for line in lines:
        if not line.startswith(' '):
            if len(det):
                if len(det) > 2:
                    det = re.sub(r' +', ' ', det)
                    det = re.sub(r'\[ +', '[', det)
                    det = re.sub(r'\[ +\[', '[[', det)
                    det = re.sub(r' +\]', ']', det)
                    det = re.sub(r'\] +\]', ']]', det)
                    det = np.array(eval(det.replace(' ', ',')), dtype=np.float64)
                else:
                    det = np.empty((0, 5), dtype=np.float64)
                dets.append(det)
            det = line[line.find(' ') + 1:]
        else:
            det += ' ' + line
    return dets


def main():
    prediction_output_file = '../data/out_ch01_20210331085158.txt'
    output_file = '../data/output_sort_sieu_xin.yaml'

    dets = load_prediction(prediction_output_file)

    tracker = Sort(max_age=20, min_hits=0, iou_threshold=0.3)

    tracklets = {}
    for frame_id, det in tqdm(enumerate(dets), total=len(dets)):
        det, _, _ = tracker.update(det)
        for d in det:
            bbox = d[:-1].round(decimals=4).tolist()  # x1y1x2y2conf
            track_id = int(d[-1])

            if track_id not in tracklets.keys():
                # init if not exist
                tracklets[track_id] = {
                    'start': frame_id,
                    'end': frame_id,
                    'bbox': None,
                    'frames': {
                        frame_id: bbox
                    }
                }
            else:
                tracklet = tracklets[track_id]

                # set bboxes at missing frames to None
                tracklet['frames'].update({
                    fid: None for fid in range(tracklet['end'] + 1, frame_id)
                })
                # update new bbox
                tracklet['frames'].update({
                    frame_id: bbox
                })
                # update new end frame_id
                tracklet['end'] = frame_id

    # compute fat bbox
    for tracklet in tracklets.values():
        all_bboxes = np.stack([bbox for bbox in tracklet['frames'].values()
                               if bbox is not None])
        tracklet['bbox'] = [all_bboxes[:, 0].min().item(),
                            all_bboxes[:, 1].min().item(),
                            all_bboxes[:, 2].max().item(),
                            all_bboxes[:, 3].max().item()]

    # save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as of:
        yaml.safe_dump(tracklets, of, indent=4, default_flow_style=None, sort_keys=False)


if __name__ == '__main__':
    main()
