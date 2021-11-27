import torch
import numpy as np
from skimage import io
import matplotlib
import cv2
from sort import Sort
import yaml
import time

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import trange


def main():
    colours = np.random.randint(0, 256, (32, 3))
    output_file = '../data/out_ch01_20210331085158.txt'
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
        else:
            det += ' ' + line

    mot_tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

    tracklets = {}
    for frame_id in trange(1, len(dets), 1):
        det = dets[frame_id - 1]
        frame = f'D:/code/sort_xin/data/ch01_20210331085158_split_image/ch01_20210331085158_{str(frame_id).zfill(4)}.png'
        image = cv2.imread(frame)
        window_name = 'Video Tracking'
        if len(det) == 0:
            continue

        matches, _, _ = mot_tracker.update(det)

        for d in matches:
            d = d.int().numpy()
            bbox, track_id = d[:4], d[-1].item()

            if track_id not in tracklets.keys():
                # init if not exist
                tracklets[track_id] = {'start': None,
                                       'end': None,
                                       'bbox': None,
                                       'frames': {}}
            else:
                # set bboxes at missing_frame_ids to None
                for missing_frame_id in range(max(tracklets[track_id]['frames'].keys()) + 1, frame_id):
                    tracklets[track_id]['frames'][missing_frame_id] = None
            # update bbox at current frame_id
            tracklets[track_id]['frames'][frame_id] = d[:4].tolist()

            # visualize
            cv2.rectangle(image,
                          (d[0], d[1]), (d[2], d[3]),
                          thickness=2,
                          color=colours[d[-1].item() % len(colours), :].tolist())
        # cv2.imshow(window_name, cv2.resize(image, None, fx=0.5, fy=0.5))
        # key = cv2.waitKey(1)
        # if key == ord('q'):  # quit
        #     break

    # compute fat bbox
    for track_id in tracklets.keys():
        tracklet = tracklets[track_id]
        tracklets[track_id]['start'] = min(tracklet['frames'].keys())
        tracklets[track_id]['end'] = max(tracklet['frames'].keys())
        all_bboxes = np.stack([bbox for bbox in tracklet['frames'].values()
                               if bbox is not None])
        tracklets[track_id]['bbox'] = [all_bboxes[:, 0].min().item(),
                                       all_bboxes[:, 1].min().item(),
                                       all_bboxes[:, 2].max().item(),
                                       all_bboxes[:, 3].max().item()]

    tracking_output_file = 'out_track.yaml'
    with open(tracking_output_file, 'w') as of:
        yaml.safe_dump(tracklets, of, indent=4, default_flow_style=None, sort_keys=False)


if __name__ == '__main__':
    main()
