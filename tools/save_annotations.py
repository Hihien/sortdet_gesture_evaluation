import os

import numpy as np
import xmltodict
import yaml
from tqdm import tqdm


def main():
    annotation_dir = 'D:/datasets/lan2021/MICA_210331'
    output_file = '../data/ground_truths.yaml'
    sampling_rate = 5
    max_age = 5

    tracklets = {}
    for annotation_file in tqdm(list(filter(lambda _: _.endswith('.xml'), sorted(os.listdir(annotation_dir))))):
        frame_id = int(annotation_file[annotation_file.rfind('_') + 1:annotation_file.rfind('.')]) - 1

        with open(os.path.join(annotation_dir, annotation_file)) as f:
            xml_data = '\n'.join(f.readlines())
        xml_dict = xmltodict.parse(xml_data)['annotation']
        if isinstance(xml_dict['object'], list):
            objs = xml_dict['object']
        else:
            objs = [xml_dict['object']]
        objs = [_ for _ in objs if _['name'] in ['raising_hand', 'raising hand']]
        if len(objs) == 0:
            continue

        for obj in objs:
            if int(obj['deleted']) or not obj['attributes'].isdigit() or len(obj['polygon']['pt']) != 4:
                continue
            subject_id = int(obj['attributes'])
            bbox = np.array(
                [[_['x'], _['y']] for _ in obj['polygon']['pt']]
            ).astype(np.int64)
            bbox = np.clip(bbox, 0, None)
            xmin = bbox[:, 0].min()
            xmax = bbox[:, 0].max()
            ymin = bbox[:, 1].min()
            ymax = bbox[:, 1].max()
            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]  # x1y1x2y2

            matched_track_ids = [track_id for track_id, tracklet in tracklets.items()
                                 if tracklet['subject_id'] == subject_id
                                 and frame_id * sampling_rate - tracklet['end'] <= max_age]
            if not len(matched_track_ids):
                # init if not exist
                track_id = len(tracklets) + 1
                tracklets[track_id] = {
                    'subject_id': subject_id,
                    'start': frame_id * sampling_rate,
                    'end': (frame_id + 1) * sampling_rate - 1,
                    'bbox': None,
                    'frames': {
                        fid: bbox for fid in range(frame_id * sampling_rate, (frame_id + 1) * sampling_rate)
                    },
                }
            else:
                track_id = matched_track_ids[0]
                tracklet = tracklets[track_id]

                # set bboxes at missing frames to None
                tracklet['frames'].update({
                    fid: None for fid in range(tracklet['end'] + 1, frame_id * sampling_rate)
                })
                # update new bbox
                tracklet['frames'].update({
                    fid: bbox for fid in range(frame_id * sampling_rate, (frame_id + 1) * sampling_rate)
                })
                # update new end frame_id
                tracklet['end'] = (frame_id + 1) * sampling_rate - 1

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
