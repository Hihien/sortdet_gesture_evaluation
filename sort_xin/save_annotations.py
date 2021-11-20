import os
import yaml
import numpy as np
import xmltodict


def main():
    annotation_dir = 'D:/code/recognition_hand_raising/data/MICA_210331'

    ground_truths = []
    for annotation_file in sorted(os.listdir(annotation_dir)):
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
        bboxes = []
        for obj in objs:
            anno_id = obj['attributes']
            bbox = np.array([[_['x'], _['y']] for _ in obj['polygon']['pt']]).astype(np.int)
            assert bbox.size == 8
            xmin = bbox[:, 0].min()
            xmax = bbox[:, 0].max()
            ymin = bbox[:, 1].min()
            ymax = bbox[:, 1].max()
            x1 = xmin
            y1 = ymin
            width = xmax - xmin
            height = ymax - ymin
            bbox = np.array([x1, y1, width, height])
            bboxes.append(bbox)
        exit()
        print("Hien bao khong phai lam nua")


if __name__ == '__main__':
    main()
