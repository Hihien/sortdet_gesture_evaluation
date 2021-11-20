import cv2
import os
import numpy as np
import torch

colours = np.random.randint(0, 256, (32, 3))
data_dir = "D:/code/sort_xin/data"
folders = ['ch01_20210331085158_split_image']
anno_file = 'D:/code/sort_xin/data/anno_ch01_20210331085158_of.txt'
window_name = 'Video Tracking'

with open(anno_file, 'r') as f:
    lines = [_.strip() for _ in f.readlines()]

for folder in folders:
    folder = os.path.join(data_dir, folder)
    data = []
    for line in lines:
        dat = np.fromstring(line, dtype = int, sep = ' ')
        data.append(dat)

    for image in os.listdir(folder):
        if not image.endswith('.png'):
            continue
        image = os.path.join(folder, image)
        img = cv2.imread(image)

        for i in range(len(data)):
            dat = data[i]
            start_frame = dat[1]
            start_image = f'{folder}\\ch01_20210331085158_{str(dat[1]).zfill(4)}.png'

            end_frame = dat[2]

            if image == start_image:
                cv2.rectangle(img,
                              (dat[3], dat[4]), (dat[5], dat[6]),
                              thickness=2,
                              color=colours[dat[-1].item() % len(colours), :].tolist())
                dat[1] +=1

            cv2.imshow(window_name, cv2.resize(img, None, fx=0.5, fy=0.5))
            key = cv2.waitKey(1)

            if key == ord('q'):  # quit
                break
            if dat[1] == dat[2]:
                break
    # for i in range(len(data)):
    #     dat = data[i]
    #     end_frame = dat[2]
    #     end_image = f'{folder}/ch01_20210331085158_{str(dat[2]).zfill(4)}.png'
    #
    #     for image in os.listdir(folder):
    #         if not image.endswith('.png'):
    #             continue
    #         image = os.path.join(folder, image)
    #         img = cv2.imread(image)
    #         start_frame = dat[1]
    #         start_image = f'{folder}\\ch01_20210331085158_{str(dat[1]).zfill(4)}.png'
    #
    #         if image == start_image:
    #             cv2.rectangle(img,
    #                           (dat[3], dat[4]), (dat[5], dat[6]),
    #                           thickness=2,
    #                           color=colours[dat[-1].item() % len(colours), :].tolist())
    #             dat[1] += 1
    #
    #         cv2.imshow(window_name, cv2.resize(img , None, fx=0.5, fy=0.5))
    #         key = cv2.waitKey(1)
    #
    #         if key == ord('q'):  # quit
    #             break
