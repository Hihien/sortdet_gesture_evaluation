import os
import cv2
import shutil
import pathlib


def split_video(video_path):
    stream = cv2.VideoCapture(video_path)

    output_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    video_name = video_name[:video_name.rfind('.')]

    save_folder = pathlib.Path(f'./{output_dir}/{video_name}_split_image/')
    shutil.rmtree(str(save_folder), ignore_errors=True)
    save_folder.mkdir(parents=True, exist_ok=True)

    total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    length = len(str(total_frames)) + 1

    i = 1
    while True:
        grabbed, frame = stream.read()

        if not grabbed:
            print(f'Split totally {i + 1} images from video.')
            break

        save_path = f'{save_folder}/{video_name}_{str(i).zfill(4)}.png'
        # save_path = f'{save_folder}/video_2_{str(i)}.jpg'
        cv2.imwrite(save_path, frame)

        i += 1

    # count = -1
    # current_frame = 0
    # total = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    # while current_frame < total:
    #     current_frame += 1
    #     grabbed, frame = stream.read()
    #     if current_frame % 5 == 0:
    #         count += 1
    #         save_path = f'{save_folder}/video_2_{str(count)}.jpg'
    #         cv2.imwrite(save_path, frame)

    saved_path = os.path.dirname(save_path)
    print(f'Split images saved in {saved_path}')

    return saved_path

def main():
    split_video('D:/code/sort_xin/data/ch01_20210331085158.mp4')

if __name__ == '__main__':
    main()