from collections.abc import Sequence

import cv2


def time_to_frame_id(time, fps):
    if not isinstance(time, Sequence):
        time = [time]
    while len(time) < 3:
        time = [0] + list(time)
    sec = 3600 * time[0] + 60 * time[1] + time[2]
    return round(sec * fps)


def main():
    video_file = '../lan2021/ch01_20210331084200.mp4'

    # 00:09:59 -> 00:11:34
    start_time = (0, 9, 59)
    end_time = (0, 11, 34)

    ffmpeg_frame_offset = 2  # empirically
    ffmpeg_sampling_rate = 5  # cut 1 out of 5 frames

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 14977 -> 17347
    start_frame_id = time_to_frame_id(start_time, fps) + ffmpeg_frame_offset
    end_frame_id = time_to_frame_id(end_time, fps) + ffmpeg_frame_offset

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
    for frame_id in range(start_frame_id, end_frame_id):
        _, img = cap.read()
        if (frame_id - start_frame_id) % ffmpeg_sampling_rate == 0:
            output_frame_id = (frame_id - start_frame_id) // ffmpeg_sampling_rate + 1
            print(f'Frame {frame_id:05d} saved as {output_frame_id:04d}.jpg')
        else:
            print(f'Frame {frame_id:05d} was skipped')
        # Thích lưu ra hay làm gì thì làm
    cap.release()


if __name__ == '__main__':
    main()
