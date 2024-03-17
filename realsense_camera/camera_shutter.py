# import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

def capture_img(output_path, begin_num=0):
    cap = cv2.VideoCapture(8)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # set the resolution width
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    num = begin_num
    while True:

        ret, resized_color_image = cap.read()
        resized_color_image = cv2.flip(resized_color_image, -1)
        if ret == True:
            # frame = cv2.flip(frame, 0)
            cv2.namedWindow('RealSense', 0)
            cv2.resizeWindow('RealSense', 1280, 960)
            cv2.imshow('RealSense', resized_color_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(output_path + '%012d.png' % num, resized_color_image)
                print('image saved', num) # Toggle recording state
                num += 1
        else:
            break

def capture_video():

    cap = cv2.VideoCapture(8)
    # set the resolution height
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # set the resolution width
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('real_sundry_5.avi', fourcc, fps, (w, h))

    recording = False
    print('To start recording, please press "s"')
    num = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # frame = cv2.flip(frame, 0)
            cv2.namedWindow('frame', 0)
            cv2.resizeWindow('frame', 1280, 960)
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                print('start recording', num)
                recording = not recording  # Toggle recording state
            elif key == ord('p'):
                print('pause recording', num)
                recording = not recording  # Toggle recording state

            if recording:
                out.write(frame)
            num += 1
        else:
            break

    # Release everything if job is finishedq
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def show_save_video(output_path):

    cap = cv2.VideoCapture('real_sundry_5.avi')
    frame_read = 0
    num = 5000
    while (cap.isOpened()):
        ret, frame = cap.read()

        flag = np.random.random()
        if flag < 0.5:
            kernel_size = np.random.choice([1, 3, 5])
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_read % 1 == 0:
            print('this is save num', num)
            cv2.imwrite(output_path + '%012d.png' % num, frame)
            num += 1
        frame_read += 1
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.02)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    # img_path = '../../knolling_dataset/yolo_pile_820_real_box/origin_images/'
    # os.makedirs(img_path, exist_ok=True)
    # num_img_start = 60
    # num_img_end = 100
    # for i in range(num_img_start, num_img_end):
    #     capture(img_path, i)

    output_path = '../../knolling_dataset/yolo_seg_real_sundry_227/star_images/'
    os.makedirs(output_path, exist_ok=True)

    begin_num = 90
    capture_img(output_path, begin_num)
    # show_save_video(output_path)