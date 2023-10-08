import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

def capture(img_path, epoch):

    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
    # Start streaming
    pipeline.start(config)

    total_pred_result = []
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_colormap_dim = color_image.shape
        resized_color_image = color_image
        # resized_color_image = cv2.line(resized_color_image, (320, 0), (320, 480), (255, 255, 0), 1)
        # resized_color_image = cv2.line(resized_color_image, (0, 240), (640, 240), (255, 255, 0), 1)

        ratio = 34 / 30
        x_ratio = 0.975
        y_ratio = 480 * x_ratio * ratio / 640
        first_point = int((640 - 640 * y_ratio) / 2), int((480 - 480 * x_ratio) / 2)
        second_point = int((640 - 640 * y_ratio) / 2 + int(640 * y_ratio)), int((480 - 480 * x_ratio) / 2) + int(480 * x_ratio)
        print(first_point)
        print(second_point)
        # resized_color_image = cv2.rectangle(resized_color_image, first_point, second_point, (0, 0, 255))

        cv2.imwrite(img_path + '%d.png' % epoch, resized_color_image)

        cv2.namedWindow('zzz', 0)
        cv2.resizeWindow('zzz', 1280, 960)
        cv2.imshow('zzz', resized_color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
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
    out = cv2.VideoWriter('real_box_4.avi', fourcc, fps, (w, h))

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

    cap = cv2.VideoCapture('real_box_4.avi')
    frame_read = 0
    num = 9252
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
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # time.sleep(0.02)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    # img_path = '../../knolling_dataset/yolo_pile_820_real_box/origin_images/'
    # os.makedirs(img_path, exist_ok=True)
    # num_img_start = 60
    # num_img_end = 100
    # for i in range(num_img_start, num_img_end):
    #     capture(img_path, i)

    output_path = '../../knolling_dataset/yolo_pile_830_real_box/origin_images/'
    os.makedirs(output_path, exist_ok=True)
    capture_video()
    # show_save_video(output_path)