import pyrealsense2 as rs
import numpy as np
import cv2
import os

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

        cv2.imwrite(img_path + '%012d.png' % epoch, resized_color_image)

        cv2.namedWindow('zzz', 0)
        cv2.resizeWindow('zzz', 1280, 960)
        cv2.imshow('zzz', resized_color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':

    img_path = '../../knolling_dataset/yolo_segmentation_820_real/'
    os.makedirs(img_path, exist_ok=True)
    num_img_start = 20
    num_img_end = 50
    for i in range(num_img_start, num_img_end):
        capture(img_path, i)