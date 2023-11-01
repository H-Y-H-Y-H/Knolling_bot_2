import numpy as np
import torch
import cv2



def test_print_canvas():
    data_root = "../../../knolling_dataset/learning_data_1019/"
    data_raw = np.loadtxt(data_root + 'num_30_after_0.txt')[:8, :60].reshape(-1, 10, 6)

    factor = 1
    padding_value = 255

    tar_lw = data_raw[:, :, 2:4]
    tar_pos = data_raw[:, :, :2]
    tar_cls = data_raw[:, :, 5]
    num_cls = np.unique(tar_cls.flatten()).astype(np.int32)
    mm2px = 530 / (0.34 * factor)

    corner_topleft = 0
    corner_downright = 0

    print('test', np.clip(2, 1, None))

    tar_pos_px = tar_pos * mm2px
    tar_lw_px = tar_lw * mm2px

    data = np.concatenate(((tar_pos_px[:, :, 0] - tar_lw_px[:, :, 0] / 2)[:, :, np.newaxis],
                            (tar_pos_px[:, :, 1] - tar_lw_px[:, :, 1] / 2)[:, :, np.newaxis],
                             (tar_pos_px[:, :, 0] + tar_lw_px[:, :, 0] / 2)[:, :, np.newaxis],
                              (tar_pos_px[:, :, 1] + tar_lw_px[:, :, 1] / 2)[:, :, np.newaxis],
                           ), axis=2).astype(np.int32)

    # Define the dimensions of the canvas
    canvas_width, canvas_height = int(640 / factor), int(480 / factor)

    # Iterate through each rectangle and draw them on the canvas
    for i in range(data.shape[0]):
        canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

        # x_index = np.concatenate([np.arange(start, end + 1) for start, end in zip(data[i, :, 0], data[i, :, 2])])
        # y_index = np.concatenate([np.arange(start, end + 1) for start, end in zip(data[i, :, 1], data[i, :, 3])])
        # canvas[x_index, y_index] += 255

        for j in range(data.shape[1]):
            corner_data = data[i, j]
            canvas[corner_data[0]:corner_data[2], corner_data[1]:corner_data[3]] += padding_value
            # canvas = cv2.rectangle(canvas, (corner_data[0], corner_data[1]), (corner_data[2], corner_data[3]), 255, -1)

        over_lap_num = len(canvas > padding_value)

        cv2.namedWindow('zzz', 0)
        # cv2.resizeWindow('zzz', 1280, 960)
        cv2.imshow('zzz', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Print the resulting canvas
    print(canvas)

def test_max_pixel():

    factor = 3

    data_root = "../../../knolling_dataset/learning_data_1019/"
    data_raw = np.loadtxt(data_root + 'num_30_after_0.txt')[:100, :60].reshape(-1, 10, 6)

    tar_lw = data_raw[:, :, 2:4]
    tar_pos = data_raw[:, :, :2]
    tar_cls = data_raw[:, :, 5]
    num_cls = np.unique(tar_cls.flatten()).astype(np.int32)
    mm2px = 530 / (0.34 * factor)

    canvas_x = 480 / factor
    canvas_y = 640 / factor

    tar_pos_px = tar_pos * mm2px
    tar_lw_px = tar_lw * mm2px
    max_x = np.max(tar_pos_px[:, :, 0])
    max_y = np.max(tar_pos_px[:, :, 1])

    print('this is max x', max_x)
    print('this is max y', max_y)



if __name__ == "__main__":

    test_print_canvas()
    # test_max_pixel()