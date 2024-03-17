import cv2
import numpy as np
import os

def change_name(source_path, target_path, start_idx, end_idx):

    label_num = 100
    sol_num = 12

    os.makedirs(target_path + 'images/', exist_ok=True)
    input_path = target_path + 'img_input/'
    output_path = target_path + 'img_output/'
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    num = 0
    for i in range(label_num):

        for j in range(sol_num):

            orig_img = cv2.imread(target_path + 'origin_images/label_%d_%d.png' % (i, j))
            img_input = orig_img[:, :640, :]
            img_output = orig_img[:, 640:, :]
            # print('here')
            cv2.imwrite(input_path + '%d.png' % num, img_input)
            cv2.imwrite(output_path + '%d.png' % num, img_output)

            num += 1

    pass

if __name__ == "__main__":

    source_path = '../../../knolling_dataset/VAE_314/'
    target_path = '../../../knolling_dataset/VAE_314/'
    start_idx = 0
    end_idx = 1000
    change_name(source_path=source_path, target_path=target_path, start_idx=start_idx, end_idx=end_idx)