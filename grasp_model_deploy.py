import numpy as np


class grasp_model():

    def __init__(self):

        pass

    def pred(self, manipulator_before, lwh_list, conf_list):

        knolling_flag = False
        num_item = len(conf_list)
        move_list = np.arange(int(num_item / 2), num_item)

        return move_list, knolling_flag