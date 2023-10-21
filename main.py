import numpy as np

from utils import *
from ENV.task.KnollingEnv import knolling_env
from ENV.robot.KnollingRobot import knolling_robot
from ASSET.arrange_model_deploy import *
from ASSET.visual_perception import *
import cv2

class knolling_main():

    def __init__(self, para_dict=None, knolling_para=None, lstm_dict=None, arrange_dict=None):
        # super(knolling_main, self).__init__(para_dict=para_dict, knolling_para=knolling_para, lstm_dict=lstm_dict, arrange_dict=arrange_dict)

        self.para_dict = para_dict
        self.knolling_para = knolling_para

        self.task = knolling_env(para_dict=para_dict, lstm_dict=lstm_dict)
        self.robot = knolling_robot(para_dict=para_dict, knolling_para=knolling_para)

        self.success_manipulator_after = []
        self.success_manipulator_before = []
        self.success_lwh = []
        self.success_num = 0
        self.finished_index = []
        self.rest_index = np.arange(self.para_dict['boxes_num'])

        if self.para_dict['use_knolling_model'] == True:
            self.arrange_dict = arrange_dict
            self.arrange_model = Arrange_model(para_dict=para_dict, arrange_dict=arrange_dict, max_num=self.para_dict['boxes_num'])

    def get_candidate_index(self, images, arrangement_num=8):

        num_row = 2
        num_col = int(arrangement_num / num_row)

        for i , image in enumerate(images):
            images[i] = cv2.resize(images[i], dsize=(320, 240), interpolation=cv2.INTER_CUBIC)

        image_height, image_width, _ = images[0].shape

        # Create a canvas to display the images
        canvas_height = num_row * image_height
        canvas_width = num_col * image_width
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Paste the images onto the canvas
        for i, image in enumerate(images):
            row = i // num_col
            col = i % num_col
            canvas[row * image_height:(row + 1) * image_height, col * image_width:(col + 1) * image_width] = image

        # Function to handle mouse clicks
        # selected_image_index = 0
        def mouse_click(event, x, y, flags, param):
            # global selected_image_index
            if event == cv2.EVENT_LBUTTONDOWN:
                print('mouse clicked!')
                col = x // image_width
                row = y // image_height
                print('this is col', col)
                print('this is row', row)
                self.selected_image_index = row * num_col + col
                # print(self.selected_image_index)
                # return selected_image_index

        # Create a window and set the mouse callback function
        cv2.namedWindow('Candidate Arrangement')
        cv2.setMouseCallback('Candidate Arrangement', mouse_click)

        # Display the canvas with all images
        cv2.imshow('Candidate Arrangement', canvas)
        # while True:
        #     # key = cv2.waitKey(1) & 0xFF
        if cv2.waitKey(0):

            print('this is selected index', self.selected_image_index)
            # Close the window
            cv2.destroyAllWindows()

            # Print the selected image index (0-based)
            if self.selected_image_index is not None:
                print(f"Selected image index: {self.selected_image_index}")
            else:
                print("No image selected.")

            output_img_path = self.para_dict['data_source_path'] + 'sim_images/total.png'
            cv2.imwrite(output_img_path, canvas)

            return self.selected_image_index

    def get_knolling_data(self, pos_before, ori_before, lwh_list, success_index, offline_flag=True):

        success_index = np.arange(len(pos_before))

        if self.para_dict['use_knolling_model'] == True:

            if offline_flag == True:

                demo_data = np.loadtxt('./knolling_demo/num_10_after.txt')[0].reshape(-1, 5)
                record_data = np.loadtxt('./knolling_demo/num_10_lwh.txt').reshape(-1, 5)
                lwh_list_classify = lwh_list
                # lwh_list_classify = record_data[:, 2:4]
                # pos_before = np.concatenate((record_data[:, :2], np.ones((len(record_data), 1)) * 0.006), axis=1)
                # ori_before = np.concatenate((np.zeros((len(demo_data), 2)), record_data[:, -1].reshape(len(record_data), 1)), axis=1)

                recover_config = np.concatenate((demo_data[:, :2],
                                                 np.ones(len(demo_data)).reshape(len(demo_data), 1) * 0.006,
                                                 np.zeros((len(demo_data), 2)),
                                                 demo_data[:, -1].reshape(len(demo_data), 1),
                                                 demo_data[:, 2:4],
                                                 np.ones(len(demo_data)).reshape(len(demo_data), 1) * 0.016), axis=1)
                self.task.recover_objects(config_data=recover_config)

                manipulator_before = np.concatenate((pos_before, ori_before), axis=1)
                manipulator_after = recover_config[:, :6]

            else:

                if self.success_num > 0:
                    create_pos = np.concatenate((self.success_manipulator_after[:, :3], pos_before[self.success_num:, :]), axis=0)
                    create_ori = np.concatenate((self.success_manipulator_after[:, 3:6], ori_before[self.success_num:, :]), axis=0)
                    create_lwh = np.concatenate((self.success_lwh[:, :3], lwh_list[self.success_num:, :]), axis=0)
                    create_ori[:len(self.success_manipulator_after), 2] = 0
                else:
                    create_pos = np.copy(pos_before)
                    create_ori = np.copy(ori_before)
                    create_lwh = np.copy(lwh_list)

                ################### recreate the new object ####################
                self.task.create_objects(pos_data=create_pos, ori_data=create_ori, lwh_data=create_lwh)
                ################### recreate the new object ####################

                arrangement_num = 10  # provide several candidates to select the best output
                candidate_img = []
                candidate_after = []
                candidate_before = []
                candidate_lwh = []
                self.before_arrange_state = p.saveState()
                for i in range(arrangement_num):
                    # input_index = np.setdiff1d(np.arange(len(pos_before)), success_index)
                    input_index = np.copy(success_index)
                    pos_before_input = pos_before.astype(np.float32)
                    ori_before_input = ori_before.astype(np.float32)
                    lwh_list_input = lwh_list.astype(np.float32)
                    ori_after = np.zeros((len(ori_before_input), 3))

                    #################### exchange the length and width randomly enrich the input ##################
                    for j in range(len(input_index)):
                        if np.random.random() < 0.5:
                            temp = lwh_list_input[j, 1]
                            lwh_list_input[j, 1] = lwh_list_input[j, 0]
                            lwh_list_input[j, 0] = temp
                            ori_after[j, 2] += np.pi / 2
                    #################### exchange the length and width randomly enrich the input ##################

                    # input include all objects(finished, success, fail),
                    pos_after = self.arrange_model.pred(pos_before_input, ori_before_input, lwh_list_input, input_index)
                    # manipulator_before = np.concatenate((pos_before_input[input_index], ori_before_input[input_index]), axis=1)
                    # manipulator_after = np.concatenate((pos_after[input_index].astype(np.float32), ori_after), axis=1)
                    # lwh_list_classify = lwh_list_input[input_index]

                    manipulator_before = np.concatenate((pos_before_input, ori_before_input), axis=1)
                    manipulator_after = np.concatenate((pos_after.astype(np.float32), ori_after), axis=1)
                    lwh_list_classify = np.copy(lwh_list_input)
                    # rotate_index = np.where(lwh_list_classify[:, 1] > lwh_list_classify[:, 0])[0]
                    # # manipulator_after[rotate_index, -1] += np.pi / 2
                    # ##################### add offset to the knolling data #####################
                    manipulator_after[:, 0] += self.arrange_dict['arrange_x_offset']
                    manipulator_after[:, 1] += self.arrange_dict['arrange_y_offset']
                    # ##################### add offset to the knolling data #####################

                    recover_config = np.concatenate((manipulator_after, lwh_list_classify), axis=1)
                    self.task.recover_objects(config_data=recover_config, recover_index=input_index)

                    manipulator_before = manipulator_before[input_index]
                    manipulator_after = manipulator_after[input_index]
                    lwh_list_classify = lwh_list_classify[input_index]

                    candidate_img.append(self.task.get_obs(look_flag=True, epoch=i, img_path=None))
                    candidate_before.append(manipulator_before)
                    candidate_after.append(manipulator_after)
                    candidate_lwh.append(lwh_list_classify)

                candidate_index = self.get_candidate_index(candidate_img, arrangement_num)
                manipulator_before = candidate_before[candidate_index]
                manipulator_after = candidate_after[candidate_index]
                lwh_list_classify = candidate_lwh[candidate_index]


            p.restoreState(self.before_arrange_state)
            print('here')

        else:
            # determine the center of the tidy configuration
            if len(self.lwh_list) <= 2:
                print('the number of item is too low, no need to knolling!')

            Manual_classfier = Manual_config(self.knolling_para)
            lwh_list_classify, pos_before_classify, ori_before_classify, all_index_classify, transform_flag_classify, crowded_index_classify = Manual_classfier.judge(
                lwh_list, pos_before, ori_before, success_index)
            Manual_classfier.get_manual_para(lwh_list_classify, all_index_classify, transform_flag_classify)
            pos_after_classify, ori_after_classify = Manual_classfier.calculate_block()
            # after this step the length and width of one box in self.lwh_list may exchanged!!!!!!!!!!!
            # but the order of self.lwh_list doesn't change!!!!!!!!!!!!!!
            # the order of pos after and ori after is based on lwh list!!!!!!!!!!!!!!

            ################## change order based on distance between boxes and upper left corner ##################
            order = change_sequence(pos_before_classify)
            pos_before_classify = pos_before_classify[order]
            ori_before_classify = ori_before_classify[order]
            lwh_list_classify = lwh_list_classify[order]
            pos_after_classify = pos_after_classify[order]
            ori_after_classify = ori_after_classify[order]
            crowded_index_classify = crowded_index_classify[order]
            ################## change order based on distance between boxes and upper left corner ##################

            x_low = np.min(pos_after_classify, axis=0)[0]
            x_high = np.max(pos_after_classify, axis=0)[0]
            y_low = np.min(pos_after_classify, axis=0)[1]
            y_high = np.max(pos_after_classify, axis=0)[1]
            center = np.array([(x_low + x_high) / 2, (y_low + y_high) / 2, 0])
            x_length = abs(x_high - x_low)
            y_length = abs(y_high - y_low)
            # print(x_low, x_high, y_low, y_high)
            if self.knolling_para['random_offset'] == True:
                self.knolling_para['total_offset'] = np.array([random.uniform(self.task.x_low_obs + x_length / 2, self.task.x_high_obs - x_length / 2),
                                              random.uniform(self.task.y_low_obs + y_length / 2, self.task.y_high_obs - y_length / 2), 0.0])
            else:
                pass
            pos_after_classify += np.array([0, 0, 0.006])
            pos_after_classify = pos_after_classify + self.knolling_para['total_offset']

            ########## after generate the neat configuration, pay attention to the difference of urdf ori and manipulator after ori! ############
            items_ori_list_arm = np.copy(ori_after_classify)
            for i in range(len(lwh_list_classify)):
                if lwh_list_classify[i, 0] <= lwh_list_classify[i, 1]:
                    ori_after_classify[i, 2] += np.pi / 2
            ########## after generate the neat configuration, pay attention to the difference of urdf ori and manipulator after ori! ############

            manipulator_before = np.concatenate((pos_before_classify, ori_before_classify), axis=1)
            manipulator_after = np.concatenate((pos_after_classify, ori_after_classify), axis=1)
            print('this is manipulator after\n', manipulator_after)

        return manipulator_before, manipulator_after, lwh_list_classify

    def clean_grasp(self):
        if self.para_dict['real_operate'] == False:
            gripper_width = 0.020
            gripper_height = 0.034
        else:
            gripper_width = 0.024
            gripper_height = 0.04

        # workbench_center = np.array([(self.x_high_obs + self.x_low_obs) / 2,
        #                              (self.y_high_obs + self.y_low_obs) / 2])
        offset_low = np.array([0, 0, 0.01])
        offset_high = np.array([0, 0, 0.04])

        crowded_index = np.intersect1d(np.where(self.pred_cls == 0)[0], self.rest_index)

        restrict_gripper_diagonal = np.sqrt(gripper_width ** 2 + gripper_height ** 2)
        gripper_box_gap = 0.006

        while len(crowded_index) >= len(self.rest_index):

            crowded_pos = self.manipulator_before[crowded_index, :3]
            crowded_ori = self.manipulator_before[crowded_index, 3:6]
            theta = self.manipulator_before[crowded_index, -1]
            length_box = self.lwh_list[crowded_index, 0]
            width_box = self.lwh_list[crowded_index, 1]

            trajectory_pos_list = []
            trajectory_ori_list = []
            for i in range(len(crowded_index)):
                break_flag = False
                once_flag = False
                if length_box[i] < width_box[i]:
                    theta[i] += np.pi / 2
                matrix = np.array([[np.cos(theta[i]), -np.sin(theta[i])],
                                   [np.sin(theta[i]), np.cos(theta[i])]])
                target_point = np.array([[(length_box[i] + gripper_height + gripper_box_gap) / 2,
                                          (width_box[i] + gripper_width + gripper_box_gap) / 2],
                                         [-(length_box[i] + gripper_height + gripper_box_gap) / 2,
                                          (width_box[i] + gripper_width + gripper_box_gap) / 2],
                                         [-(length_box[i] + gripper_height + gripper_box_gap) / 2,
                                          -(width_box[i] + gripper_width + gripper_box_gap) / 2],
                                         [(length_box[i] + gripper_height + gripper_box_gap) / 2,
                                          -(width_box[i] + gripper_width + gripper_box_gap) / 2]])
                target_point_rotate = (matrix.dot(target_point.T)).T
                print('this is target point rotate\n', target_point_rotate)
                sequence_point = np.concatenate((target_point_rotate, np.zeros((4, 1))), axis=1)

                t = 0
                for j in range(len(sequence_point)):
                    vertex_break_flag = False
                    for k in range(len(self.manipulator_before)):
                        # exclude itself
                        if np.linalg.norm(crowded_pos[i] - self.manipulator_before[k][:3]) < 0.001:
                            continue
                        restrict_item_k = np.sqrt((self.lwh_list[k][0]) ** 2 + (self.lwh_list[k][1]) ** 2)
                        if 0.001 < np.linalg.norm(sequence_point[0] + crowded_pos[i] - self.manipulator_before[k][
                                                                                       :3]) < restrict_item_k / 2 + restrict_gripper_diagonal / 2 + 0.001:
                            print(np.linalg.norm(sequence_point[0] + crowded_pos[i] - self.manipulator_before[k][:3]))
                            p.addUserDebugPoints([sequence_point[0] + crowded_pos[i]], [[0.1, 0, 0]], pointSize=5)
                            p.addUserDebugPoints([self.manipulator_before[k][:3]], [[0, 1, 0]], pointSize=5)
                            print("this vertex doesn't work")
                            vertex_break_flag = True
                            break
                    if vertex_break_flag == False:
                        print("this vertex is ok")
                        once_flag = True
                        break
                    else:
                        # should change the vertex and try again
                        sequence_point = np.roll(sequence_point, -1, axis=0)
                        print(sequence_point)
                        t += 1
                    if t == len(sequence_point):
                        # all vertex of this cube fail, should change the cube
                        break_flag = True

                # problem, change another crowded cube
                if break_flag == True:
                    if i == len(crowded_index) - 1:
                        print('cannot find any proper vertices to insert, we should unpack the heap!!!')
                        x_high = np.max(self.manipulator_after[:, 0])
                        x_low = np.min(self.manipulator_after[:, 0])
                        y_high = np.max(self.manipulator_after[:, 1])
                        y_low = np.min(self.manipulator_after[:, 1])
                        crowded_x_high = np.max(crowded_pos[:, 0])
                        crowded_x_low = np.min(crowded_pos[:, 0])
                        crowded_y_high = np.max(crowded_pos[:, 1])
                        crowded_y_low = np.min(crowded_pos[:, 1])

                        trajectory_pos_list.append([1, 0])
                        trajectory_pos_list.append([(x_high + x_low) / 2, (y_high + y_low) / 2, offset_high[2]])
                        trajectory_pos_list.append([(x_high + x_low) / 2, (y_high + y_low) / 2, offset_low[2]])
                        trajectory_pos_list.append(
                            [(crowded_x_high + crowded_x_low) / 2, (crowded_y_high + crowded_y_low) / 2,
                             offset_low[2]])
                        trajectory_pos_list.append(
                            [(crowded_x_high + crowded_x_low) / 2, (crowded_y_high + crowded_y_low) / 2,
                             offset_high[2]])
                        trajectory_pos_list.append(self.para_dict['reset_pos'])

                        trajectory_ori_list.append(self.para_dict['reset_ori'])
                        trajectory_ori_list.append(self.para_dict['reset_ori'])
                        trajectory_ori_list.append(self.para_dict['reset_ori'])
                        trajectory_ori_list.append(self.para_dict['reset_ori'])
                        trajectory_ori_list.append(self.para_dict['reset_ori'])
                        trajectory_ori_list.append(self.para_dict['reset_ori'])
                    else:
                        pass
                else:
                    trajectory_pos_list.append([1, 0])
                    print('this is crowded pos', crowded_pos[i])
                    print('this is sequence point', sequence_point)
                    trajectory_pos_list.append(crowded_pos[i] + offset_high + sequence_point[0])
                    trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[0])
                    trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[1])
                    trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[2])
                    trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[3])
                    trajectory_pos_list.append(crowded_pos[i] + offset_low + sequence_point[0])
                    trajectory_pos_list.append(crowded_pos[i] + offset_high + sequence_point[0])
                    # reset the manipulator to read the image
                    trajectory_pos_list.append(self.para_dict['reset_pos'])

                    trajectory_ori_list.append(self.para_dict['reset_ori'])
                    trajectory_ori_list.append(self.para_dict['reset_ori'] + crowded_ori[i])
                    trajectory_ori_list.append(self.para_dict['reset_ori'] + crowded_ori[i])
                    trajectory_ori_list.append(self.para_dict['reset_ori'] + crowded_ori[i])
                    trajectory_ori_list.append(self.para_dict['reset_ori'] + crowded_ori[i])
                    trajectory_ori_list.append(self.para_dict['reset_ori'] + crowded_ori[i])
                    trajectory_ori_list.append(self.para_dict['reset_ori'] + crowded_ori[i])
                    trajectory_ori_list.append(self.para_dict['reset_ori'] + crowded_ori[i])
                    # reset the manipulator to read the image
                    trajectory_ori_list.append([0, math.pi / 2, 0])

                # only once!
                if once_flag == True:
                    break
            last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
            last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
            # trajectory_pos_list = np.asarray(trajectory_pos_list)
            # trajectory_ori_list = np.asarray(trajectory_ori_list)

            ######################### add the debug lines for visualization ####################
            line_id = []
            four_points = trajectory_pos_list[2:6]
            line_id.append(p.addUserDebugLine(lineFromXYZ=four_points[0], lineToXYZ=four_points[1]))
            line_id.append(p.addUserDebugLine(lineFromXYZ=four_points[1], lineToXYZ=four_points[2]))
            line_id.append(p.addUserDebugLine(lineFromXYZ=four_points[2], lineToXYZ=four_points[3]))
            line_id.append(p.addUserDebugLine(lineFromXYZ=four_points[3], lineToXYZ=four_points[0]))
            ######################### add the debug line for visualization ####################

            for j in range(len(trajectory_pos_list)):
                if len(trajectory_pos_list[j]) == 3:
                    last_pos = self.robot.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j])
                    last_ori = np.copy(trajectory_ori_list[j])
                elif len(trajectory_pos_list[j]) == 2:
                    self.robot.gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])

            ######################### remove the debug lines after moving ######################
            for i in line_id:
                p.removeUserDebugItem(i)
            ######################### remove the debug lines after moving ######################

            self.exclude_objects()
            crowded_index = np.intersect1d(np.where(self.pred_cls == 0)[0], self.rest_index)

            ################## Check the results to determine whether to clean again #####################
            # self.manipulator_before, self.lwh_list, self.pred_cls, self.pred_conf = self.task.get_obs()
            # crowded_index = np.where(self.pred_cls == 0)[0]
            # ############## exclude boxes which have been knolling ##############
            # manipulator_before = manipulator_before[len(self.success_manipulator_after):]
            # prediction = prediction[len(self.success_manipulator_after):]
            # new_lwh_list = new_lwh_list[len(self.success_manipulator_after):]
            # crowded_index = np.setdiff1d(crowded_index, np.arange(len(self.success_manipulator_after)))
            # model_output = model_output[len(self.success_manipulator_after):]
            # ############## exclude boxes which have been knolling ##############
            # self.yolo_pose_model.plot_grasp(manipulator_before, prediction, model_output)

            ################### Check the results to determine whether to clean again #####################

            # manipulator_before = np.concatenate((self.success_manipulator_after, manipulator_before), axis=0)
            # new_lwh_list = np.concatenate((self.success_lwh, new_lwh_list), axis=0)

        return

    def unstack(self):

        self.offset_low = np.array([0, 0, 0.003])
        self.offset_high = np.array([0, 0, 0.04])
        while True:
            crowded_index = np.where(self.pred_cls == 0)[0]
            manipulator_before_input = self.manipulator_before[crowded_index]
            lwh_list_input = self.lwh_list[crowded_index]
            rays = self.task.get_ray(manipulator_before_input, lwh_list_input)
            out_times = 0
            fail_times = 0
            for i in range(len(rays)):

                trajectory_pos_list = [self.para_dict['reset_pos'], # move to the destination
                                       [1, 0],  # gripper close!
                                       self.offset_high + rays[i, :3],  # move directly to the above of the target
                                       self.offset_low + rays[i, :3],  # decline slowly
                                       self.offset_low + rays[i, 6:9],
                                       self.offset_high + rays[i, 6:9],
                                       self.para_dict['reset_pos'],] # unstack

                trajectory_ori_list = [self.para_dict['reset_ori'],
                                       [1, 0],
                                       self.para_dict['reset_ori'] + rays[i, 3:6],
                                       self.para_dict['reset_ori'] + rays[i, 3:6],
                                       self.para_dict['reset_ori'] + rays[i, 9:12],
                                       self.para_dict['reset_ori'] + rays[i, 9:12],
                                       self.para_dict['reset_ori'],]

                if self.para_dict['real_operate'] == True:
                    last_pos = self.para_dict['reset_pos']
                    last_ori = self.para_dict['reset_ori']
                    left_pos = None
                    right_pos = None
                else:
                    last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
                    last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
                    left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
                    right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])

                for j in range(len(trajectory_pos_list)):
                    if len(trajectory_pos_list[j]) == 3:
                        last_pos = self.robot.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j], index=j)
                        last_ori = np.copy(trajectory_ori_list[j])
                    elif len(trajectory_pos_list[j]) == 2:
                        ####################### Detect whether the gripper is disturbed by other objects during closing the gripper ####################
                        self.robot.gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])
                        ####################### Detect whether the gripper is disturbed by other objects during closing the gripper ####################

            self.manipulator_before, self.lwh_list, self.pred_cls, self.pred_conf = self.task.get_obs()
            crowded_index = np.where(self.pred_cls == 0)[0]
            if len(crowded_index) == 0:
                print('unstack success!')
                break

            # if out_times == 0 and fail_times == 0:
            #     break
            # else:
            #     add_num = fail_times + out_times
            #     print('add additional rays', add_num)

        # rewrite this variable to ensure to load data one by one
        return self.img_per_epoch

    def knolling(self):

        crowded_index = np.where(self.pred_cls == 0)[0]
        success_index = np.intersect1d(self.rest_index, np.where(self.pred_cls == 1)[0])
        if self.success_num > 0:
            pos_before = np.concatenate((self.success_manipulator_before[:, :3], self.manipulator_before[self.success_num:, :3]), axis=0)
            ori_before = np.concatenate((self.success_manipulator_before[:, 3:6], self.manipulator_before[self.success_num:, 3:6]), axis=0)
            lwh_list = np.concatenate((self.success_lwh[:, :3], self.lwh_list[self.success_num:, :3]), axis=0)
        else:
            pos_before = self.manipulator_before[:, :3]
            ori_before = self.manipulator_before[:, 3:6]
            lwh_list = np.copy(self.lwh_list)
        manipulator_before, manipulator_after, lwh_list = self.get_knolling_data(pos_before=pos_before,
                                                                                 ori_before=ori_before,
                                                                                 lwh_list=lwh_list,
                                                                                 success_index=success_index,
                                                                                 offline_flag=False)

        # manipulator_before = manipulator_before[self.success_num:]
        # manipulator_after = manipulator_after[self.success_num:]
        # lwh_list = lwh_list[self.success_num:]
        # after the transformer model, manipulator before and after only contain boxes which can be grasped!
        start_end = np.concatenate((manipulator_before, manipulator_after), axis=1)


        self.success_manipulator_after = np.append(self.success_manipulator_after, manipulator_after).reshape(-1, 6)
        self.success_manipulator_before = np.append(self.success_manipulator_before, manipulator_before).reshape(-1, 6)
        self.success_lwh = np.append(self.success_lwh, lwh_list).reshape(-1, 3)
        self.success_num += len(manipulator_after)

        offset_low = np.array([0, 0, 0.00])
        offset_low_place = np.array([0, 0, 0.005])
        offset_high = np.array([0, 0, 0.04])
        grasp_width = np.min(lwh_list[:, :2], axis=1)
        for i in range(len(start_end)):
            trajectory_pos_list = [[0, grasp_width[i]],  # gripper open!
                                   offset_high + start_end[i][:3],  # move directly to the above of the target
                                   offset_low + start_end[i][:3],  # decline slowly
                                   [1, grasp_width[i]],  # gripper close
                                   offset_high + start_end[i][:3],  # lift the box up
                                   offset_high + start_end[i][6:9],  # to the target position
                                   offset_low_place + start_end[i][6:9],  # decline slowly
                                   [0, grasp_width[i]],  # gripper open!
                                   offset_high + start_end[i][6:9]]  # rise without box
            trajectory_ori_list = [self.para_dict['reset_ori'] + start_end[i][3:6],
                                   self.para_dict['reset_ori'] + start_end[i][3:6],
                                   self.para_dict['reset_ori'] + start_end[i][3:6],
                                   [1, grasp_width[i]],
                                   self.para_dict['reset_ori'] + start_end[i][3:6],
                                   self.para_dict['reset_ori'] + start_end[i][9:12],
                                   self.para_dict['reset_ori'] + start_end[i][9:12],
                                   [0, grasp_width[i]],
                                   self.para_dict['reset_ori'] + start_end[i][9:12]]
            if i == 0:
                last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
                last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
                left_pos = np.asarray(p.getLinkState(self.arm_id, 7)[0])
                right_pos = np.asarray(p.getLinkState(self.arm_id, 8)[0])
            else:
                pass

            for j in range(len(trajectory_pos_list)):
                if len(trajectory_pos_list[j]) == 3:
                    print('ready to move', trajectory_pos_list[j])
                    # print('ready to move cur ori', last_ori)
                    last_pos = self.robot.move(last_pos, last_ori, trajectory_pos_list[j], trajectory_ori_list[j], j, task='knolling')
                    last_ori = np.copy(trajectory_ori_list[j])
                    # print('this is last ori after moving', last_ori)

                elif len(trajectory_pos_list[j]) == 2:
                    self.robot.gripper(trajectory_pos_list[j][0], trajectory_pos_list[j][1])

        ############### Back to the reset pos and ori ###############
        last_pos = self.robot.move(last_pos, last_ori, self.para_dict['reset_pos'], self.para_dict['reset_ori'])
        last_ori = np.copy(self.para_dict['reset_ori'])
        ik_angles0 = p.calculateInverseKinematics(self.arm_id, 9, targetPosition=self.para_dict['reset_pos'],
                                                  maxNumIterations=200,
                                                  targetOrientation=p.getQuaternionFromEuler(self.para_dict['reset_ori']))
        for motor_index in range(5):
            p.setJointMotorControl2(self.arm_id, motor_index, p.POSITION_CONTROL,
                                    targetPosition=ik_angles0[motor_index], maxVelocity=7)
        for i in range(30):
            p.stepSimulation()
            # time.sleep(1 / 48)
        ############### Back to the reset pos and ori ###############

        # self.task.get_obs(look_flag=True, img_path=self.para_dict['data_source_path'] + 'real_images/after.png')

    def exclude_objects(self):

        self.manipulator_before, self.lwh_list, self.pred_cls, self.pred_conf = self.task.get_obs()

        if len(self.success_manipulator_after) == 0:
            self.rest_index = np.arange(len(self.manipulator_before))
        else:
            for i in range(len(self.manipulator_before)):
                distance = np.linalg.norm(self.manipulator_before[i, :2] - self.success_manipulator_after[:, :2], axis=1)
                match_flag = np.any(distance < 0.01)
                if match_flag == True:
                    self.finished_index = np.append(self.finished_index, i)
                    # self.finished_index.append(i)
            self.finished_index = np.asarray(self.finished_index)
            self.rest_index = np.setdiff1d(np.arange(len(self.manipulator_before)), self.finished_index)

        self.manipulator_before_total = np.copy(self.manipulator_before)
        self.lwh_list_total = np.copy(self.lwh_list)
        self.pred_cls_total = np.copy(self.pred_cls)

        # self.manipulator_before = self.manipulator_before[rest_index]
        # self.lwh_list = self.lwh_list[rest_index]
        # self.pred_cls = self.pred_cls[rest_index]

    def reset(self, epoch=None, manipulator_after=None, lwh_after=None, recover_flag=False):

        p.resetSimulation()
        self.task.create_scene()
        self.arm_id = self.robot.create_arm()
        if recover_flag == False:
            manipulator_before, lwh_list, pred_cls, pred_conf = self.task.create_objects(manipulator_after, lwh_after)
        else:
            info_path = self.para_dict['data_source_path'] + 'sim_info/%012d.txt' % epoch
            self.task.recover_objects(info_path)
            self.task.delete_objects()
        self.img_per_epoch = 0

        self.state_id = p.saveState()
        # return img_per_epoch_result

        if recover_flag == False:
            return manipulator_before, lwh_list, pred_cls, pred_conf
        else:
            pass

    def step(self):

        self.manipulator_before, self.lwh_list, self.pred_cls, self.pred_conf = self.reset() # reset the table
        # self.boxes_id_recover = np.copy(self.boxes_index)
        # self.manual_knolling() # generate the knolling after data based on manual or the model
        self.robot.calculate_gripper()
        self.conn, self.real_table_height, self.sim_table_height = self.robot.arm_setup()

        #######################################################################################
        # 1: clean_grasp + knolling, 3: knolling, 4: check_accuracy of knolling, 5: get_camera
        crowded_index = np.where(self.pred_cls == 0)[0]
        while self.success_num < self.task.num_boxes:
            self.clean_grasp()
            # self.unstack()
            self.knolling()
            self.exclude_objects()
            print('here')
        #######################################################################################

        if self.para_dict['real_operate'] == True:
            end = np.array([0], dtype=np.float32)
            self.conn.sendall(end.tobytes())

if __name__ == '__main__':

    # np.random.seed(21)
    # random.seed(21)

    np.set_printoptions(precision=5)
    para_dict = {'start_num': 0, 'end_num': 10, 'thread': 9, 'evaluations': 1,
                 'yolo_conf': 0.6, 'yolo_iou': 0.6, 'device': 'cuda:0',
                 'reset_pos': np.array([0, 0, 0.12]), 'reset_ori': np.array([0, np.pi / 2, 0]),
                 'save_img_flag': True,
                 'init_pos_range': [[0.13, 0.17], [-0.03, 0.03], [0.01, 0.02]], 'init_offset_range': [[-0.05, 0.05], [-0.1, 0.1]],
                 'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
                 'boxes_num': np.random.randint(5, 6),
                 'is_render': True,
                 'box_range': [[0.016, 0.048], [0.016], [0.01, 0.02]],
                 'box_mass': 0.1,
                 'gripper_threshold': 0.002, 'gripper_sim_step': 10, 'gripper_force': 3,
                 'move_threshold': 0.001, 'move_force': 3,
                 'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
                 'box_lateral_friction': 1, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
                 'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
                 'data_source_path': './knolling_box/',
                 'urdf_path': './ASSET/urdf/',
                 'yolo_model_path': './ASSET/models/627_pile_pose/weights/best.pt',
                 'real_operate': False, 'data_collection': False,
                 'use_knolling_model': True, 'use_lstm_grasp_model': False, 'use_yolo_model': True}

    if para_dict['real_operate'] == True:
        para_dict['yolo_model_path'] = './ASSET/models/1007_pile_sundry/weights/best.pt'
    if para_dict['use_yolo_model'] == True:
        para_dict['yolo_model_path'] = './ASSET/models/924_grasp/weights/best.pt'


    knolling_para = {'total_offset': [0.035, -0.17 + 0.016, 0], 'gap_item': 0.015,
                     'gap_block': 0.015, 'random_offset': False,
                     'area_num': 2, 'ratio_num': 1,
                     'kind_num': 5,
                     'order_flag': 'confidence',
                     'item_odd_prevent': True,
                     'block_odd_prevent': True,
                     'upper_left_max': True,
                     'forced_rotate_box': False}

    lstm_dict = {'input_size': 6,
                 'hidden_size': 32,
                 'num_layers': 8,
                 'output_size': 2,
                 'hidden_node_1': 32, 'hidden_node_2': 8,
                 'batch_size': 1,
                 'device': 'cuda:0',
                 'set_dropout': 0.1,
                 'threshold': 0.45,
                 'grasp_model_path': './ASSET/models/LSTM_918_0/best_model.pt',}

    arrange_dict = {'running_name': 'devoted-terrain-29',
                    'transformer_model_path': './ASSET/models/devoted-terrain-29',
                    'use_yaml': True,
                    'arrange_x_offset': 0.03,
                    'arrange_y_offset': 0.0}

    main_env = knolling_main(para_dict=para_dict, knolling_para=knolling_para, lstm_dict=lstm_dict, arrange_dict=arrange_dict)

    evaluation = 1
    for evaluation in range(para_dict['evaluations']):
        # env.get_parameters(evaluations=evaluation,
        #                    knolling_generate_parameters=knolling_generate_parameters,
        #                    dynamic_parameters=dynamic_parameters,
        #                    general_parameters=general_parameters,
        #                    knolling_env=knolling_env)
        main_env.step()