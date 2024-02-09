from utils import *
from ENV.task.KnollingEnv import knolling_env
from ENV.robot.KnollingRobot import knolling_robot
import cv2

class knolling_main():

    def __init__(self, para_dict=None, knolling_para=None, lstm_dict=None, arrange_dict=None, rl_dict=None):
        # super(knolling_main, self).__init__(para_dict=para_dict, knolling_para=knolling_para, lstm_dict=lstm_dict, arrange_dict=arrange_dict)

        self.para_dict = para_dict
        self.rl_dict = rl_dict
        self.knolling_para = knolling_para

        self.task = knolling_env(para_dict=para_dict, lstm_dict=lstm_dict)
        self.robot = knolling_robot(para_dict=para_dict, knolling_para=knolling_para)

        if self.para_dict['use_knolling_model'] == True:
            self.arrange_dict = arrange_dict
            from ASSET.arrange_model_deploy import Arrange_model
            self.arrange_model = Arrange_model(para_dict=para_dict, arrange_dict=arrange_dict, max_num=self.para_dict['boxes_num'])
        if self.para_dict['visual_perception_model'] == 'lstm_grasp':
            # self.lstm_dict = lstm_dict
            from ASSET.visual_perception import Yolo_pose_model
            self.task.visual_perception_model = Yolo_pose_model(para_dict=para_dict, lstm_dict=lstm_dict, use_lstm=True)
            from ASSET.yolo_seg_deploy import Yolo_seg_model
            self.task.visual_perception_model = Yolo_seg_model(para_dict=para_dict)
        if self.para_dict['visual_perception_model'] == 'yolo_grasp':
            from ASSET.yolo_grasp_deploy import Yolo_grasp_model
            self.task.visual_perception_model = Yolo_grasp_model(para_dict=para_dict)
        if self.para_dict['rl_enable_flag'] == True:
            from RL_motion_model.rl_model_deploy import rl_unstack_model
            self.rl_model = rl_unstack_model(rl_dict=rl_dict)

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

        # test for all objects!!!!!!!!
        success_index = np.arange(len(pos_before))

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

            if self.finished_num > 0:
                create_pos = np.concatenate((self.success_manipulator_after[:, :3], pos_before[self.finished_num:, :]), axis=0)
                create_ori = np.concatenate((self.success_manipulator_after[:, 3:6], ori_before[self.finished_num:, :]), axis=0)
                create_lwh = np.concatenate((self.finished_lwh[:, :3], lwh_list[self.finished_num:, :]), axis=0)
                create_ori[:len(self.success_manipulator_after), 2] = 0
            else:
                create_pos = np.copy(pos_before)
                create_ori = np.copy(ori_before)
                create_lwh = np.copy(lwh_list)

            ################### recreate the new object ####################
            self.task.create_objects(pos_data=create_pos, ori_data=create_ori, lwh_data=create_lwh)
            ################### recreate the new object ####################

            candidate_num = 10  # provide several candidates to select the best output
            candidate_img = []
            candidate_after = []
            candidate_before = []
            candidate_lwh = []
            self.before_arrange_state = p.saveState()

            lwh_list = np.around(lwh_list, decimals=3)
            for i in range(candidate_num):
                # input_index = np.setdiff1d(np.arange(len(pos_before)), success_index)
                input_index = np.copy(success_index)
                pos_before_input = pos_before.astype(np.float32)
                ori_before_input = ori_before.astype(np.float32)
                lwh_list_input = lwh_list.astype(np.float32)
                ori_after = np.zeros((len(ori_before_input), 3))

                # #################### exchange the length and width randomly enrich the input ##################
                # for j in input_index:
                #     if np.random.random() < 0.5:
                #         temp = lwh_list_input[j, 1]
                #         lwh_list_input[j, 1] = lwh_list_input[j, 0]
                #         lwh_list_input[j, 0] = temp
                #         ori_after[j, 2] += np.pi / 2
                # #################### exchange the length and width randomly enrich the input ##################

                # input include all objects(finished, success, fail),
                pos_after = self.arrange_model.pred(pos_before_input, ori_before_input, lwh_list_input, input_index)[:len(lwh_list_input), :]

                manipulator_before = np.concatenate((pos_before_input[input_index], ori_before_input[input_index]), axis=1)
                manipulator_after = np.concatenate((pos_after[input_index].astype(np.float32), ori_after), axis=1)

                # manipulator_before = np.concatenate((pos_before_input, ori_before_input), axis=1)
                # manipulator_after = np.concatenate((pos_after.astype(np.float32), ori_after), axis=1)
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

            candidate_index = self.get_candidate_index(candidate_img, candidate_num)
            manipulator_before = candidate_before[candidate_index]
            manipulator_after = candidate_after[candidate_index]
            lwh_list_classify = candidate_lwh[candidate_index]

        p.restoreState(self.before_arrange_state)
        print('here')

        return manipulator_before, manipulator_after, lwh_list_classify

    def rl_combine_obs(self, last_action):

        obj_info = np.concatenate((self.manipulator_before[:, [0, 1, 5]], self.lwh_list[:, :2]), axis=1)
        obj_info = obj_info[obj_info[:, 0].argsort()]
        padded_obj_info = np.concatenate((obj_info, np.zeros((self.rl_dict['obj_num'] - len(obj_info), 5))), axis=0).reshape(-1, )
        obs = np.concatenate((padded_obj_info, last_action))
        return obs

    def rl_unstack_table(self):

        crowded_index = np.intersect1d(np.where(self.pred_cls == 0)[0], self.rest_index)


        self.robot.gripper(1, 0)
        offset_high = np.array([0, 0, 0.04])
        if self.para_dict['real_operate'] == True:
            test_offset = np.array([0, 0, 0.007]) + self.robot.real_table_height # this is temporary offset for RL model
        else:
            test_offset = np.array([0, 0, 0]) + self.robot.sim_table_height

        while len(crowded_index) >= 1:
            # action = np.concatenate((self.para_dict['reset_pos'], [self.para_dict['reset_ori'][2]]))
            action = np.array([0.042, 0, 0.005, 0])
            last_pos = np.asarray(p.getLinkState(self.arm_id, 9)[0])
            last_ori = np.asarray(p.getEulerFromQuaternion(p.getLinkState(self.arm_id, 9)[1]))
            for i in range(self.rl_dict['max_step']):
                obs = self.rl_combine_obs(last_action=action)
                trajectory = self.rl_model.model_pred(obs=obs)
                print('this is trajectory', trajectory)

                trajectory[:3] += test_offset

                last_pos = self.robot.move(last_pos, last_ori, trajectory[:3], trajectory[3:])
                last_ori = np.copy(trajectory[3:])
                action = np.concatenate((last_pos, [last_ori[2]]))

            # lift the arm up and back to home
            trajectory_high = np.copy(trajectory)
            trajectory_high[:3] += offset_high
            last_pos = self.robot.move(last_pos, last_ori, trajectory_high[:3], trajectory_high[3:])
            self.robot.move(last_pos, last_ori, self.para_dict['reset_pos'], self.para_dict['reset_ori'])

            self.exclude_objects()
            crowded_index = np.where(self.pred_cls == 0)[0]

    def manual_unstack_table(self):
        if self.para_dict['real_operate'] == False:
            gripper_width = 0.020
            gripper_height = 0.034
        else:
            gripper_width = 0.024
            gripper_height = 0.04

        offset_low = np.array([0, 0, 0.005])
        offset_high = np.array([0, 0, 0.04])

        crowded_index = np.intersect1d(np.where(self.pred_cls == 0)[0], self.rest_index)

        restrict_gripper_diagonal = np.sqrt(gripper_width ** 2 + gripper_height ** 2)
        gripper_box_gap = 0.006

        while len(crowded_index) >= 1:
            # len(crowded_index) >= len(self.rest_index):

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
                    trajectory_ori_list.append([0, np.pi / 2, 0])

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

            ######### this is temp setting, to make all objects graspable ########
            # crowded_index = np.intersect1d(np.where(self.pred_cls == 0)[0], self.rest_index)
            crowded_index = np.where(self.pred_cls == 0)[0]
            ######### this is temp setting, to make all objects graspable ########

        return

    def knolling(self):

        # crowded_index = np.where(self.pred_cls == 0)[0]
        success_index = np.intersect1d(self.rest_index, np.where(self.pred_cls == 1)[0])
        if self.finished_num > 0:
            pos_before = np.concatenate((self.success_manipulator_before[:, :3], self.manipulator_before[self.finished_num:, :3]), axis=0)
            ori_before = np.concatenate((self.success_manipulator_before[:, 3:6], self.manipulator_before[self.finished_num:, 3:6]), axis=0)
            lwh_list = np.concatenate((self.finished_lwh[:, :3], self.lwh_list[self.finished_num:, :3]), axis=0)
        else:
            pos_before = self.manipulator_before[:, :3]
            ori_before = self.manipulator_before[:, 3:6]
            lwh_list = np.copy(self.lwh_list)

        # After the knolling model, manipulator before and after only contain boxes which can be grasped in current scenario!
        # Also, the lwh_list will change the length and width randomly!
        manipulator_before, manipulator_after, lwh_list = self.get_knolling_data(pos_before=pos_before,
                                                                                 ori_before=ori_before,
                                                                                 lwh_list=lwh_list,
                                                                                 success_index=success_index,
                                                                                 offline_flag=False)

        # manipulator_before = manipulator_before[self.success_num:]
        # manipulator_after = manipulator_after[self.success_num:]
        # lwh_list = lwh_list[self.success_num:]
        start_end = np.concatenate((manipulator_before, manipulator_after), axis=1)


        self.success_manipulator_after = np.append(self.success_manipulator_after, manipulator_after).reshape(-1, 6)
        self.success_manipulator_before = np.append(self.success_manipulator_before, manipulator_before).reshape(-1, 6)
        self.finished_lwh = np.append(self.finished_lwh, lwh_list).reshape(-1, 3)
        self.finished_num += len(manipulator_after)

        if self.para_dict['real_operate'] == True:
            offset_low = np.array([0, 0, 0.005]) + self.robot.real_table_height
            offset_low_place = np.array([0, 0, 0.010]) + self.robot.real_table_height
            offset_high = np.array([0, 0, 0.04]) + self.robot.real_table_height
        else:
            offset_low = np.array([0, 0, 0.005]) + self.robot.sim_table_height
            offset_low_place = np.array([0, 0, 0.010]) + self.robot.sim_table_height
            offset_high = np.array([0, 0, 0.04]) + self.robot.sim_table_height
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

        # create the gripper mapping from sim to real
        self.robot.calculate_gripper()

        # setup and connect the real world robot arm
        self.conn, self.real_table_height, self.sim_table_height = self.robot.arm_setup()

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

        # initiate the list for the Hierarchical knolling
        self.success_manipulator_after = []
        self.success_manipulator_before = []
        self.finished_lwh = []
        self.finished_num = 0
        self.finished_index = []
        self.rest_index = np.arange(self.para_dict['boxes_num'])

        # reset the table
        self.manipulator_before, self.lwh_list, self.pred_cls, self.pred_conf = self.reset()

        # self.manual_knolling() # generate the knolling after data based on manual or the model

        #######################################################################################
        # 1: clean_grasp + knolling, 3: knolling, 4: check_accuracy of knolling, 5: get_camera
        # crowded_index = np.where(self.pred_cls == 0)[0]
        while self.finished_num < self.task.num_boxes:
            if self.para_dict['rl_enable_flag'] == True:
                self.rl_unstack_table()
            else:
                self.manual_unstack_table()
            # self.unstack()
            self.knolling()
            self.exclude_objects()
        #######################################################################################

        if self.para_dict['real_operate'] == True:
            end = np.array([0], dtype=np.float32)
            self.conn.sendall(end.tobytes())

if __name__ == '__main__':

    np.random.seed(1)
    random.seed(1)
    # 记一下25！！！

    # default: conf 0.6, iou 0.6
    np.set_printoptions(precision=5)
    para_dict = {'yolo_conf': 0.3, 'yolo_iou': 0.4, 'device': 'cuda:0',
                 'arm_reset_pos': np.array([0, 0, 0.12]), 'arm_reset_ori': np.array([0, np.pi / 2, 0]),
                 'save_img_flag': True,
                 'init_pos_range': [[0.03, 0.27], [-0.13, 0.13], [0.01, 0.02]], 'init_offset_range': [[-0.0, 0.0], [-0., 0.]],
                 'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
                 'boxes_num': np.random.randint(5, 6),
                 'is_render': True,
                 'box_range': [[0.016, 0.048], [0.016], [0.01, 0.02]],
                 'data_source_path': './IMAGE/',
                 'urdf_path': './ASSET/urdf/',
                 'real_operate': False,
                 'object': 'box', # box, polygon
                 'use_knolling_model': True, 'visual_perception_model': 'lstm_grasp', # lstm_grasp, yolo_grasp, yolo_seg
                 'lstm_enable_flag': True, 'rl_enable_flag': False,
                }

    # visual_perception_model： lstm_grasp, yolo_grasp, yolo_seg
    # determine to choose which model
    if para_dict['visual_perception_model'] == 'yolo_grasp':
        para_dict['yolo_model_path'] = './ASSET/models/924_grasp/weights/best.pt'
    elif para_dict['visual_perception_model'] == 'lstm_grasp':
        if para_dict['real_operate'] == True:
            para_dict['yolo_model_path'] = './ASSET/models/1007_pile_sundry/weights/best.pt'
        else:
            para_dict['yolo_model_path'] = './ASSET/models/627_pile_pose/weights/best.pt'
    else:
        if para_dict['visual_perception_model'] == 'yolo_seg':
            para_dict['yolo_model_path'] = './ASSET/models/820_pile_seg/weights/best.pt'


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
                 'threshold': 0.55,
                 'grasp_model_path': './ASSET/models/LSTM_918_0/best_model.pt',}
    if para_dict['real_operate'] == True:
        lstm_dict['threshold'] = 0.40

    arrange_dict = {'running_name': 'autumn-meadow-16',
                    'transformer_model_path': './ASSET/models/devoted-terrain-29',
                    'use_yaml': False,
                    'arrange_x_offset': 0.03,
                    'arrange_y_offset': 0.00}

    rl_dict = {'logger_id': '16', 'obj_num': para_dict['boxes_num'], 'rl_mode': 'SAC', 'max_step': 3}

    main_env = knolling_main(para_dict=para_dict, knolling_para=knolling_para, lstm_dict=lstm_dict,
                             arrange_dict=arrange_dict, rl_dict=rl_dict)

    evaluation = 1
    main_env.step()