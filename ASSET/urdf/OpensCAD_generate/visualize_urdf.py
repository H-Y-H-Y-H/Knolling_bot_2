import numpy as np
import pybullet as p
import pybullet_data as pd
import cv2

class visualize_env():

    def __init__(self, para_dict, knolling_para=None, lstm_dict=None, arrange_dict=None):

        self.para_dict = para_dict
        self.knolling_para = knolling_para

        self.kImageSize = {'width': 480, 'height': 480}
        self.init_pos_range = para_dict['init_pos_range']
        self.init_ori_range = para_dict['init_ori_range']
        self.init_offset_range = para_dict['init_offset_range']
        self.urdf_path = para_dict['urdf_path']
        self.pybullet_path = pd.getDataPath()
        self.is_render = para_dict['is_render']
        self.save_img_flag = para_dict['save_img_flag']
        self.objects_num = para_dict['objects_num']

        self.x_low_obs = 0.03
        self.x_high_obs = 0.27
        self.y_low_obs = -0.14
        self.y_high_obs = 0.14
        self.z_low_obs = 0.0
        self.z_high_obs = 0.05
        x_grasp_accuracy = 0.2
        y_grasp_accuracy = 0.2
        z_grasp_accuracy = 0.2
        self.x_grasp_interval = (self.x_high_obs - self.x_low_obs) * x_grasp_accuracy
        self.y_grasp_interval = (self.y_high_obs - self.y_low_obs) * y_grasp_accuracy
        self.z_grasp_interval = (self.z_high_obs - self.z_low_obs) * z_grasp_accuracy
        self.table_boundary = 0.03
        self.table_center = np.array([(self.x_low_obs + self.x_high_obs) / 2, (self.y_low_obs + self.y_high_obs) / 2])

        self.gripper_interval = 0.01

        if self.is_render:
            p.connect(p.GUI, options="--width=1280 --height=720")
        else:
            p.connect(p.DIRECT)

        self.camera_parameters = {
            'width': 640.,
            'height': 480,
            'fov': 42,
            'near': 0.1,
            'far': 100.,
            'camera_up_vector':
                [1, 0, 0],  # I really do not know the parameter's effect.
            'light_direction': [
                0.5, 0, 1
            ],  # the direction is from the light source position to the origin of the world frame.
        }
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.150, 0, 0], #0.175
                                                               distance=0.4,
                                                               yaw=90,
                                                               pitch = -90,
                                                               roll=0,
                                                               upAxisIndex=2)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=self.camera_parameters['fov'],
                                                              aspect=self.camera_parameters['width'] /
                                                                     self.camera_parameters['height'],
                                                              nearVal=self.camera_parameters['near'],
                                                              farVal=self.camera_parameters['far'])
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[0.1, 0, 0])
        p.setAdditionalSearchPath(pd.getDataPath())
        # p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setTimeStep(1. / 120.)

        self.grid_size = 5
        self.x_range = (-0.001, 0.001)
        self.y_range = (-0.001, 0.001)
        x_center = 0
        y_center = 0
        x_offset_values = np.linspace(self.x_range[0], self.x_range[1], self.grid_size)
        y_offset_values = np.linspace(self.y_range[0], self.y_range[1], self.grid_size)
        xx, yy = np.meshgrid(x_offset_values, y_offset_values)
        sigma = 0.01
        kernel = np.exp(-((xx - x_center) ** 2 + (yy - y_center) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
        self.kernel = kernel / np.sum(kernel)

        self.main_demo_epoch = 0

        self.create_entry_num = 0

    def create_scene(self):

        if np.random.uniform(0, 1) > 0.5:
            p.configureDebugVisualizer(lightPosition=[np.random.randint(1, 2), np.random.uniform(0, 1.5), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(0, 1) / 10)
        else:
            p.configureDebugVisualizer(lightPosition=[np.random.randint(1, 2), np.random.uniform(-1.5, 0), 2],
                                       shadowMapResolution=8192, shadowMapIntensity=np.random.randint(0, 1) / 10)
        self.baseid = p.loadURDF(self.urdf_path + "plane_zzz.urdf", useMaximalCoordinates=True)

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_high_obs + self.table_boundary, self.y_low_obs - self.table_boundary, self.z_low_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs + self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs],
            lineToXYZ=[self.x_low_obs - self.table_boundary, self.y_high_obs + self.table_boundary, self.z_low_obs])

        background = np.random.randint(1, 5)
        textureId = p.loadTexture(self.urdf_path + f"img_{background}.png")
        p.changeVisualShape(self.baseid, -1, textureUniqueId=textureId, specularColor=[0, 0, 0])

        p.setGravity(0, 0, -10)

    def create_objects(self, pos_data=None, ori_data=None, lwh_data=None):

        rdm_ori_roll  = np.random.uniform(self.init_ori_range[0][0], self.init_ori_range[0][1], size=(self.objects_num, 1))
        rdm_ori_pitch = np.random.uniform(self.init_ori_range[1][0], self.init_ori_range[1][1], size=(self.objects_num, 1))
        rdm_ori_yaw   = np.random.uniform(self.init_ori_range[2][0], self.init_ori_range[2][1], size=(self.objects_num, 1))
        rdm_ori = np.concatenate((rdm_ori_roll, rdm_ori_pitch, rdm_ori_yaw), axis=1)
        rdm_pos_x = np.random.uniform(self.init_pos_range[0][0], self.init_pos_range[0][1], size=(self.objects_num, 1))
        rdm_pos_y = np.random.uniform(self.init_pos_range[1][0], self.init_pos_range[1][1], size=(self.objects_num, 1))
        rdm_pos_z = np.random.uniform(self.init_pos_range[2][0], self.init_pos_range[2][1], size=(self.objects_num, 1))
        x_offset = np.random.uniform(self.init_offset_range[0][0], self.init_offset_range[0][1])
        y_offset = np.random.uniform(self.init_offset_range[1][0], self.init_offset_range[1][1])
        print('this is offset: %.04f, %.04f' % (x_offset, y_offset))
        rdm_pos = np.concatenate((rdm_pos_x + x_offset, rdm_pos_y + y_offset, rdm_pos_z), axis=1)

        self.objects_index = []
        urdf_filenames = [ # Add the filenames of your URDFs here
            "charger_1_L1.00_T1.00.urdf",
            "charger_1_L0.65_T0.65.urdf",
            "charger_1_L0.95_T0.95.urdf",
        
        ]   

        for i in range(self.objects_num):
            urdf_file = self.urdf_path + urdf_filenames[i % len(urdf_filenames)] # Cycle through the list of URDF files
            obj_id = p.loadURDF(urdf_file,
                                basePosition=rdm_pos[i],
                                baseOrientation=p.getQuaternionFromEuler(rdm_ori[i]), useFixedBase=0,
                                flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
            self.objects_index.append(obj_id)
            r = np.random.uniform(0, 0.9)
            g = np.random.uniform(0, 0.9)
            b = np.random.uniform(0, 0.9)
            p.changeVisualShape(obj_id, -1, rgbaColor=(r, g, b, 1))

        for _ in range(int(100)):
            p.stepSimulation()
            if self.is_render == True:
                pass
                # time.sleep(1/96)

        p.changeDynamics(self.baseid, -1, lateralFriction=self.para_dict['base_lateral_friction'],
                         contactDamping=self.para_dict['base_contact_damping'],
                         contactStiffness=self.para_dict['base_contact_stiffness'])


    def create_arm(self):
        self.arm_id = p.loadURDF(self.urdf_path + "robot_arm928/robot_arm1_backup.urdf",
                                 basePosition=[-0.08, 0, 0.02], useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

        p.changeDynamics(self.arm_id, 7, lateralFriction=self.para_dict['gripper_lateral_friction'],
                         contactDamping=self.para_dict['gripper_contact_damping'],
                         contactStiffness=self.para_dict['gripper_contact_stiffness'])
        p.changeDynamics(self.arm_id, 8, lateralFriction=self.para_dict['gripper_lateral_friction'],
                         contactDamping=self.para_dict['gripper_contact_damping'],
                         contactStiffness=self.para_dict['gripper_contact_stiffness'])

    def get_images(self):
        (width, length, image, image_depth, seg_mask) = p.getCameraImage(width=640,
                                                                         height=480,
                                                                         viewMatrix=self.view_matrix,
                                                                         projectionMatrix=self.projection_matrix,
                                                                         renderer=p.ER_BULLET_HARDWARE_OPENGL)
        far_range = self.camera_parameters['far']
        near_range = self.camera_parameters['near']
        depth_data = far_range * near_range / (far_range - (far_range - near_range) * image_depth)
        top_height = 0.4 - depth_data
        my_im = image[:, :, :3]
        temp = np.copy(my_im[:, :, 0])  # change rgb image to bgr for opencv to save
        my_im[:, :, 0] = my_im[:, :, 2]
        my_im[:, :, 2] = temp
        img = np.copy(my_im)

        cv2.namedWindow('zzz', 0)
        cv2.resizeWindow('zzz', 1280, 960)
        cv2.imshow('zzz', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def setup(self):

        p.resetSimulation()
        self.create_scene()
        self.create_arm()
        self.create_objects()
        self.get_images()

        while True:
            p.stepSimulation()

if __name__ == '__main__':

    np.random.seed(0)
    para_dict = {
        'reset_pos': np.array([0, 0, 0.12]), 'reset_ori': np.array([0, np.pi / 2, 0]),
        'save_img_flag': True,
        'init_pos_range': [[0.03, 0.27], [-0.15, 0.15], [0.01, 0.02]], 'init_offset_range': [[-0.00, 0.00], [-0., 0.]],
        'init_ori_range': [[-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4], [-np.pi / 4, np.pi / 4]],
        'objects_num': 3,
        'is_render': True,
        'gripper_lateral_friction': 1, 'gripper_contact_damping': 1, 'gripper_contact_stiffness': 50000,
        'box_lateral_friction': 1, 'box_contact_damping': 1, 'box_contact_stiffness': 50000,
        'base_lateral_friction': 1, 'base_contact_damping': 1, 'base_contact_stiffness': 50000,
        'urdf_path': './',
    }

    visualize_env = visualize_env(para_dict=para_dict)
    visualize_env.setup()