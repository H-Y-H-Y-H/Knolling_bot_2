from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import TQC
import time
import numpy as np
import RL_dynamic_model.knolling_gym as knolling_gym
import gymnasium as gym
import os
import socket
from function import *
import pybullet_utils.bullet_client as bc
import copy

# env = gym.make('KnollingPickAndPlace-v0', render_mode='human')
#
# model = TQC.load('/home/zhizhuo/Creative_Machines_Lab/Knolling_bot_2/RL_dynamic_model/KnollingPickAndPlace-v0_25/best_model', env=env)

class RL_real_world_test():

    def __init__(self):
        self.env = gym.make('KnollingPickAndPlace-v0', render_mode='human')
        self.model = TQC.load('/home/zhizhuo/Creative_Machines_Lab/Knolling_bot_2/RL_dynamic_model/KnollingPickAndPlace-v0_25/best_model', env=self.env)
        self.physics_client = bc.BulletClient(connection_mode=p.DIRECT)
        pass

    def euler_to_quaternion(self, eular_angle):
        """
        Convert Euler angles to a quaternion.

        :param roll: Rotation around the X-axis (in radians)
        :param pitch: Rotation around the Y-axis (in radians)
        :param yaw: Rotation around the Z-axis (in radians)
        :return: Quaternion as a numpy array [w, x, y, z]
        """
        roll = eular_angle[0]
        pitch = eular_angle[1]
        yaw = eular_angle[2]
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])

    def quaternion_to_euler(self, quaternion):
        """
        Convert a quaternion to Euler angles.

        :param w: Quaternion scalar component
        :param x: Quaternion x component
        :param y: Quaternion y component
        :param z: Quaternion z component
        :return: Euler angles (roll, pitch, yaw) in radians
        """
        w = quaternion[0]
        x = quaternion[1]
        y = quaternion[2]
        z = quaternion[3]
        # Roll (x-axis rotation)
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))

        # Pitch (y-axis rotation)
        sin_pitch = 2 * (w * y - z * x)
        if abs(sin_pitch) >= 1:
            pitch = np.sign(sin_pitch) * np.pi / 2
        else:
            pitch = np.arcsin(sin_pitch)

        # Yaw (z-axis rotation)
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
        # roll = 0
        # pitch = np.pi / 2

        return np.array([roll, pitch, yaw])

    def real_world_move(self, last_obs, new_obs):

        d = np.array([0, 0.3])
        z_bias = np.array([0.002, 0.01])
        x_bias = np.array([-0.004, 0.000])  # yolo error is +2mm along x axis!
        y_bias = np.array([0.001, 0.005])
        z_parameters = np.polyfit(d, z_bias, 1)
        x_parameters = np.polyfit(d, x_bias, 1)
        y_parameters = np.polyfit(d, y_bias, 1)
        new_z_formula = np.poly1d(z_parameters)
        new_x_formula = np.poly1d(x_parameters)
        new_y_formula = np.poly1d(y_parameters)

        cmd_xyz = new_obs[:3]
        cmd_xyz[0] = np.clip(cmd_xyz[0], 0, 0.28)
        cmd_xyz[1] = np.clip(cmd_xyz[1], -0.14, 0.14)
        cmd_xyz[2] = np.clip(cmd_xyz[2], -0.01, 0.18)
        distance = cmd_xyz[0]
        cmd_xyz_input = np.copy(cmd_xyz)
        # distance_y = tar_pos[0]
        cmd_xyz_input[2] = cmd_xyz_input[2] + new_z_formula(distance) + 0.01
        # print('this is z add', new_z_formula(distance))

        last_ori = self.quaternion_to_euler(new_obs[3:7])
        cmd_ori = np.copy(last_ori)
        if cmd_ori[2] > np.pi:
            cmd_ori[2] -= np.pi
            last_ori[2] -= np.pi
        elif cmd_ori[2] < 0:
            cmd_ori[2] += np.pi
            last_ori[2] += np.pi
        cmd_width = np.clip(new_obs[7], 0, 0.064)
        cmd_width_input = 0.064 - cmd_width
        send_data = np.concatenate((cmd_xyz_input, cmd_ori, [cmd_width_input], last_obs[:3], last_ori, [last_obs[7]]),
                                   dtype=np.float32)

        if cmd_xyz[2] < -0.01 or cmd_xyz[2] > 0.18 or cmd_xyz[0] < 0 or cmd_xyz[0] > 0.25 or cmd_xyz[1] > 0.14 or \
                cmd_xyz[1] < -0.14:
            print('this is cmd_xyz', cmd_xyz)
            end = np.array([0], dtype=np.float32)
            self.conn.sendall(end.tobytes())
            print(f'evaluation over!!!!!')
            return None, None
            pass
        else:
            self.conn.sendall(send_data.tobytes())

        # cmd_xyz = last_obs[:3] + action[:3] * 0.05
        # cmd_xyz[0] = np.clip(cmd_xyz[0], 0, 0.28)
        # cmd_xyz[1] = np.clip(cmd_xyz[1], -0.14, 0.14)
        # cmd_xyz[2] = np.clip(cmd_xyz[2], -0.01, 0.18)
        # distance = cmd_xyz[0]
        # cmd_xyz_input = np.copy(cmd_xyz)
        # # distance_y = tar_pos[0]
        # cmd_xyz_input[2] = cmd_xyz_input[2] + new_z_formula(distance) + 0.01
        # # print('this is z add', new_z_formula(distance))
        #
        #
        # last_ori = self.quaternion_to_euler(last_obs[3:7])
        # cmd_ori = np.copy(last_ori)
        # cmd_ori[2] += action[3] * np.pi / 4
        # if cmd_ori[2] > np.pi:
        #     cmd_ori[2] -= np.pi
        #     last_ori[2] -= np.pi
        # elif cmd_ori[2] < 0:
        #     cmd_ori[2] += np.pi
        #     last_ori[2] += np.pi
        # cmd_width = np.clip(last_obs[7] + action[4] * 0.1, 0, 0.064)
        # cmd_width_input = 0.064 - cmd_width
        # send_data = np.concatenate((cmd_xyz_input, cmd_ori, [cmd_width_input], last_obs[:3], last_ori, [last_obs[7]]), dtype=np.float32)
        #
        # if cmd_xyz[2] < -0.01 or cmd_xyz[2] > 0.18 or cmd_xyz[0] < 0 or cmd_xyz[0] > 0.25 or cmd_xyz[1] > 0.14 or cmd_xyz[1] < -0.14:
        #     # print('this is cmd_xyz', cmd_xyz)
        #     # end = np.array([0], dtype=np.float32)
        #     # self.conn.sendall(end.tobytes())
        #     # print(f'evaluation over!!!!!')
        #     # return None, None
        #     pass
        # else:
        #     self.conn.sendall(send_data.tobytes())

            recall_data = self.conn.recv(8192)
            recall_data = np.frombuffer(recall_data, dtype=np.float32).reshape(-1, 7)
            real_xyz = recall_data[0, :3]
            real_xyz_output = np.copy(real_xyz)
            real_xyz_output[2] = real_xyz_output[2] - new_z_formula(distance) - 0.01
            real_ori = self.physics_client.getQuaternionFromEuler(recall_data[0, 3:6])
            # real_ori = self.euler_to_quaternion(recall_data[0, 3:6])
            real_width = 0.064 - recall_data[0, -1]
            achieved_goal = np.array([0.1, 0.05, 0.03])
            tar_ori = np.array([0, 0, np.pi / 2])

            ################### try to use cmd_width to return ####################
            observation = np.concatenate((real_xyz_output, real_ori, [cmd_width], achieved_goal, tar_ori))
            desired_goal = np.array([0.1, -0.05, 0.03])
            obs = {'observation': observation, 'achieved_goal': achieved_goal, 'desired_goal': desired_goal}
            ################### try to use cmd_width to return ####################

            distance = np.linalg.norm(real_xyz_output - achieved_goal)
            print('this is distance', distance)

            return obs, None

    def arm_setup(self, real_world_flag=False):

        if real_world_flag == True:

            HOST = "192.168.0.186"  # Standard loopback interface address (localhost)
            PORT = 8881  # Port to listen on (non-privileged ports are > 1023)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((HOST, PORT))
            # It should be an integer from 1 to 65535, as 0 is reserved. Some systems may require superuser privileges if the port number is less than 8192.
            # associate the socket with a specific network interface
            s.listen()
            print(f"Waiting for connection...\n")
            self.conn, addr = s.accept()
            print(self.conn)
            print(f"Connected by {addr}")
            table_surface_height = 0.003
            sim_table_surface_height = 0
            num_motor = 5
            # ! reset the pos in both real and sim
            reset_pos = np.array([0.02, 0, 0.1])
            reset_ori = np.array([0, np.pi / 2, 0])
            cmd_motor = np.asarray(inverse_kinematic(np.copy(reset_pos), np.copy(reset_ori)), dtype=np.float32)
            print('this is the reset motor pos', cmd_motor)
            self.conn.sendall(cmd_motor.tobytes())

            recall_data = self.conn.recv(8192)
            # print('received')
            recall_data = np.frombuffer(recall_data, dtype=np.float32).reshape(-1, 7)
            real_xyz = recall_data[0, :3]
            real_ori = self.physics_client.getQuaternionFromEuler(reset_ori)
            # real_ori = np.array([0, np.pi / 4, 0, np.pi / 4])
            real_width = 0.064 - recall_data[0, -1]

            # observation: xyz, wxyz, width, object_pos, object_ori
            achieved_goal = np.array([0.1, 0.05, 0.01])
            tar_ori = np.array([0, 0, np.pi / 2])
            observation = np.concatenate((reset_pos, real_ori, [real_width], achieved_goal, tar_ori))
            desired_goal = np.array([0.1, -0.05, 0.01])
            real_obs = {'observation': observation, 'achieved_goal': achieved_goal, 'desired_goal': desired_goal}

            sim_obs, info = self.env.reset()
        else:
            sim_obs, info = self.env.reset()

        for _ in range(50):
            # action: xyz displacement, ori displacement, width displacement
            if real_world_flag == True:
                action, _states = self.model.predict(sim_obs)
                last_sim_obs = copy.deepcopy(sim_obs)
                sim_obs, reward, done, truncated, info = self.env.step(action)
                real_obs, _ = self.real_world_move(np.copy(last_sim_obs['observation']), np.copy(sim_obs['observation']))
                # sim_obs['observation'][8:11] = np.array([0.1, 0.05, 0.01])
                # sim_obs['observation'][11:] = np.array([0, 0, np.pi / 2])
                print('here')
            else:
                action, _states = self.model.predict(sim_obs)
                sim_obs, reward, done, truncated, info = self.env.step(action)
                print('here')
            # move robot #
            if done:
                for j in range(4):
                    action = np.array([0, 0, 1.0, 0.0, -1.0])
                    obs, reward, done, truncated, info = self.env.step(action)
                print('grasp success!!!!!!!!!!')
                end = np.array([0], dtype=np.float32)
                self.conn.sendall(end.tobytes())
                print(f'evaluation over!!!!!')
                #time.sleep(10)
                quit()
            time.sleep(0.1)

        if real_world_flag == True:
            end = np.array([0], dtype=np.float32)
            self.conn.sendall(end.tobytes())
            print(f'evaluation over!!!!!')

if __name__ == '__main__':

    real_world_env = RL_real_world_test()

    euler = np.array([0, np.pi / 2, 0])
    quaternion = real_world_env.euler_to_quaternion(euler)
    test_quaternion = np.array([0, np.pi / 4, 0, np.pi / 4])
    test_quaternion_2 = np.array([0, 0.70716, 0, 0.70716])
    # new_euler = real_world_env.quaternion_to_euler(quaternion)
    test_euler_2 = real_world_env.quaternion_to_euler(test_quaternion_2)
    real_world_env.arm_setup(real_world_flag=True)