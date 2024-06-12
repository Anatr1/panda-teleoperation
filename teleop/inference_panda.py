# """ =================================================
# Copyright (C) 2018 Vikash Kumar
# Author  :: Vikash Kumar (vikashplus@gmail.com)
# Source  :: https://github.com/vikashplus/robohive
# License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
# ================================================= """
DESC = """
TUTORIAL: Arm+Gripper tele-op using oculus \n
    - NOTE: Tutorial is written for franka arm and robotiq gripper. This demo is a tutorial, not a generic functionality for any any environment
EXAMPLE:\n
    - python tutorials/ee_teleop.py -e rpFrankaRobotiqData-v0\n
"""
# TODO: (1) Enforce pos/rot/grip limits (b) move gripper to delta commands

import time
import numpy as np
import click
import gym
from robohive.utils.quat_math import euler2quat, euler2mat, mat2quat, diffQuat, mulQuat
from robohive.utils.inverse_kinematics import IKResult, qpos_from_site_pose
from robohive.robot import robot
import my_env
import cv2
import imageio
from scipy.spatial.transform import Rotation as R
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from octo.model.octo_model import OctoModel
import jax

class PandaAction():
    def __init__(self, gripper, arm_xyz, arm_rpy, base_xy, base_yaw, terminate):
        self.gripper = gripper
        self.arm_xyz = arm_xyz
        self.arm_rpy = arm_rpy
        self.base_xy = base_xy
        self.base_yaw = base_yaw
        self.terminate = terminate

    def arm_action(self):
        return list(np.concatenate([self.arm_xyz, self.arm_rpy]))

    def arm_gripper_action(self):
        return list(np.concatenate([self.gripper, self.arm_xyz, self.arm_rpy]))
    
    def to_string(self):
        return " gripper: {}\n arm_xyz: {}\n arm_rpy: {}\n base_xy: {}\n base_yaw: {}\n terminate: {}\n\n".format(
            self.gripper, self.arm_xyz, self.arm_rpy, self.base_xy, self.base_yaw, self.terminate)

MAX_STEPS = 1000
INSTRUCTION = "Pick up the object"
WINDOW_SIZE = 1
PRETRAINED_MODEL_PATH = "./DH_128_real_small_80k"

try:
    from oculus_reader import OculusReader
except ImportError as e:
    raise ImportError("(Missing oculus_reader. HINT: Install and perform the setup instructions from https://github.com/rail-berkeley/oculus_reader)")

# VR ==> MJ mapping when teleOp user is standing infront of the robot
def vrfront2mj(pose):
    pos = np.zeros([3])
    pos[0] = -1.*pose[2][3]
    pos[1] = -1.*pose[0][3]
    pos[2] = +1.*pose[1][3]

    mat = np.zeros([3, 3])
    mat[0][:] = -1.*pose[2][:3]
    mat[1][:] = +1.*pose[0][:3]
    mat[2][:] = -1.*pose[1][:3]

    return pos, mat2quat(mat)

# VR ==> MJ mapping when teleOp user is behind the robot
def vrbehind2mj(pose):
    pos = np.zeros([3])
    pos[0] = +1.*pose[2][3]
    pos[1] = +1.*pose[0][3]
    pos[2] = +1.*pose[1][3]

    mat = np.zeros([3, 3])
    mat[0][:] = +1.*pose[2][:3]
    mat[1][:] = -1.*pose[0][:3]
    mat[2][:] = -1.*pose[1][:3]

    return pos, mat2quat(mat)

def quat2euler(quat):
    r = R.from_quat(quat)
    return r.as_euler('xyz', degrees=True)

def build_action(target_position, target_orientation, target_gripper, current_position, current_orientation, terminate=False):
    return [target_gripper, target_position[0]-current_position[0], target_position[1]-current_position[1], target_position[2]-current_position[2], target_orientation[0]-current_orientation[0], target_orientation[1]-current_orientation[1], target_orientation[2]-current_orientation[2], int(terminate)]

def resize_preserve_type(image):
    original_dtype = image.dtype
    image = tf.image.resize_with_pad(image, target_width=256, target_height=256)
    image = tf.cast(image, original_dtype)
    return image.numpy()

def extract_window_images(images, window_size):
    window_images = []
    for i in range(len(images)-window_size+1):
        window_images.append(images[i:i+window_size])
    return window_images[-1]

def parse_action(action):
  """Action comes in the form:
  ['grasp', 'x', 'y', 'z', 'yaw', 'pitch', 'roll', 'terminate']
  """
  gripper = [float(2*(action[0]-0.5)*255.0)]
  arm_xyz = action[1:4]
  arm_rpy = action[4:7]
  base_xy = [0.0, 0.0]
  base_yaw = 0.0
  terminate = terminate_bool_to_act(action[7])

  print(f"{gripper} {arm_xyz} {arm_rpy} {terminate}")

  return PandaAction(gripper, arm_xyz, arm_rpy, base_xy, base_yaw, terminate)

@click.command(help=DESC)
@click.option('-e', '--env_name', type=str, help='environment to load', default='rpFrankaRobotiqData-v0')
@click.option('-ea', '--env_args', type=str, default='', help=('env args. E.g. --env_args "{\'is_hardware\':True}"'))
@click.option('-rn', '--reset_noise', type=float, default=0.0, help=('Amplitude of noise during reset'))
@click.option('-an', '--action_noise', type=float, default=0.0, help=('Amplitude of action noise during rollout'))
@click.option('-o', '--output', type=str, default="teleOp_trace.h5", help=('Output name'))
@click.option('-h', '--horizon', type=int, help='Rollout horizon', default=100)
@click.option('-n', '--num_rollouts', type=int, help='number of repeats for the rollouts', default=2)
@click.option('-f', '--output_format', type=click.Choice(['RoboHive', 'RoboSet']), help='Data format', default='RoboHive')
@click.option('-c', '--camera', multiple=True, type=str, default=[], help=('list of camera topics for rendering'))
@click.option('-r', '--render', type=click.Choice(['onscreen', 'offscreen', 'none']), help='Where to render?', default='onscreen')
@click.option('-s', '--seed', type=int, help='seed for generating environment instances', default=123)
@click.option('-gs', '--goal_site', type=str, help='Site that updates as goal using inputs', default='ee_target')
@click.option('-ts', '--teleop_site', type=str, help='Site used for teleOp/target for IK', default='end_effector')
@click.option('-ps', '--pos_scale', type=float, default=0.05, help=('position scaling factor'))
@click.option('-rs', '--rot_scale', type=float, default=0.1, help=('rotation scaling factor'))
@click.option('-gs', '--gripper_scale', type=float, default=1, help=('gripper scaling factor'))
@click.option('-tx', '--x_range', type=tuple, default=(-0.5, 0.5), help=('x range'))
@click.option('-ty', '--y_range', type=tuple, default=(-0.5, 0.5), help=('y range'))
@click.option('-tz', '--z_range', type=tuple, default=(-0.5, 0.5), help=('z range'))
@click.option('-rx', '--roll_range', type=tuple, default=(-0.5, 0.5), help=('roll range'))
@click.option('-ry', '--pitch_range', type=tuple, default=(-0.5, 0.5), help=('pitch range'))
@click.option('-rz', '--yaw_range', type=tuple, default=(-0.5, 0.5), help=('yaw range'))
@click.option('-gr', '--gripper_range', type=tuple, default=(0, 1), help=('z range'))
def main(env_name, env_args, reset_noise, action_noise, output, horizon, num_rollouts, output_format, camera, seed, render, goal_site, teleop_site, pos_scale, rot_scale, gripper_scale, x_range, y_range, z_range, roll_range, pitch_range, yaw_range, gripper_range):

    # Get how many episodes have been recorded
    episode_id = len([name for name in os.listdir('../data/pick/npy/')])

    cap_wrist = cv2.VideoCapture(4)
    cap = cv2.VideoCapture(10)

    # seed and load environments
    env_args = {'is_hardware': True, 
                'config_path': '/home/teleop/project/teleop/my_env/franka_robotiq.config',
                #'config_path': './teleop/my_env/franka_robotiq.config', 
                # 'model_path': '/franka_robotiq.xml', 
                # 'target_pose': np.array([0, 0, 0, 0, 0, 0, 0, 0])
                }

    np.random.seed(seed)
    env = gym.make(env_name, **env_args)
    env.seed(seed)
    env.env.mujoco_render_frames = True if 'onscreen'in render else False
    goal_sid = env.sim.model.site_name2id(goal_site)
    teleop_sid = env.sim.model.site_name2id(teleop_site)
    env.sim.model.site_rgba[goal_sid][3] = 1 # make visible

    # prep input device
    oculus_reader = OculusReader()
    pos_offset = env.sim.model.site_pos[goal_sid].copy()
    quat_offset = env.sim.model.site_quat[goal_sid].copy()
    oculus_reader_ready = False
    while not oculus_reader_ready:
        # Get the controller and headset positions and the button being pushed
        transformations, buttons = oculus_reader.get_transformations_and_buttons()
        if transformations or buttons:
            oculus_reader_ready = True
        else:
            print("Oculus reader not ready. Check that headset is awake and controller are on")
        time.sleep(0.10)

    print('Oculus Ready!')

    # default actions
    act = np.zeros(env.action_space.shape)
    gripper_state = delta_gripper = 0

    # Reset
    reset_noise = reset_noise*np.random.uniform(low=-1, high=1, size=env.init_qpos.shape)
    env.reset(reset_qpos=env.init_qpos+reset_noise, blocking=True)
    # Reset goal site back to nominal position
    env.sim.model.site_pos[goal_sid] = env.sim.data.site_xpos[teleop_sid]
    env.sim.model.site_quat[goal_sid] = mat2quat(np.reshape(env.sim.data.site_xmat[teleop_sid], [3,-1]))

    # recover init state
    obs, rwd, done, env_info = env.forward()
    act = np.zeros(env.action_space.shape)
    gripper_state = 0

    reset = True
    current_pos = [0, 0, 0]
    current_rpy = [0, 0, 0]
    #count = 0

    images = []
    images_wrist = []
    episode = []

    # Load the model
    model = OctoModel.load_pretrained(PRETRAINED_MODEL_PATH)
    task = model.create_tasks(texts=[INSTRUCTION])

    # Filling observation window
    for i in range(WINDOW_SIZE*10):
        print(f"Filling observation window {i+1}/{WINDOW_SIZE*10}")
        status_wrist, photo_wrist = cap_wrist.read()
        if not status_wrist:
            print("Wrist camera not found")
        photo_wrist = cv2.cvtColor(photo_wrist, cv2.COLOR_BGR2RGB)
        images_wrist.append(photo_wrist)

        status, photo = cap.read()
        if not status:
            print("Camera not found")
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
        images.append(photo)

    # start rolling out
    #for i in range(MAX_STEPS):
    while True:
        print(f"Step {i+1}/{MAX_STEPS}")
        status_wrist, photo_wrist = cap_wrist.read()
        if not status_wrist:
            print("Wrist camera not found")
        photo_wrist = cv2.cvtColor(photo_wrist, cv2.COLOR_BGR2RGB)

        status, photo = cap.read()
        if not status:
            print("Camera not found")
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)

        resized_photo = resize_preserve_type(photo)
        resized_photo_wrist = resize_preserve_type(photo_wrist)
        images.append(resized_photo)
        images_wrist.append(resized_photo_wrist)
        input_images = np.stack(extract_window_images(images, WINDOW_SIZE))[None]
        input_images_wrist = np.stack(extract_window_images(images_wrist, WINDOW_SIZE))[None]
        observation = {
            'image_primary': input_images,
            #'image_wrist': input_images_wrist,
            'timestep_pad_mask': np.full((1, input_images.shape[1]), True, dtype=bool),
        }
        
        # Get the action from the model
        norm_action = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))
        norm_action = norm_action[0][0]

        # Denormalize the action
        action = (norm_action * model.dataset_statistics['action']['std'] + model.dataset_statistics['action']['mean'])
        parsed_action = parse_action(action)

        print(f"Action: {parsed_action.to_string()}")

        # Execute the action on the robot

        # poll input device --------------------------------------
        transformations, buttons = oculus_reader.get_transformations_and_buttons()

        # Check for reset request
        if buttons and buttons['B']:
            env.sim.model.site_pos[goal_sid] = pos_offset
            env.sim.model.site_quat[goal_sid] = quat_offset
            print("Rollout done. ")
            break
            

        # recover actions using input ----------------------------
        if transformations and 'r' in transformations:
            right_controller_pose = transformations['r']
            # VRpos, VRquat = vrfront2mj(right_controller_pose)
            VRpos, VRquat = vrbehind2mj(right_controller_pose)

            # Update targets if engaged
            if buttons['RG']:
                # dVRP/R = VRP/Rt - VRP/R0
                dVRP = VRpos - VRP0
                # dVRR = VRquat - VRR0
                dVRR = diffQuat(VRR0, VRquat)
                # MJP/Rt =  MJP/R0 + dVRP/R
                env.sim.model.site_pos[goal_sid] = MJP0 + dVRP
                env.sim.model.site_quat[goal_sid] = mulQuat(MJR0, dVRR)
                delta_gripper = buttons['rightTrig'][0]

            # Adjust origin if not engaged
            else:
                # RP/R0 = RP/Rt
                MJP0 = env.sim.model.site_pos[goal_sid].copy()
                MJR0 = env.sim.model.site_quat[goal_sid].copy()

                # VP/R0 = VP/Rt
                VRP0 = VRpos
                VRR0 = VRquat

            # udpate desired pos
            target_pos = env.sim.model.site_pos[goal_sid]
            # target_pos[:] += pos_scale*delta_pos
            # update desired orientation
            target_quat =  env.sim.model.site_quat[goal_sid]
            # target_quat[:] = mulQuat(euler2quat(rot_scale*delta_euler), target_quat)
            # update desired gripper
            gripper_state = gripper_scale*delta_gripper # TODO: Update to be delta

            if reset:
                current_pos = target_pos.copy()
                current_rpy = quat2euler(target_quat)
                reset = False
                #count=+1
            else:
                #if count % 100 == 0:
                print(f"Action: {build_action(target_pos, quat2euler(target_quat), gripper_state, current_pos, current_rpy)}")
                joint_state = np.append(env_info['obs_dict']['qp_arm'], gripper_state).astype(np.float32)
                print(f"Joint State: {joint_state}")
                print(f"Instruct: {INSTRUCTION}")

                current_pos = target_pos.copy()
                current_rpy = quat2euler(target_quat)

                # Get an observation image from the connected usb camera with ID=4 with OpenCV
                '''
                status_wrist, photo_wrist = cap_wrist.read()
                if not status_wrist:
                    print("Wrist camera not found")
                photo_wrist = cv2.cvtColor(photo_wrist, cv2.COLOR_BGR2RGB)

                images_wrist.append(photo_wrist)

                status, photo = cap.read()
                if not status:
                    print("Camera not found")
                photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)

                images.append(photo)
                episode.append({
                    'image': np.asarray(photo),
                    'image_wrist': np.asarray(photo_wrist),
                    'state': joint_state,
                    'action': build_action(target_pos, quat2euler(target_quat), gripper_state, current_pos, current_rpy),
                    'language_instruction': INSTRUCTION
                })
                '''
                #cv2.imshow("Webcam Video Stream", photo)
                
                #    count = 0
                #else:
                #    count+=1

            # Find joint space solutions
            ik_result = qpos_from_site_pose(
                        physics = env.sim,
                        site_name = teleop_site,
                        target_pos= target_pos,
                        target_quat= target_quat,
                        inplace=False,
                        regularization_strength=1.0)
            
            #print(ik_result)

            # Command robot
            if ik_result.success==False:
                print(f"Status:{ik_result.success}, total steps:{ik_result.steps}, err_norm:{ik_result.err_norm}")
            else:
                act[:7] = ik_result.qpos[:7]
                act[7:] = gripper_state
                if action_noise:
                    act = act + env.env.np_random.uniform(high=action_noise, low=-action_noise, size=len(act)).astype(act.dtype)
                if env.normalize_act:
                    #print(act[-1])
                    act = env.env.robot.normalize_actions(act)
                    #print(act[-1])
        # print(f't={env.time:2.2}, a={act}, o={obs[:3]}')

        # step env using action from t=>t+1 ----------------------
        
        obs, rwd, done, env_info = env.step(act)

        # Detect jumps
        qpos_now = env_info['obs_dict']['qp_arm']
        qpos_arm_err = np.linalg.norm(ik_result.qpos[:7]-qpos_now[:7])
        if qpos_arm_err>0.5:
            print("Jump detechted. Joint error {}. This is likely caused when hardware detects something unsafe. Resetting goal to where the arm curently is to avoid sudden jumps.".format(qpos_arm_err))
            # Reset goal back to nominal position
            env.sim.model.site_pos[goal_sid] = env.sim.data.site_xpos[teleop_sid]
            env.sim.model.site_quat[goal_sid] = mat2quat(np.reshape(env.sim.data.site_xmat[teleop_sid], [3,-1]))

    cap.release()
    cv2.destroyAllWindows()

    '''
    episode.append({
        'image': np.asarray(photo),
        'image_wrist': np.asarray(photo_wrist),
        'state': joint_state,
        'action': build_action(target_pos, quat2euler(target_quat), gripper_state, current_pos, current_rpy, terminate=True),
        'language_instruction': INSTRUCTION
    })

    # Convert the images to numpy arrays and add them to a list
    frames = [np.asarray(image) for image in images]
    frames_wrist = [np.asarray(image) for image in images_wrist]

    # Save the frames as an mp4 video
    imageio.mimsave(f'../data/pick/mp4/episode_{episode_id}.mp4', frames, fps=9)
    imageio.mimsave(f'../data/pick/mp4/episode_{episode_id}_wrist.mp4', frames_wrist, fps=9)

    # Save as npy
    np.save(f'../data/pick/npy/episode_{episode_id}.npy', episode)
    '''

    print("rollout end")
    time.sleep(0.5)
    # save and close
    env.close()

if __name__ == '__main__':
    main()
