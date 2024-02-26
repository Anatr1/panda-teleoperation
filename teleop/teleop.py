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
from robohive.utils.quat_math import mat2quat, diffQuat, mulQuat
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from oculus_reader import OculusReader

from utils.pose import vrfront2mj, vrbehind2mj
from config import Config as cfg

def setup_env():
    np.random.seed(cfg.Noise.seed)

    env = gym.make(cfg.Env.name, **cfg.Env.args_args.to_dict())
    env.seed(cfg.Noise.seed)

    env.env.mujoco_render_frames = True if 'onscreen' in cfg.Output.render else False
    goal_sid = env.sim.model.site_name2id(cfg.Control.goal_site)
    teleop_sid = env.sim.model.site_name2id(cfg.Control.teleop_site)
    env.sim.model.site_rgba[goal_sid][3] = 0.2

    return env, goal_sid, teleop_sid

def setup_oculus(env, goal_sid):
    oculus_reader = OculusReader() # when this line is executed the oculus visualization changes
    pos_offset = env.sim.model.site_pos[goal_sid].copy() # i guess this is the green rectangle position in simulation
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

    return oculus_reader, pos_offset, quat_offset

def reset(env, goal_sid, teleop_sid):
    # Reset
    reset_noise = cfg.Noise.reset_noise*np.random.uniform(low=-1, high=1, size=env.init_qpos.shape)
    env.reset(reset_qpos=env.init_qpos+reset_noise, blocking=True)
    # Reset goal site back to nominal position
    env.sim.model.site_pos[goal_sid] = env.sim.data.site_xpos[teleop_sid]
    env.sim.model.site_quat[goal_sid] = mat2quat(np.reshape(env.sim.data.site_xmat[teleop_sid], [3,-1]))

def main():

    env, goal_sid, teleop_sid = setup_env()
    oculus_reader, pos_offset, quat_offset = setup_oculus(env, goal_sid)

    act = np.zeros(env.action_space.shape)
    gripper_state = 0

    for _ in range(1):
        reset(env, goal_sid, teleop_sid)

        # recover init state
        _, _, _, env_info = env.forward()
        act = np.zeros(env.action_space.shape)
        gripper_state = 0

        while True:

            # poll input device --------------------------------------
            transformations, buttons = oculus_reader.get_transformations_and_buttons()

            # Check for reset request
            if buttons and buttons['B']:
                env.sim.model.site_pos[goal_sid] = pos_offset
                env.sim.model.site_quat[goal_sid] = quat_offset
                print("Rollout done. ")
                break

            if not (transformations and 'r' in transformations):
                continue

            right_controller_pose = transformations['r']
            VRpos, VRquat = vrbehind2mj(right_controller_pose)

            # Adjust origin if not engaged
            if not buttons['RG']:
                MJP0 = env.sim.model.site_pos[goal_sid].copy()
                MJR0 = env.sim.model.site_quat[goal_sid].copy()

                VRP0 = VRpos
                VRR0 = VRquat
                continue

            # Update targets if engaged
            dVRP = VRpos - VRP0 
            dVRR = diffQuat(VRR0, VRquat)

            env.sim.model.site_pos[goal_sid] = MJP0 + dVRP
            env.sim.model.site_quat[goal_sid] = mulQuat(MJR0, dVRR)
            gripper_state = buttons['rightTrig'][0]

            target_pos = env.sim.model.site_pos[goal_sid]  # udpate desired pos
            target_quat =  env.sim.model.site_quat[goal_sid] # update desired orientation

            # Find joint space solutions
            ik_result = qpos_from_site_pose(physics = env.sim, site_name = cfg.Control.goal_site,
                                            target_pos= target_pos, target_quat= target_quat,
                                            inplace=False, regularization_strength=1.0)

            if not ik_result.success:
                print(f"Status:{ik_result.success}, total steps:{ik_result.steps}, err_norm:{ik_result.err_norm}")
                continue

            act[:7] = ik_result.qpos[:7]
            act[7:] = gripper_state
            if cfg.Noise.action_noise:
                act = act + env.env.np_random.uniform(high=cfg.Noise.action_noise, low=-cfg.Noise.action_noise, size=len(act)).astype(act.dtype)
            if env.normalize_act:
                act = env.env.robot.normalize_actions(act)
            
            obs, rwd, done, env_info = env.step(act)

            # Detect jumps
            qpos_now = env_info['obs_dict']['qp_arm']
            qpos_arm_err = np.linalg.norm(ik_result.qpos[:7]-qpos_now[:7])
            if qpos_arm_err > 0.5:
                print("Jump detechted. Joint error {}. This is likely caused when hardware detects something unsafe. Resetting goal to where the arm curently is to avoid sudden jumps.".format(qpos_arm_err))
                # Reset goal back to nominal position
                env.sim.model.site_pos[goal_sid] = env.sim.data.site_xpos[teleop_sid]
                env.sim.model.site_quat[goal_sid] = mat2quat(np.reshape(env.sim.data.site_xmat[teleop_sid], [3,-1]))

        print("rollout end")
        time.sleep(0.5)
        # save and close
        env.close()

if __name__ == '__main__':
    main()
