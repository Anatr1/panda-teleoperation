from gym.envs.registration import register
import numpy as np
import os
import gym
curr_dir = os.path.dirname(os.path.abspath(__file__))


if 'rpFrankaRobotiqData-v1' not in gym.envs.registration.registry.env_specs:
    # Pose to fixed target
    encoder_type = "2d"
    # img_res="480x640"
    img_res="240x424"
    register(
        id='rpFrankaRobotiqData-v1',
        entry_point='teleop.my_env.franka_robotiq_data_v1:FrankaRobotiqData',
        max_episode_steps=50, #50steps*40Skip*2ms = 4s
        kwargs={
            'model_path': '/my_env/franka_robotiq.xml',
            'config_path': curr_dir+'/my_env/franka_robotiq.config',
            'nq_arm':7,
            'nq_ee':1,
            'name_ee':'end_effector',
            'visual_keys':[
                # customize the visual keys
                "rgb:left_cam:{}:{}".format(img_res, encoder_type),
                "rgb:right_cam:{}:{}".format(img_res, encoder_type),
                "rgb:top_cam:{}:{}".format(img_res, encoder_type),
                "rgb:Franka_wrist_cam:{}:{}".format(img_res, encoder_type),
                "d:left_cam:{}:{}".format(img_res, encoder_type),
                "d:right_cam:{}:{}".format(img_res, encoder_type),
                "d:top_cam:{}:{}".format(img_res, encoder_type),
                "d:Franka_wrist_cam:{}:{}".format(img_res, encoder_type),
                ]
            }
    )