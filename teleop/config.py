from clearconf import BaseConfig

class Config(BaseConfig):
    class Env:
        name = 'rpFrankaRobotiqData-v0' # environment to load
        args = {'is_hardware': True, 'config_path': './assets/franka_robotiq.config'}
    class Noise:
        seed = 1234 # seed for generating environment instances
        reset_noise = 0.0 # Amplitude of noise during reset
        action_noise = 0.0 # Amplitude of action noise during rollout
    class Output:
        output = "teleOp_trace.h5" # Output name
        horizon = 100 # Rollout horizon
        num_rollouts = 2 # number of repeats for the rollouts
        output_format = 'RoboHive' # Data format ['RoboHive', 'RoboSet']

        render = 'onscreen' # Where to render? ['onscreen', 'offscreen', 'none']
        camera = [] # list of camera topics for rendering
    class Control:
        goal_site = 'ee_target' # Site that updates as goal using inputs
        teleop_site = 'end_effector' # Site used for teleOp/target for IK