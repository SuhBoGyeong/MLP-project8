import sys
import gym

from stable_baselines.gail import generate_expert_traj
from envs.floors_env import FloorEnv
from utils.arg_parser import common_arg_parser

arg_parser = common_arg_parser()
args, unknown_args = arg_parser.parse_known_args(sys.argv)

env = FloorEnv(args=args)
# Here the expert is a random agent
# but it can be any python function, e.g. a PID controller
def dummy_expert(_obs):
    """
    Random agent. It samples actions randomly
    from the action space of the environment.

    :param _obs: (np.ndarray) Current observation
    :return: (np.ndarray) action taken by the expert
    """

    action = env.current_agent().autopilot(flag="fcfs", return_floor=True)
    return action
# Data will be saved in a numpy archive named `expert_cartpole.npz`
# when using something different than an RL expert,
# you must pass the environment object explicitly
generate_expert_traj(dummy_expert, 'expert_fcfs_w4_2dim', env, n_episodes=50)