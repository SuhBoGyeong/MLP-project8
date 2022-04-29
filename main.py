import sys

# from baselines.common.cmd_util import common_arg_parser, parse_unknown_args

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.gail import ExpertDataset

from stable_baselines import SAC, PPO2, DQN, A2C

from envs.floors_env import FloorEnv
from utils.callbacks import getBestRewardCallback, logDir
from utils.arg_parser import common_arg_parser, parseLayersFromArgs

import os
import tensorflow as tf

def make_env(args, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # env = gym.make(env_id)
        env = FloorEnv(args=args)
        env.seed(seed + rank)

        if rank == 0:
            env = Monitor(env, logDir()+args.prefix+"/log", allow_early_resets=True)
            
        return env
    set_global_seeds(seed)
    return _init

def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    env = FloorEnv(args=args)

    # env = SubprocVecEnv([make_env(args, i) for i in range(4)])
    
    env = Monitor(env, logDir()+args.prefix+"/log", allow_early_resets=True)

    layers = parseLayersFromArgs(args=args) # default [32, 32]

    bestRewardCallback = getBestRewardCallback(args)

    # policy = MlpLstmPolicy
    # policy_kwargs = dict(layers=layers)

    # env = DummyVecEnv([lambda: env])

    # if args.model_path != None:
    #     model = PPO2.load(args.model_path, env=env, verbose=1, nminibatches=1, n_steps=1024, tensorboard_log=os.path.join("tensorboard_"+args.env,args.prefix), policy_kwargs=policy_kwargs)
    # else:
    #     model = PPO2(policy, env, verbose=1, nminibatches=1, n_steps=1024, tensorboard_log=os.path.join("tensorboard_"+args.env,args.prefix), policy_kwargs=policy_kwargs)

    
    # Using only one expert trajectory
    # you can specify `traj_limitation=-1` for using the whole dataset
    # Pretrain the PPO2 model
    if env.dim == 2:
        model = DQN(MlpPolicy, env, verbose=1)
    elif env.dim == 1:
        model = DQN(MlpPolicy, env, verbose=1)

    dataset = ExpertDataset(expert_path='expert_fcfs_w4_2dim.npz',
                            traj_limitation=1, batch_size=1024)
    model.pretrain(dataset, n_epochs=10000)
   
    model.learn(total_timesteps=args.num_timesteps, log_interval=100, callback=bestRewardCallback)
    # model.learn(total_timesteps=2000*1000000, log_interval=100)

if __name__ == '__main__':
    main(sys.argv)

