import sys

# from baselines.common.cmd_util import common_arg_parser, parse_unknown_args

from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor

from stable_baselines import SAC, PPO2, DQN

from utils.callbacks import getBestRewardCallback, logDir, rmsLogging
from utils.arg_parser import common_arg_parser, parseLayersFromArgs

import numpy as np
import os
from glob import glob
import tensorflow as tf

import csv
import re

import json

from PyInquirer import prompt, print_json
from argparse import Namespace

# from envs.simple_env import Network
from envs.floors_env import FloorEnv

def main(args):
    buffers = []
    env = FloorEnv(args=args)
    # env = DummyVecEnv([lambda: env])

    layers = parseLayersFromArgs(args=args) # default [32, 32]
    policy_kwargs = dict(layers=layers)

    model = DQN.load(args.model_path)

    test_runs = 1

    for i in range(test_runs):
        obs = env.reset()

        print(i+1,"/",test_runs)
        while True:
            
            action, _states = model.predict(obs)
            # print(action)
            obs, rewards, dones, info = env.step(action)

            if dones == True:
                # env.render(show=True)
                # print(info)
                # buffers.append(info["buffers"])
                break

        # env.render(movie=True, movie_name=args.prefix)

    def simulate(env, autopilot_flag="min"):
        env.reset()
        env.title = autopilot_flag

        while True:
            action = env.current_agent().autopilot(flag=autopilot_flag, return_floor=True)
            # print("")
            # print("ITERATION:",i)
            obs, reward, done, info = env.step(action)
            # print(reward)
            if done == True:
                print("ELAPSED SIM-TIME: ", env.sim_time, " | RL")
                break

        return env.buffers

    env.render(save=True, movie_name=args.prefix)
    
    # for autopilot_flag in ["min","fcfs"]:
    #     buf = simulate(env, autopilot_flag)
    #     buffers.append(buf)

    # # buf = rl_test(env)
    # buffers.append(buf)

    # buffers = dict(pair for d in buffers for pair in d.items())

    # env.render(buffers=buffers,movie_name=args.prefix, save=True, show=False)

                
     
if __name__ == '__main__':
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)

    log_files = sorted(glob(logDir()+"/*"))

    questions = [
        {
            'type': 'list',
            'name': 'target_model',
            'message': 'Which run run run?',
            'choices':log_files
        }
    ]

    answers = prompt(questions)

    f = open(answers['target_model']+'/log.monitor.csv', 'r')
    _args = json.loads(f.readline().replace('#',''))['args']
    _args['play'] = True

    model_files = sorted(glob(answers['target_model'].replace('.monitor.csv','')+'/*_model.pkl'))
    model_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    
    _args['model_path'] = model_files[-1]

    if args.prefix != "":
        _args['prefix'] = args.prefix

    args = Namespace(**_args)
    print("Load saved args", args)
    f.close()

    main(args=args)

