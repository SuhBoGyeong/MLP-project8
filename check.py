import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines import SAC, PPO2, DQN, A2C
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy

from envs.floors_env import FloorEnv
from utils.callbacks import getBestRewardCallback, logDir
from utils.arg_parser import common_arg_parser, parseLayersFromArgs

import random


def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    print(args)
    env = FloorEnv(args=args)
    model = DQN(MlpPolicy, env, verbose=0)
    all_actions = []

    obs = env.reset()
    done = False
    num_steps = 1
    count = 0

    while not done and count < num_steps:
        action = random.randrange(0, 6)
        # if count == 0:
        #     cursor_thread = 0
        # else:
        #     cursor_thread = random.randrange(0, 2)
        cursor_thread = 0
        obs, reward, done, buffers = env.step(action, cursor_thread)
        debug_obs = env.spit_debug_obs()[0] * 100 + 127.5
        mem = env.spit_memoery(True) * 100 + 127.5
        # steps = env.spit_steps()

        count += 1
        plt.imshow(debug_obs, cmap='gray')
        plt.title(f'#{count} Action {action}')
        plt.savefig(f'obs/ob{count}.png')

        for i in range(4):
            plt.imshow(mem[i], cmap='gray')
            plt.title(f'#{count}-m{i}')
            plt.savefig(f'obs/memory/ob{count}-m{i}.png')
        
        # for i in range(len(steps)):
        #     plt.imshow(steps[i], cmap='gray')
        #     plt.title(f'#{count}-step{steps[i]}')
        #     plt.savefig(f'obs/steps/ob{count}-s{i}.png')

## step에서 while문 돌아갈때마다 map정보 빼와보자
## action에 따라 배정은 잘 되지만, while loop 한번 돌 때 obs가 두번 돌아가는지 memory에 두개가 연속으로 들어간다.
## -> step 초반부에 reward 구하면서 한번 memory에 들어가고, while문 다 돌고 또 obs가 불려져서 memory에 들어가게 된다.
## 최우측에만 쌓이는 이유가 뭘까? 로직 확인하자
## test중인 애들이 보이기는 하나?

## 초반에 왜 안움직이는지 알아보고
## 두개씩 들어가는거 수정해야하고

if __name__ == '__main__':
    main(sys.argv)