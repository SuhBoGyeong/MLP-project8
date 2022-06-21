import random
import sys
import copy, os, json, time

import numpy as np
from gym import Env, spaces

from utils.arg_parser import common_arg_parser
from utils.callbacks import getBestRewardCallback, logDir

from envs.components.map import Map
from envs.components.pallet import Pallet


class FloorEnv(Env):
    '''
    Action : the floor that the agent wants to assign the pallet to
    States : 1-D np array, a series of the states of the map saved in a memory dictionary
    Reward : Float
    '''
    def __init__(self, args=None, dim=2, title="RL"):
        '''
        Initialize FloorEnv instance.
        It spits out states every step and save the status of the map for rendering
        '''
        self.pallet_counts = args.pallet_counts
        self.title = title
        self.dim = dim
        self.args = args

        obs = self.reset()
        self.resetBuffer()

        self.action_space = spaces.Discrete(5)

        ##################################################
        # To point an agent depending on the testers that a pallet is supposed to be assigned to
        self.cursor_thread = 0
        ##################################################

        if self.dim == 2:
            if self.args.window_size > 1:
                obs_shape = obs.shape
            else:
                obs_shape = obs.shape
        elif self.dim == 1:
            if self.args.window_size > 1:
                obs_shape = (self.args.window_size, len(obs),)
            else:
                obs_shape = (len(obs),)

        self.observation_space = spaces.Box(low=0, high=4, shape=obs_shape)

    def reset(self):
        '''
        Reset every field when starting another episode
        '''
        self.map = Map()
        self.pallet_idx = 0
        self.pallets = {}

        self.cursor = 0 # Pallet IDX
        self.done_count = 0
        self.done = False

        self.sim_time = 0
        
        ##################################################
        self.cursor_thread = 0
        ##################################################

        if self.dim == 2:
            result, _, _ = self.empty_obs("a")
            self.memoryA = [result] * self.args.window_size

            result, _, _ = self.empty_obs("b")
            self.memoryB = [result] * self.args.window_size

        elif self.dim == 1:
            result, _, _ = self.empty_obs("a")
            r = self.flatten_obs(result)
            self.memoryA = [copy.deepcopy(r)] * self.args.window_size

            result, _, _ = self.empty_obs("b")
            r = self.flatten_obs(result)
            self.memoryB = [copy.deepcopy(r)] * self.args.window_size

        self.resetBuffer()
        
        for pallet_idx in range(self.pallet_counts):
            if pallet_idx == 0:
                enter = True
            else:
                enter = False
            a = self.createPallet(enter=enter)
            ## self.map.agents에 agent들 넣어줘야했다! 0520
            # self.map.agents[pallet_idx] = a
            ## 0523 지금 보니 saveBuffer에 해당 로직이 있엇는데 오타가 있었다.
            ## self.map.pallets = self.pallets가 아니라 self.map.agents여야함

        self.saveBuffer(self.title)
        self.resetPalletBuffer()

        obs = self.obs(tester_type=self.pallets[0].tester_type())

        return obs

    def get_action_meanings(self):
        return ["F1", "F2", "F3", "F4", "F5"] + ['NOOP']

    def createPallet(self, enter):
        a = Pallet(self.map, self.pallet_idx, enter=enter, env=self)
        self.pallets[self.pallet_idx] = a
        self.pallet_idx += 1

        return a

    def resetBuffer(self):
        self.buffers = {}

    def saveBuffer(self, buffer_type="a"):
        if not buffer_type in self.buffers:
            self.buffers[buffer_type] = []

        self.map.agents = self.pallets
        self.buffers[buffer_type].append(copy.deepcopy(self.pallets))

        self.sim_time += 1
    
    def render(self, buffers=None, save=False, show=True, still=False, movie_name="movie_name"):
        self.map.agents = self.pallets
        if still == True:
            self.map.render(buffers=None)
        else:
            if buffers is None:
                buffers = self.buffers
            self.map.render(buffers=buffers, save=save, show=show, movie_name=movie_name)

    def step(self, action, cursor_thread, timestep):
        '''
        Agent instance calls this function every step while training RL
        Once it's called, it evaluates the action and gives a reward
        It accesses to every pallet in the order they are initialized, and move them
        As soon as it finds a pallet that needs to be assigned to a tester, it checks which agent should manage it
        Fix the cursor_thread accordingly, and returns "states, reward, done, {"buffers": self.buffers}"

        A series of states is saved in memoryA and memoryB
        The states that are used for training are the states contained in memoryA/B at that moment.
        '''

        # pallet that should be assigned to a tester
        a = self.pallets[self.cursor]

        # evaluate and generate a reward
        if action == 5:
            # on hold
            reward = -1
        else:
            routes = a.autopilot(flag='rl', floor=action)
            
            if routes == False:
                reward = -2
            else:
                a.move(a.actions[0])
                
                reward = np.count_nonzero(self.get_memory(tester_type=a.tester_type()) > 2) / 25

                current_plane = self.get_memory(tester_type=a.tester_type()).reshape((4,14,8))[0]
                crowdness = []
                for n_floor in range(5):
                    crowdness.append(0.8 * np.count_nonzero(current_plane[n_floor*3,1:7]>2) + 0.2 * np.count_nonzero(current_plane[n_floor*3+1,1:7]>2))
                print(crowdness)
                u, inv, counts = np.unique(crowdness, return_inverse=True, return_counts=True)
                csum = np.zeros_like(counts)
                csum[1:] = counts[:-1].cumsum()
                crowdness_rank = csum[inv]
                crowdness_rank_chosen = crowdness_rank[a.target[1]-1]
                print(crowdness_rank_chosen, min(crowdness_rank), crowdness_rank)

        if self.cursor == self.pallet_counts -1:
            # save the status for rendering
            self.saveBuffer(self.title)

        # point the next pallet
        self.cursor += 1

        # move every pallet and break when it finds a pallet that needs to be assigned to a tester        
        while True:
            # Simulation
            self.cursor = self.cursor % self.pallet_counts
            a = self.pallets[self.cursor]

            if a.state == None:
                a.enter()

            if a.state is not None:
                if a.done == False:                
                    if len(a.actions) == 0:
                        self.update_memory()
                        break
                        
                    elif len(a.actions) > 0:
                        a.move(a.actions[0])

               
                if a.done == True:
                    self.done_count += 1                

                self.update_memory()

                if self.done_count == self.pallet_counts:
                    print("ALL DONE, SIMTIME:", self.sim_time)
                    break

            self.cursor += 1

            if self.cursor == self.pallet_counts -1:
                self.saveBuffer(self.title)

        obs = self.get_memory(tester_type=a.tester_type()) # current state from the memory dictionary         

        if self.done_count == self.pallet_counts:
            self.done = True
        print(self.done_count, '/', self.pallet_counts)
        log_dir = logDir()+self.args.prefix+"/log"
        os.makedirs(log_dir, exist_ok=True)
        csv_path = (log_dir+'/log.model{}.csv').format(cursor_thread)
        if not os.path.exists(csv_path):
            with open(csv_path, 'wt') as file_handler:
                file_handler.write('#%s\n' % json.dumps({'args':vars(self.args)}))

        with open(csv_path, 'a') as file_handler:
            file_handler.write(str(reward)+","+str(time.time())+"\n")


        # change the cursor depending on 
        if self.pallets[self.cursor].tester_type() == 'a':
            self.cursor_thread = 0
        else:
            self.cursor_thread = 1

        return obs, reward, self.done, {"buffers": self.buffers}

    def current_pallet(self):
        return self.pallets[self.cursor]

    def empty_obs(self, tester_type):
        ys = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        if tester_type == "a":
            xs = [0,1,2,3,4,5,6,7]
        else: # tester_type == "b":
            xs = [7,8,9,10,11,12,13,14]

        result = np.zeros((len(ys), len(xs)))

        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                map_type = self.map.map[y][x]
                if map_type == self.map.invalid_flag:
                    # 빈칸
                    result[i][j] = -1
                elif map_type == self.map.lift_flag:
                    # Lift
                    result[i][j] = 0
                elif map_type == self.map.tester_flag:
                    # Tester
                    result[i][j] = 0
                elif map_type == self.map.path_flag:
                    # Path
                    result[i][j] = 0

        return result, ys, xs

    def flatten_obs(self, result):
        r = result.flatten()
        r = np.delete(r, np.where(r < 0))

        return r

    def obs(self, tester_type):
        self.update_memory()
        return self.get_memory(tester_type)

    def get_memory(self, tester_type):
        '''
        get current state
        '''
        a = self.pallets[self.cursor]
        if a.state is not None:
            if tester_type == 'a':
                memory = copy.deepcopy(self.memoryA)
            else:
                memory = copy.deepcopy(self.memoryB)
            
            i = a.state[0]
            j = 0
            memory[0][i][j] = 10

            return np.array(memory).flatten()
        else:
            if tester_type == 'a':
                return np.array(self.memoryA).flatten()
            else:
                return np.array(self.memoryB).flatten()

    def update_memory(self):
        '''
        update the memory dictionaries every time it moves a pallet
        '''
        result, ys, xs = self.empty_obs('a')

        # Pallet 분포
        for pallet_idx in self.pallets:
            a = self.pallets[pallet_idx]
            if a.target is not None:
                if a.target[0] == 'a':
                    i = 3 * a.target[1] + 1 # floor
                    j = a.target[2] + 1

                    result[i][j] = 2 # Occupied / Reserved
            if a.state is not None:
                if a.state[0] in ys and a.state[1] in xs:
                    i = ys.index(a.state[0])
                    j = xs.index(a.state[1])

                    if result[i][j] != 2:
                        result[i][j] = 1 # Pallet Located
                    if a.test_time > 0:
                        # 테스트중일 경우...
                        result[i][j] = 2 + a.test_time / self.map.tester_mean
        del self.memoryA[-1]
        self.memoryA.insert(0, result)

        result, ys, xs = self.empty_obs('b')

        for pallet_idx in self.pallets:
            a = self.pallets[pallet_idx]
            if a.target is not None:
                if a.target[0] == 'b':
                    i = 3 * a.target[1] + 1 # floor
                    j = a.target[2] + 1

                    result[i][j] = 2 # Occupied / Reserved
            if a.state is not None:
                if a.state[0] in ys and a.state[1] in xs:
                    i = ys.index(a.state[0])
                    j = xs.index(a.state[1])

                    if result[i][j] != 2:
                        result[i][j] = 1 # Pallet Located
                    if a.test_time > 0:
                        # 테스트중일 경우...
                        result[i][j] = 2 + a.test_time / self.map.tester_mean

        del self.memoryB[-1]
        self.memoryB.insert(0, result)
    
    def check_plane(self, state = (1, 0)):
        # for debugging
        result = np.zeros((len(self.map.map), len(self.map.map[0])))

        # 맵 구성
        for i in range(len(self.map.map)):
            for j in range(len(self.map.map[0])):
                map_type = self.map.map[i][j]
                if map_type == self.map.invalid_flag:
                    # 빈칸
                    result[i][j] = -1
    
        for pallet_idx in self.pallets:
            a = self.pallets[pallet_idx]
            if a.target is not None:
                if a.target[0] == "a":
                    i = 3 * a.target[1] + 1 # floor
                    j = a.target[2] + 2
                else:
                    i = 3 * a.target[1] + 1
                    j = a.target[2] + 2 + 7
                
                result[i][j] = 2 # Occupied / Reserved
                
            if a.state is not None:
                i = a.state[0]
                j = a.state[1]

                if result[i][j] != 2:
                    result[i][j] = 1 # Pallet Located
                if a.test_time > 0:
                    result[i][j] = 2 + a.test_time / self.map.tester_mean
        
        if state is not None:
            result[state[0]][state[1]] = -1
        # result = np.concatenate((np.array(result_a), np.array(result_b)), axis=1)
        # result = result[::-1]
        return result


if __name__ == "__main__":
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)

    env = FloorEnv(args=args)
    
    random.seed(1234)

    def simulate(env, autopilot_flag="min"):
        env.reset()
        env.title = autopilot_flag

        while True:
            action = env.current_pallet().autopilot(flag=autopilot_flag, return_floor=True)
            # print("")
            # print("ITERATION:",i)
            obs, reward, done, info = env.step(action)

            # print(reward)
            if done == True:
                print("ELAPSED SIM-TIME: ", env.sim_time, " | RL")
                break

        return env.buffers

    def rl_test(env):
        env.reset()

        while True:
            action = random.choice([0,1,2,3,4,5])
            # print("")
            # print("ITERATION:",i)
            obs, reward, done, info = env.step(action)
            # print(reward)
            if done == True:
                print("ELAPSED SIM-TIME: ", env.sim_time, " | RL")
                break

        return env.buffers


    buffers = []
    for autopilot_flag in ["fcfs"]:
        env.title = autopilot_flag
        buf = simulate(env, autopilot_flag)
        buffers.append(buf)

        # env.render(movie_name="fcfs_200", save=True)

    # buf = rl_test(env)
    # buffers.append(buf)

    buffers = dict(pair for d in buffers for pair in d.items())

    env.render(buffers=buffers,movie_name="autopilots_200", save=True, show=False)
