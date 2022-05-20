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
    def __init__(self, args=None, dim=2, title="RL"):

        self.debug_obs = []
        self.steps = []



        self.pallet_counts = args.pallet_counts
        self.title = title
        self.dim = dim
        self.args = args

        obs = self.reset()
        self.resetBuffer()

        self.action_space = spaces.Discrete(5)

        ##################################################
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
        self.map = Map()
        self.pallet_idx = 0
        self.pallets = {}

        self.cursor = 0 # Pallet IDX
        self.done_count = 0
        self.done = False

        self.sim_time = 0

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

        self.saveBuffer(self.title)

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

        self.map.pallets = self.pallets
        self.buffers[buffer_type].append(copy.deepcopy(self.pallets))

        self.sim_time += 1

    def render(self, buffers=None, save=False, show=True, still=False, movie_name="movie_name"):
        self.map.pallets = self.pallets
        if still == True:
            self.map.render(buffers=None)
        else:
            if buffers is None:
                buffers = self.buffers
            self.map.render(buffers=buffers, save=save, show=show, movie_name=movie_name)

    def step(self, action, cursor_thread):
        '''
        이미 Route가 Assign된 애들은 simulate하고, 액션이 필요한 애만 지정해야함.
        따라서 Return하는 State는 다음에 Action이 필요한 pallet가 바라본 현 상태여야함.
        Action : 1~5층 중 어디에 넣을까!
        가만히 있는 action이 필요할까? 만약 모든 검사기가 꽉 찼으면.. Penalize를 하면 안됨

        220428 TODO
        Window를 지정했을 때 메모리에 저장할 데이터는 action이 필요했던 순간들을 저장해야할까
        아니면 매 시뮬레이션 스텝마다 저장해서 봐야할까 아마 이거인듯.. 근데 위에껄로 되어있음 고치자
        '''

        # 0519
        self.steps = []

        # print("IN STEP:", cursor_thread, id(self))
        a = self.pallets[self.cursor]

        if action == 5:
            # 대기
            reward = np.count_nonzero(self.obs(tester_type=a.tester_type()) == 2) / 25
        else:
            # 대기가 아님
            routes = a.autopilot(flag='rl', floor=action)
            
            if routes == False:
                # 해당 검사기가 꽉참. Penalize!
                # print("FULL", a.id)
                reward = -1
            else:
                # print("RUN RL ACTION ID", a.id, a.state, a.target, a.test_count, self.done_count)
                a.move(a.actions[0])
                
                # Assign된 검사기의 수 리턴
                reward = np.count_nonzero(self.obs(tester_type=a.tester_type()) == 2) / 25

        ################# 0519
        ## debuging을 위해 action을 취한 직후의 memory를 그대로 빼와봤다
        self.debug_obs = self.spit_memoery().copy()

        if self.cursor == self.pallet_counts -1:
            # 한바퀴를 다 수행하였을 때만 현화면 저장
            self.saveBuffer(self.title)

        # 대기 혹은 wrong assign 때 다른 pallet에 대해 simulate 진행
        # 어차피 다음 차례에 이 pallet로 돌아옴
        self.cursor += 1
        
        while True:
            # Simulation
            self.cursor = self.cursor % self.pallet_counts
            a = self.pallets[self.cursor]
            # 입장 처리
            if a.state == None:
                a.enter()

            # 입장이 됨

            ## 0519
            print('plane added to steps ------------')
            self.steps.append(self.check_plane())
            
            if a.state is not None:
                if a.done == False:                
                    if len(a.actions) == 0:
                        # Action이 필요한 애가 선정.
                        # print("#####")
                        # print("ID", a.id, a.state, a.target)
                        # print("BREAK")

                        break
                        
                    if len(a.actions) > 0:

                        #print(a.actions[0])                     
                        a.move(a.actions[0]) 

                if a.done == True:
                    # print("DONE: ", a.id)
                    self.done_count += 1
                
                if self.done_count == self.pallet_counts:
                    print("ALL DONE, SIMTIME:", self.sim_time)
                    break

            self.cursor += 1

            if self.cursor == self.pallet_counts -1:
                # 한바퀴를 다 수행하였을 때만 현화면 저장
                self.saveBuffer(self.title)

        obs = self.obs(tester_type=a.tester_type()) # 현상태의 state           

        if self.done_count == self.pallet_counts:
            self.done = True

        log_dir = logDir()+self.args.prefix+"/log"
        os.makedirs(log_dir, exist_ok=True)
        csv_path = (log_dir+'/log.model{}.csv').format(cursor_thread)
        if not os.path.exists(csv_path):
            with open(csv_path, 'wt') as file_handler:
                file_handler.write('#%s\n' % json.dumps({'args':vars(self.args)}))

        with open(csv_path, 'a') as file_handler:
            file_handler.write(str(reward)+","+str(time.time())+"\n")

        return obs, reward, self.done, {"buffers": self.buffers}

    def current_pallet(self):
        return self.pallets[self.cursor]

    def empty_obs(self, tester_type):
        ys = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        if tester_type == "a":
            xs = [1,2,3,4,5,6,7]
        else: # tester_type == "b":
            xs = [8,9,10,11,12,13,14]

        result = np.zeros((len(ys), len(xs)))

        # 맵 구성
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

    def obs(self, tester_type, renew_memory=True):
        result, ys, xs = self.empty_obs(tester_type)

        # Pallet 분포
        for pallet_idx in self.pallets:
            a = self.pallets[pallet_idx]
            if a.target is not None:
                if a.target[0] == tester_type:
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

        if self.dim == 1:
            r = self.flatten_obs(result)
        elif self.dim == 2:
            r = result
        
        if self.args.window_size > 0:
            if tester_type == 'a':
                del self.memoryA[-1]
                self.memoryA.insert(0, r)
                return np.array(self.memoryA).flatten()

            else:
                del self.memoryB[-1]
                self.memoryB.insert(0, r)
                return np.array(self.memoryB).flatten()

            # TODO dim=2인 경우에 flatten을 할 경우 차원 사라짐
            return np.array(self.memory).flatten()
        else:
            return r

    def spit_memoery(self, reverse=False):
        mem = np.array(self.memoryA)
        if reverse:
            for i in range(len(mem)):
                mem[i] = mem[i][::-1]
        return mem

    def spit_debug_obs(self):
        do = self.debug_obs.copy()
        for i in range(len(do)):
            do[i] = do[i][::-1]
        return do
    
    def check_plane(self):
        result_a, ys, xs = self.empty_obs("a")
        for pallet_idx in self.pallets:
            a = self.pallets[pallet_idx]
            if a.target is not None:
                if a.target[0] == "a":
                    i = 3 * a.target[1] + 1 # floor
                    j = a.target[2] + 1

                    result_a[i][j] = 2 # Occupied / Reserved
            if a.state is not None:
                if a.state[0] in ys and a.state[1] in xs:
                    i = ys.index(a.state[0])
                    j = xs.index(a.state[1])

                    if result_a[i][j] != 2:
                        result_a[i][j] = 1 # Pallet Located
                    if a.test_time > 0:
                        result_a[i][j] = 2 + a.test_time / self.map.tester_mean
        
        result_b, ys, xs = self.empty_obs("b")
        for pallet_idx in self.pallets:
            a = self.pallets[pallet_idx]
            if a.target is not None:
                if a.target[0] == "b":
                    i = 3 * a.target[1] + 1 # floor
                    j = a.target[2] + 1

                    result_b[i][j] = 2 # Occupied / Reserved
            if a.state is not None:
                if a.state[0] in ys and a.state[1] in xs:
                    i = ys.index(a.state[0])
                    j = xs.index(a.state[1])

                    if result_b[i][j] != 2:
                        result_b[i][j] = 1 # Pallet Located
                    if a.test_time > 0:
                        result_b[i][j] = 2 + a.test_time / self.map.tester_mean
        
        result = np.concatenate((np.array(result_a), np.array(result_b)), axis=1)
        result = result[::-1]
        return result
    
    def spit_steps(self):
        return np.array(self.steps)


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
