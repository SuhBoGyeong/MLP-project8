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
    
    #### 0523 김건 ####
    # 나중에 시각화 한 것을 이해하는데 쓸듯. saveBuffer 괜히 건드리면 안될거같아서 그냥 따로 만듬
    def resetPalletBuffer(self):
        self.pallet_buffer = []

    def savePalletBuffer(self):
        # if not timestep:
        #     self.pallet_buffer[timestep] = []
        
        self.pallet_buffer = []
        for key in self.pallets:
            pallet = self.pallets[key]
            if pallet.state is None:
                state = None
            else:
                state = list(pallet.state)
            self.pallet_buffer.append([pallet.id, state])
        self.pallet_buffer = np.array(self.pallet_buffer)
        # self.pallet_buffer[timestep].append(copy.deepcopy(self.pallets))
    ##################


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
        이미 Route가 Assign된 애들은 simulate하고, 액션이 필요한 애만 지정해야함.
        따라서 Return하는 State는 다음에 Action이 필요한 pallet가 바라본 현 상태여야함.
        Action : 1~5층 중 어디에 넣을까!
        가만히 있는 action이 필요할까? 만약 모든 검사기가 꽉 찼으면.. Penalize를 하면 안됨

        220428 TODO
        Window를 지정했을 때 메모리에 저장할 데이터는 action이 필요했던 순간들을 저장해야할까
        아니면 매 시뮬레이션 스텝마다 저장해서 봐야할까 아마 이거인듯.. 근데 위에껄로 되어있음 고치자
        '''
        # print("IN STEP:", cursor_thread, id(self))
        a = self.pallets[self.cursor]

        if action == 5:
            # 대기
            reward = np.count_nonzero(self.get_memory(tester_type=a.tester_type()) == 2) / 25
        else:
            # 대기가 아님
            routes = a.autopilot(flag='rl', floor=action)
            
            if routes == False:
                # 해당 검사기가 꽉참. Penalize!
                # print("FULL", a.id)
                # pallet에 action 지정 자체가 안되게 됨. 얘는 그럼 enter도 안되고 걍 있는것임.
                reward = -1
            else:
                # print("RUN RL ACTION ID", a.id, a.state, a.target, a.test_count, self.done_count)
                # a.move(a.actions[0])
                
                # Assign된 검사기의 수 리턴
                reward = np.count_nonzero(self.get_memory(tester_type=a.tester_type()) == 2) / 25

        if self.cursor == self.pallet_counts -1:
            # 한바퀴를 다 수행하였을 때만 현화면 저장
            self.saveBuffer(self.title)

        # 대기 혹은 wrong assign 때 다른 pallet에 대해 simulate 진행
        # 어차피 다음 차례에 이 pallet로 돌아옴
        # self.cursor += 1
        
        count = 0
        while True:
            count += 1
            # Simulation
            self.cursor = self.cursor % self.pallet_counts
            a = self.pallets[self.cursor]
            # 입장 처리
            if a.state == None:
                a.enter()

            ##############
            # 입장 처리가 되던 말던 그냥 무조건 모든 step을 찍으려고 한다.
            if a.target is not None:
                temp = (a.target[1], a.target[2])
            else:
                temp = None
            plane = self.check_plane(a.state)
            self.savePalletBuffer()
            if cursor_thread == 0:
                np.save(f'./everystep/memory0/binary/ts{timestep:03d}-itr{count:03d}-p{self.cursor}:s{a.state},t{temp}.npy', plane)
                np.save(f'./everystep/memory0/pallets/ts{timestep:03d}-itr{count:03d}.npy', self.pallet_buffer)
            ##############

            # 입장이 됨
            if a.state is not None:
                if a.done == False:                
                    if len(a.actions) == 0:
                        # 입장이 돼서 state도 있고, done도 아닌데
                        # actions가 비어있다? 새로 tester에 배정되어야하는 상태인 것!
                        # 따라서 break!
                        break
                        
                    elif len(a.actions) > 0:
                        a.move(a.actions[0])

                # 위에서 move 후 a.done이 True로 바뀌었으면 그거 카운트해줌
                if a.done == True and a.test_count == 2:
                    # print("DONE: ", a.id)
                    self.done_count += 1
                    a.test_count += 1
                '''
                0523
                exit으로 가서 a.done이 True가 된 후, 다시 cursor가 돌아서 이 pallet로 왔다고 하자
                그럼 enter처리가 될것이지만, pallet.py코드를 보면 enter처리가 됐다고 해서 a.done이 False로 바뀌는 부분은 없었다.
                그럼 입장이 된 후, if a.done == False에 걸리지도 않을 것이고, 또 자연스래 if a.done == True에 걸려 self.done_count를 올려버릴 것이다.
                이 경우 팔레트 하나만 완료돼도 self.done_count를 계속 올려버릴 수 잇는것 아닌가?
                
                그래서 a.test_count == 2라는 조건을 붙여주고, 체크 한번 했으면 test_count += 1을 해줘서 더 이상
                중복되어 self.done_count가 올라가지 않도록 했다.
                
                '''

                # break하는 경우는 action이 필요한 경우
                # action이 필요한 pallet 전에 마지막으로 움직인 pallet가 있을 것
                # 여기에 update_memory를 배치함으로써 걔가 움직이고 난 직후 상황이 여기에 memory에 마지막에 저장되게 된다.
                self.update_memory(a.tester_type())

                if self.done_count == self.pallet_counts:
                    print("ALL DONE, SIMTIME:", self.sim_time)
                    break

            self.cursor += 1

            if self.cursor == self.pallet_counts -1:
                # 한바퀴를 다 수행하였을 때만 현화면 저장
                self.saveBuffer(self.title)

        obs = self.get_memory(tester_type=a.tester_type()) # 현상태의 state           

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

    def obs(self, tester_type):
        self.update_memory(tester_type)
        return self.get_memory(tester_type)

    def get_memory(self, tester_type):
        # window size 0인 경우 제외
        if tester_type == 'a':
            return np.array(self.memoryA).flatten()

        else:
            return np.array(self.memoryB).flatten()

    def update_memory(self, tester_type):
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
                        # 0520 테스트가 맞쳐진 경우 값이 제 각각이긴 할 것이다.
                        result[i][j] = 2 + a.test_time / self.map.tester_mean

        # window_size가 0인 경우 그냥 일단 제외하고 만들었다.
        if tester_type == 'a':
            del self.memoryA[-1]
            self.memoryA.insert(0, result)
        else:
            del self.memoryB[-1]
            self.memoryB.insert(0, result)
    
    def check_plane(self, state = (1, 0)):
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
