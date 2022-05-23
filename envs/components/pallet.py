import numpy as np

class Pallet:
    def __init__(self, map, id, enter, env):
        # Start Point
        if enter == False:
            self.state = None
        else:
            self.state = (0, 0)
        self.map = map
        self.id = id
        self.target = None
        self.test_count = 0
        self.done = False
        self.test_time = 0

        self.actions = []

    def enter(self):
        # 입장시 다른 팔레트가 이미 있는지 확인
        if self.map.entrance_status() == False and self.test_count == 0:
            self.done = False
            self.state = (0, 0)

    def exit(self):
        self.state = None
        self.target = None
        self.test_count += 1

    def move(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position on board
        """

        if action == "down" or action == "d":
            next_state = (self.state[0] - 1, self.state[1])
        elif action == "up" or action == "u":
            next_state = (self.state[0] + 1, self.state[1])
        elif action == "left" or action == "l":
            next_state = (self.state[0], self.state[1] - 1)
        else:
            next_state = (self.state[0], self.state[1] + 1)
        
        # if next state is legal
        # if (next_state[0] >= 0) and (next_state[0] <= self.map.y_limit):
        #     if (next_state[1] >= 0) and (next_state[1] <= self.map.x_limit):
        #         if self.map.map_value(next_state) != "INVALID":
        #             self.state = next_state
        # 다음 위치가 valid한지 확인
        # 리프트 사용중, 충돌 여부
        moveable = True
        for lift_type in self.map.lifts:
            lift = self.map.lifts[lift_type]
            if next_state[1] in lift["x"]:
                # 이 리프트에 타려고함
                c = self.map.lift_status(lift_type)
                if c > 0:
                    # 리프트 사용중 -> 대기
                    # 그게 자기 자신이면 -> 이동
                    s, _, _ = self.location()
                    if s == "LIFT":
                        pass
                    else:
                        moveable = False

        # 충돌 여부 확인
        if self.map.is_occupied(next_state) == True:
            moveable = False

        # 검사 로직
        s_before = self.map.map_value(self.state)
        s_after  = self.map.map_value(next_state)


        if s_before == "PATH" and s_after == "TESTERS":
            # 검사기에 진입
            self.test_count += 1
            self.test_time = 0
            # self.test_target_time = random.randint(200,300) # 4 ~ 7 timesteps 만큼 랜덤으로 머무름 # Uniform
            self.test_target_time = np.random.normal(loc=self.map.tester_mean, scale=self.map.tester_std, size=1).astype(int)[0]  # Normal Distribution


        if s_before == "TESTERS" and s_after == "PATH":
            # 검사기에서 탈출
            self.test_time += 1
            if self.test_target_time > self.test_time:
                moveable = False
                
            ##### 0523 김건
            ## tester에서 나가는 순간 그 tester는 비어잇는거니까 거기 reserved되어있다고 판단하지 않게끔
            ## pallet의 target도 None으로 바꿔주기
            if moveable == True:
                self.target = None

        if moveable == True:
            
            self.state = next_state
            self.actions.pop(0)

            if s_after == "EXIT":
                self.exit()
                self.done = True

        return self.state

    def setTarget(self, tester_type, floor):
        
        # 해당 층에 몇개의 테스터기가 점유되어 있는가
        counts = self.map.tester_status(tester_type)[floor]

        if sum(counts) >= 5:
            # 꽉 참
            return False

        # 경로 계산
        actions = ["r"]

        # 검사기 앞까지 (y)
        target_y = self.map.testers[tester_type]["y"][floor][0] - self.state[0]
        if target_y > 0:
            actions += ["u"] * target_y
        elif target_y < 0:
            actions += ["d"] * abs(target_y)

        # 검사기 앞까지 (x) / 오른쪽부터 채움
        t_idx = list(reversed(counts)).index(0)
        target_x = len(counts) - t_idx - 1

        ###여기까지 이상 없음

        actions += ["r"] * (target_x + 1)

        # 검사기 진입 후 탈출
        actions += ["u"]
        actions += ["d"]
        #self.test_count+=1

        # Lift 바로 앞까지 이동
        actions += ["r"] * (1 + t_idx)

        if tester_type == "b":
            # 종료로
            actions += ["r"]
            actions += ["d"] * self.map.testers[tester_type]["y"][floor][0]
            actions += ["r"]

        self.actions = actions
        self.target = [tester_type, floor, target_x]
        
        return actions

    def tester_type(self):
        if self.test_count == 0:
            return "a"
        elif self.test_count == 1:
            return "b"
        else:
            return None

    def autopilot(self, flag="min", floor=None, return_floor=False):
        if self.test_count == 0:
            tester_type = "a"
        elif self.test_count == 1:
            tester_type = "b"
        else:
            # 테스트 완료함
            return []

        '''
        flag : min or fcfs
        '''
        if flag == "min":
            # 각 층의 최소
            tester_status = self.map.tester_status(tester_type)
            floor = tester_status.index(min(tester_status))
        elif flag == "fcfs":
            # 앞부터 채움
            tester_status = self.map.tester_status(tester_type)
            for floor, testers in enumerate(tester_status):
                if sum(testers) < 5:
                    break
        elif flag == "rl":
            floor = floor

        r = self.setTarget(tester_type, floor)

        if return_floor == True:
            # 액션 빼앗기용
            return floor

        if r == False:
            # 대상이 꽉 참
            return False
        else:
            return r 

    def location(self):
        s = self.map.map_value(self.state)
        if s == "PATH" or s == "TESTERS":
            p = self.state
            x = p[1]
            y = p[0]

            for tester_type in self.map.testers:

                xs = self.map.testers[tester_type]["x"]
                ys = self.map.testers[tester_type]["y"]

                floor = None
                status = False
                if x in xs:
                    # x 범위에 들어옴
                    for f, _y in enumerate(ys):
                        if y in _y:
                            # 위치 찾음
                            status = True
                            floor = f

                return s, tester_type, floor                    
        else:
            return s, None, None