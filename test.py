import sys
import numpy as np
import matplotlib.pyplot as plt
from envs.floors_env import FloorEnv
from utils.arg_parser import common_arg_parser, parseLayersFromArgs

def main(args):
    ### 초기화 및 객체들 준비 ###
    '''
    FloorEnv를 만들면 reset함수가 돌면서 알아서 map, pallets들이 초기화 된다.
    다만 map에 pallet들을 넣어주는 부분이 코드 전체에 아무곳에도 없었고, 그것때문에 전체 로직이
    완전히 틀어져 난리가 났던거같다. reset함수 부분에 0520주석과 함께 그 부분을 추가해주었고,
    이 테스트를 다시 진행해보았다.
    간단한 테스트를 위해 pallet은 arg parser에서 5개로 세팅해놨다.

    테스트하면서 알게된 부분, 결과 등은 제일 아래에 조금 정리하겠다.
    '''
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    env = FloorEnv(args=args)
    pallets = env.pallets
    ### 초기화 완료 ###


    ### 간단한 초기 세팅 ###
    '''
    1번째와 2번째 팔레트를 같은 층에 배치해봄으로써 과연 2번째 팔레트는 첫번째 팔레트가 있는 곳의 좌측에
    배정되는지 확인하려 했다.
    '''
    pallets[0].autopilot(flag='rl', floor=2)
    pallets[0].move(pallets[0].actions[0])
    pallets[1].enter()
    pallets[1].autopilot(flag='rl', floor=2)
    print(f'pallet0 state : {pallets[0].state}, target : {pallets[0].target}')
    print(f'pallet1 state : {pallets[1].state}, target : {pallets[1].target}')
    ### 초기 세팅 완료 ### 

    ### 어떤 상황인지 확인 ###
    '''
    위의 초기 세팅 결과를 확인해보기 위함이다. 필요하면 저장해서 확인해보고 경로는 알아서 하자. 그대로 할거면 main함수가 있는 경로에 tests폴더를 만들어야할것이다.
    '''
    ###############33
    ############### 참고로 check_plane()은 상하 반전되서 나온다.
    # 그냥 print로 찍을꺼면 [::-1]같은거 붙여서 반전 시키고, plt.imshow할거면 그냥 하면 된다. axis 범위 내가 알아서 잘 설정해놓음
    
    #####******###### check_plane은 입구와 출구까지 보여주지 않는다. 따라서 팔레트가 0, 0에 있으면 안보이는게 정상이다.
    
    ob = env.check_plane()
    plt.imshow(ob, cmap='gray', vmin=-1.0, vmax=4.0)
    plt.axis([-0.5, 13.5, -0.5, 13.5])
    plt.title('살려줘ㅠㅠ')
    plt.savefig(f'test/0.png')
    ### 확인 완료 ###

    ### 테스트 ###
    '''
    pallet가 target이 정해져있으면 거기까지 들어갈때까지 그냥 move를 갈겨버렸다.
    actions에서 상하좌우 움직임 하나씩 잘 따라가나 확인하기 위함도 있고,
    한칸 갈때마다 actions[0]이 제대로 제거되는지 보려고 했고,
    target위치까지 잘 가나 확인도 했고,
    그때 그때 자신의 위치(state)도 잘 가지고 있나 확인하기 위함이었다.

    필요하면 알아서 시각화해보자
    '''
    for idx in pallets:
        a = pallets[idx]
        print(f'---agent{idx}')
        print(f'actions : {a.actions}')
        print(f'state : {a.state}')
        print(f'target : {a.target}')
        
        while(True and idx == 0):
            if a.actions[0] == 'd':
                a.move(a.actions[0])
                break
            a.move(a.actions[0])

        if a.actions:
            print(f'move! the new state : {a.move(a.actions[0])}')
            print(f'actions left : {a.actions}')
        
        ob = env.check_plane()
        plt.imshow(ob, cmap='gray', vmin=-1.0, vmax=4.0)
        plt.axis([-0.5, 13.5, -0.5, 13.5])
        plt.title(f'#{idx + 1} action 2')
        plt.savefig(f'test/{idx + 1}.png')
    ### 테스트 완료  ###

    ### 220522 보경 추가
    '''
    앞에서 tester a까지의 행동 수행한 후, pallet[0]에 다시 autopilot 적용하면
    다음 tester b 를 거치고 exit까지의 행동을 할당할 것. 예시로 floor=4로 설정 후 확인해봄.
    근데 위 코드에서 action이 d 일 경우 break 하게끔 설정되어 있어서, 
    일단은 임시방편으로 남은 action ['d', 'r'] 수행하는 것도 추가해 줌.
    즉, 처음으로 할당된 action이 우선 모두 실행되게끔 한다. 

    '''
    print('left actions from 1 stage of pallet 0 (~tester a): ', pallets[0].actions, 'current state: ', pallets[0].state)
    k = len(pallets[0].actions)
    pallets[0].test_target_time = 1   #### test_time=1 으로 우선 줘서, 바로 움직이게 하자.

    for i in range (k):
        pallets[0].move(pallets[0].actions[0])
        print(f'{i+1}/{k}: state - ', pallets[0].state, ' left actions - ', pallets[0].actions)

        
    print('after actions from 1 stage, state: ', pallets[0].state)

    ### 이제 tester b구역 진입, autopilot 이용해 action 할당. 임의로 floor=4 설정.
    pallets[0].autopilot(flag='rl', floor=4)
    print(f'pallet0 state : {pallets[0].state}, target : {pallets[0].target}')
    print('next actions for ~tester b ~ exit: ', pallets[0].actions)

    #### 일단 action은  제대로 할당되는 거 확인함. (exit까지의 action이 옳게 주어짐)
    ###마찬가지로 여기서도 test_time=1로 임의로 할당해 바로 움직이게 하자. (그냥 for 문 안에 줘버려도 상관없어서 일단 그렇게 함)
    
    k = len(pallets[0].actions)
    for i in range(k):
        pallets[0].test_target_time = 1
        pallets[0].move(pallets[0].actions[0])
        print(f'{i+1}/{k}: state - ', pallets[0].state, ' left actions - ', pallets[0].actions)

    ###exit 후 state None으로 바뀌면서 퇴장한 거 확인할 수 있다.

    ### 테스트 완료 ###  

    print('------------------------------------')
    '''
    테스트기 시간 할당 관련
    a 테스터기 나와도 시간을 유지한다고 되어있는데, b로 들어갈 때 어떻게 되는지 확인해야 한다.
    앞 코드에서 pallets[0]을 사용했고 퇴장시켰기 때문에 pallets[1]을 이용하자.
    '''

    while pallets[1].actions:
        print('current state: ', pallets[1].state, 'left actions: ', pallets[1].actions)
        if pallets[1].actions[0] == 'd':
            pallets[1].move(pallets[1].actions[0])
            break
        pallets[1].move(pallets[1].actions[0])
    
    print('1. now the pallet gets its test_target_time: ', pallets[1].test_target_time)
    while pallets[1].actions:
        print('current state: ', pallets[1].state, 'left actions: ', pallets[1].actions)
        pallets[1].move(pallets[1].actions[0])
    print('after testing in the a: ', pallets[1].state, pallets[1].actions, pallets[1].test_target_time)

    ### 이제 tester b로 가게끔 action 할당하자. 
    pallets[1].autopilot(flag='rl', floor=4)
    print(pallets[1].test_target_time)
    ### 일단 a를 나온 후에도 test_target_time은 동일한 값으로 유지됨. 
    while pallets[1].actions:
        #print(pallets[1].test_target_time)
        print('current state: ', pallets[1].state, 'left actions: ', pallets[1].actions)
        if pallets[1].actions[0] == 'd':
            pallets[1].move(pallets[1].actions[0])
            break
        pallets[1].move(pallets[1].actions[0])
    
    print('2. now the pallet gets its test_target_time: ', pallets[1].test_target_time)
    while pallets[1].actions:
        print('current state: ', pallets[1].state, 'left actions: ', pallets[1].actions)
        pallets[1].move(pallets[1].actions[0])
    print('after testing in the a: ', pallets[1].state, pallets[1].actions, pallets[1].test_target_time)
    
    ### 코드를 여러 번 실행하다보면, print 한 라인 중 1. now the pallet gets its test_target_time과 2. // 가 
    ### 서로 다른 값을 가지는 경우 발생함을 확인했다. 
    ### 즉 b로 들어갈 때 test_target_time도 다시 할당되는 것 확인 함. 
    ### 근데 좀 많은 경우가 같은 값으로 할당되는 것 같은데, 기분탓인가????

    ### 테스트 완료 ###


if __name__ == "__main__":
    main(sys.argv)

'''
아래 사항들을 테스트해보았다.
*********->는 앞으로 한번 확인해봐야할 것들
1. autopilot : target과 actions(해당 tester기까지 가는 루트)를 잘 설정하는가
    -> 내부에 있는 setTarget 함수까지 정상 작동하는 것으로 기억함.
2. enter : enter를 시도하려고 할 때 다른 팔레트가 있으면 안들어가는게 맞는가, enter시 actions를
    잘 할당 받는가
    -> 잘 받음
3. move : 제대로 원하는 곳으로 움직이며, 움직일 수 없는 경우 움직이지 않는가. actions에서 행동한 부분은 알아서 잘 빼는가
    -> 제대로 잘 움직이는 듯. tester 들어간 후 시간 찰때까지 안나온다. 굿. actions도 잘 갱신된다.
    
    **********-> 아직 exit까지 가는 경우 어케 되는지 확인 안해봤지만, 이대로면 잘 될듯?
4. 테스터기에 들어가면 알아서 시간 할당 잘하는가? 시간 간격은 어떤가?
    -> 잘 할당한다.
    -> 하지만 time_mean이 50, std가 10이었는데, 이 경우 타임스텝을 50번이나 돌아야 테스트기에서 나오게된다.
        의도한 부분과 맞지 않다고 생각해 5, 1로 일단 줄였다.
    **********-> 다만 이 경우 테스터기에서 나와도 그 숫자를 유지함 (2보다 큰 숫자). 이게 b로들어갈때 어떨지는 아직
    확인은 안해봄
5. *********** b테스터기로 갈때는 어떻게 되는지, 그게 끝나고서도 exit으로 잘 나가는지 확인해보지 않았다.
    -> 이 경우 main함수를 돌려 쌓은 로그를 visualize.py로 시각화 해봄으로써 조금 확인했는데 어느정도 잘 되는 느낌이었다.
    visualize.py에 관한 설명은 해당 파일에 해놓겠다.
'''

'''
추가 노트
floorsEnv의 obs함수를
update memory, get memory로 쪼갰고, 이 두함수를 합쳐서 obs를 만들었다.
step함수에서 obs가 두번 불리는데 memory를 두번 update하게 되어 step함수 한번에
observation이 두번 들어가는 상황이 발생해버렸기 때문이다.
obs의 두가지 역할을 그냥 분리한거라 생각하면된다.

또한 위 함수를 만들면서 self.dim == 1인경우와 windowsize가 0인경우는 생각안했다.
그냥 하자...
'''