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


## 2번째 팔레트 target 지정까지 잘 되는데, move해도 안움직인다. move하고 정말 안움직이는게 맞는건지 확인하고
# 움직여야하는거였다면 왜 안움직인건지 확인
# -> lift 사용중이라 안들어가던거였음 pallet.py move함수의 리프트 사용중, 충돌 여부 함수
# move마다 pallets가 어떻게 돌아가는지 step 함수처럼 구현을 해서 확인해보자

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