import numpy as np
import matplotlib.pyplot as plt

import os

## main함수가 있는 경로로부터 해당 path와 같이 폴더를 미리 만들고 main함수를 돌리면 log가 쌓인다.
## 그 후 이 파일을 실행하자
path1 = './everystep/memory0/binary/'
plane_li = os.listdir(path1)
plane_li.sort()
path2 = './everystep/memory0/pallets/'
pallets_li = os.listdir(path2)
pallets_li.sort()

for i in range(300):
    plane = plane_li[i]
    pallets = pallets_li[i]
    plane_path = path1 + plane
    pallets_path = path2 + pallets
    
    plane_f = np.load(plane_path)
    pallets_f = np.load(pallets_path, allow_pickle=True)

    plt.imshow(plane_f, cmap='gray', vmin=-1.0, vmax=4.0)
    plt.axis([-0.5, 16.5, -0.5, 13.5])
    
    states = ''
    for pallet in pallets_f:
        print(pallet)
        temp = f'{pallet[0]}-{pallet[1]}, '
        states += temp
    plt.title(states)
    # plt.annotate(states, xy=(0, 50), xytext=(50, 20) )
    plt.savefig('./everystep/memory0/'+plane[:-4]+'.png')
'''
### 간단한 설명 ###
main 함수를 돌려 쌓은 로그를 시각화 하는 파일이다.
먼저 test.py의 가장 아래 기록을 읽고 오는 것을 추천한다.

간단한 테스트를 위해 arg parser에서 timestep은 30, 팔레트 개수는 5개로 줄였다.

main함수를 돌리면 위 path아래에 로그가 쌓인다.
해당 로그는 floors_env.py의 step함수 안의 while loop에서 찍었다.
173~179번째 줄이 로그를 찍는 부분에 해당하며, check_plane()은 단순히 양옆 화면(리프트 3개, 테스트기 a, b전부 포함)을
정보를 가져와주는 놈이다.

dqn.py에서 thread 숫자 지정 로직이 내가 생각하는 맞는 방법과는 조금 다르게 돌아가던데, 이건 내가 아는게 없으니 일단 넘겼고
잘 보니까 그냥 model을 번갈아가면서 계속 학습시키길래 그냥 1번 모델기준으로 로그를 찍기 위해 cursor_thread는 0일때만으로 한정했다.

### 변수 설명 ###
- timestep은 dqn에서 돌아가는 그 iii라는 변수이다. 실제 timestep. 실제로 step함수가 호출되는 횟수와 같다.
    count는 시뮬레이션상 타임스텝. while문이 돌아갈때마다 하나씩 쌓인다.
    cursor는 팔레트의 index, state는 위치, temp는 target이다.

### 보기 전에 주의할 점 ###
- target에 나와있는 숫자는 실제 location index가 아니다. 순서대로 floor, target_x값으로
    floor는 층수, target_x는 idx가 아니라 왼쪽에서부터 센 테스트기의 위치이다.
    이 때 가장 왼쪽의 테스트기는 0번째라고 생각하면된다.
    주의!
- 파일 이름이 st001-itr020-p2:(0, 0),(3, 4)였다고 하자.
    이는 '2'번째로 불려진 step함수의 20번째 iteration이다.
    이는 3번째 팔레트의(0부터 시작이므로) 위치(0, 0)와 target(3층, 4 + 1번째 칸)를 나타낸다.
    이 팔레트가 움직일 수 있는 상태라면, 이 사진의 바로 다음번칸에서 움직일 것이다.
    로그를 찍는 부분이 while loop안에서 move가 일어나기 전에 위치하기 때문이다.
- check_plane은 입구와 출구까지 보여주지 않는다.
    따라서 팔레트가 0, 0에 있으면 사진에서 안보이는게 정상이다. 

### 생각해볼 점 ###
-********* 돌려보면 알겠지만, 0번째 timestep에서 0번째 팔레트 한번 찍히고(이건 아마 정상이다.),
    그다음 timestep에서 3, 4번째 팔레트만 찍힌다(이게 비 정상인거같은데 정상일수도).
    그 다음부터는 다 잘찍힌다.
    초반에 저게 왜 안찍히는지 확인해볼 필요가 있다.

-********* timestep이 넘어갈 때, 팔레트 위치에 급진적인 변화가 일어난다.
    이거 왜이런지 확인해야한다. 아마 정상일거같기는하다.
    -> 먼저, test가 끝나고 나와도 test기의 state가 reserved/occupied로 되어있는 것 같다.
        action이 취해지고 나면 정상 상태로 돌아오는것을 보니, 액션이 취해질 때 아마 제대로 처리가 되는거같긴하지만
        제대로 확인해야한다.
    -> cursor가 1에서 0으로 이동하는 경우가 있다. 제대로 확인.
    -> action이 취해질때 두 세 스텝씩 이동해버리는 경우가 있다.

- 또한 로그를 찍는 위치를 대충 설정했는데, 조금 더 생각해보고 바꿔봐도 될 수도 있다.

### 확인한 점 ###
- action이 필요한 놈들이 있을때마다 알아서 잘 탈출하고, 잘 지정해주는지
- timestep이 바뀌더라도 cursor가 알아서 잘 넘어가더라.

- 유심히 보면 전부 로직에 맞게 잘 움직이고, 돌아가는거같은 느낌은 든다! 100% 다 확인해보지는 않았다.




말이 너무 긴데, 까먹을까봐 생각나는거 다 써났다. 확인해볼 우선순위가 높은 것은 test.py든 visualize.py든 
별표를 앞에 뒤지게 박아놨으니 그거부터 보자.
'''