import numpy as np
import matplotlib.pyplot as plt

# f = np.load('./obs_A_B/obs_all.npy')
f1 = np.load('./steps/every_step_1.npy')
f2 = np.load('./steps/every_step_2.npy')
a1 = np.load('./steps/actions_1.npy')
a2 = np.load('./steps/actions_2.npy')

for i in range(len(f1)-1, -1, -1):
    ob1 = f1[i]
    ob1 = (ob1 - 1.0) / 5.0 * 255.0
    plt.imshow(ob1, cmap='gray')
    plt.title(f'#{i+1} action {a1[i]}')
    # plt.savefig(f'obs_A_B/ob{i+1}.png')
    plt.savefig(f'steps/t1-step{i+1}.png')
    ob2 = f1[i]
    ob2 = (ob2 - 1.0) / 5.0 * 255.0
    plt.imshow(ob2, cmap='gray')
    plt.title(f'#{i+1} action {a2[i]}')
    # plt.savefig(f'obs_A_B/ob{i+1}.png')
    plt.savefig(f'steps/t2-step{i+1}.png')


# for i in range(len(f)-1, len(f)-10, -1):
#     step = f[i]
#     ob = ob[::-1]
#     print(ob)
#     plt.imshow(ob, cmap='gray')
#     plt.title(f'#{i+1}')
#     plt.savefig(f'obs_A_B/ob{i+1}.png')