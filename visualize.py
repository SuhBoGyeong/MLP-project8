import numpy as np
import matplotlib.pyplot as plt

f = np.load('./obs_A_B/obs_all.npy')
print(f.shape)
# for i in range(500, 490, -1):
#     obs = f[i].reshape((4,98))
#     for j in range(len(obs)):
#         ob = obs[j].reshape((14,7))
#         # ob = (ob[::-1] - 1.0) / 5.0
#         # plt.imshow(ob, cmap='gray')
#         # plt.title(f'#{i+1}')
#         # plt.savefig(f'bogyeong/ob{i+1}.png')
#         if j == 3:
#             print(ob)
for i in range(999, 989, -1):
    ob = f[i]
    # ob = (ob[::-1] - 1.0) / 5.0 * 255.0
    ob = ob[::-1]
    print(ob)
    plt.imshow(ob, cmap='gray')
    plt.title(f'#{i+1}')
    plt.savefig(f'obs_A_B/ob{i+1}.png')