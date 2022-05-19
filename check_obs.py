import os
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import sys 
import pandas as pd
import cv2 

np.set_printoptions(threshold=sys.maxsize)

total_obs1 = np.load('./observation1.npy', allow_pickle=True)
total_obs2 = np.load('./observation2.npy', allow_pickle=True)


for i in range(500, 490, -1):
    #print(i)
    obs = total_obs1[i].reshape((4,98))
    # obs1, obs2, obs3, obs4 = obs
    # obs1 = obs1.reshape((14,7))
    # obs2 = obs2.reshape((14,7))
    # obs3 = obs3.reshape((14,7))
    # obs4 = obs4.reshape((14,7))
    for j in range(len(obs)):
        ob = obs[j].reshape((14, 7))
        ob = ((ob[::-1] + 1.0) / 5.0 * 255.0).astype(int)
        plt.imshow(ob, cmap='gray')
        plt.title(f'#{i+1}')
        plt.savefig('./observations/'+f'{i+1}-{j+1}.jpg')

    
    print('----------------------')


quit()


    # def render(self, buffers=None, save=False, show=True, movie_name="movie_name"):
    #     def createBackground(ax):
    #         for floor, y in enumerate(self.map):
    #             for pos, value in enumerate(y):
    #                 p = [floor, pos]
    #                 v = self.map_value(p)

    #                 if v == "INVALID":
    #                     edgecolor=None
    #                     facecolor='white'
    #                 elif v == "START":
    #                     edgecolor = None
    #                     facecolor= 'blueviolet'
    #                 elif v == "LIFT":
    #                     # edgecolor = 'darkorange'
    #                     edgecolor = None
    #                     facecolor= 'navajowhite'
    #                 elif v == "PATH":
    #                     edgecolor = None
    #                     facecolor= 'powderblue'
    #                 elif v == "TESTERS":
    #                     # TODO : 테스터기별 컬러 다르게하기
    #                     if pos < 7:
    #                         edgecolor = 'steelblue'
    #                         facecolor= 'deepskyblue'
    #                     else:
    #                         edgecolor = 'darkseagreen'
    #                         facecolor= 'palegreen'
    #                 elif v == "EXIT":
    #                     edgecolor = None
    #                     facecolor= 'darkblue'
    #                 node = patches.Rectangle((pos, floor), 1, 1, fill=True, edgecolor=edgecolor, facecolor=facecolor)
                            
    #                 # node.set_width(0.5)
    #                 # node.set_height(0.5)
    #                 # node.set_xy([x,y])

    #                 ax.add_artist(node)

    #         return ax

    #     fig = plt.figure(figsize=((1 + len(buffers)) * self.x_limit / 3, self.y_limit / 2))

    #     axes = {}
    #     for i, buffer_type in enumerate(buffers):
    #         ax = fig.add_subplot(101+i+10*len(buffers), aspect='equal', autoscale_on=True)
    #         ax.title.set_text(buffer_type.upper())

    #         ax.set_xlim(0, self.x_limit + 1)
    #         ax.set_ylim(0, self.y_limit + 1)

    #         ax = createBackground(ax)

    #         axes[buffer_type] = ax

    #     time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    #     tact_time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

    #     if buffers is None == 0:
            
    #         for agent_idx in self.agents:
    #             agent = self.agents[agent_idx]

    #             p = agent.state
    #             if p is None:
    #                 p = (-100, -100)
    #             node = patches.Circle((p[1] + 0.5, p[0] + 0.5), radius=0.3, fill=True, facecolor="white")
    #             ax.add_artist(node)
    #             ax.annotate(str(agent.id), xy=(p[1] + 0.5, p[0] + 0.4), fontsize=8, ha="center")

    #         plt.show()     
    #     else: 
    #         pals = {}
    #         anns = {}
    #         for buffer_type in buffers:
    #             buffer = buffers[buffer_type]
    #             ax = axes[buffer_type]
    #             # Pallet 생성
    #             pals[buffer_type] = []
    #             anns[buffer_type] = []
    #             for agent_idx in buffer[0]:
    #                 agent = buffer[0][agent_idx]

    #                 # p = agent.state
    #                 # if p is None:
    #                 p = (-100, -100)
    #                 node = patches.Circle((p[1] + 0.5, p[0] + 0.5), radius=0.3, fill=True, facecolor="white")

    #                 pals[buffer_type].append(node)

    #                 ax.add_artist(node)
    #                 annotation = ax.annotate(str(agent.id), xy=(p[1] + 0.5, p[0] + 0.4), fontsize=8, ha="center")

    #                 anns[buffer_type].append(annotation)

    #         def init():
    #             """initialize animation"""
    #             time_text.set_text('')
    #             tact_time_text.set_text('')
    #             pallet_nodes = []
    #             annotations  = []

    #             for buffer_type in buffers:
    #                 pallet_nodes += pals[buffer_type]
    #                 annotations  += anns[buffer_type]

    #             # return tuple(pallet_nodes) + (time_text,) + (tact_time_text,) + tuple(annotations)
    #             return tuple(pallet_nodes) + tuple(annotations)

    #         def animate(i):
    #             pallet_nodes = []
    #             annotations  = []

    #             for buffer_type in buffers:
    #                 buffer = buffers[buffer_type]
    #                 b = buffer[i]

    #                 for pidx, agent_idx in enumerate(b):
    #                     agent = b[agent_idx]
                        
    #                     p = agent.state
    #                     if p is None:
    #                         p = (-100, -100)
    #                     # node = patches.Circle((p[1] + 0.5, p[0] + 0.5), radius=0.3, fill=True, facecolor="white")

    #                     pals[buffer_type][pidx].center = p[1] + 0.5, p[0] + 0.5
    #                     anns[buffer_type][pidx].set_position((p[1] + 0.5, p[0] + 0.4))
    #                     anns[buffer_type][pidx].xy = (p[1] + 0.5, p[0] + 0.4)

    #             for buffer_type in buffers:
    #                 pallet_nodes += pals[buffer_type]
    #                 annotations  += anns[buffer_type]

    #             return tuple(pallet_nodes) + tuple(annotations)

    #         interval = 0.1 * 1000
    #         anim = animation.FuncAnimation(fig, animate, frames=len(buffer),
    #                                         interval=interval, blit=True, init_func=init)

    #         if save == True:
    #             anim.save(movie_name+'.mp4')
    #         if show == True:
    #             plt.show()








    

