import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

class Map:
    '''
    1 : Start
    2 : Lift
    3 : Testers
    4 : Line
    5 : Exit
    0 : Invalid
    It gives information, current states of the map
    '''
    def __init__(self):
        # INVD, self.invalid_flag = 0, 0
        # STRT, self.start_flag   = 1, 1
        # LIFT, self.lift_flag    = 2, 2
        # TEST, self.tester_flag  = 3, 3
        # PATH, self.path_flag    = 4, 4
        # EXIT, self.exit_flag    = 5, 5
        INVD = 0
        self.invalid_flag = INVD
        STRT = 1
        self.start_flag = STRT
        LIFT = 2
        self.lift_flag = 2
        TEST = 3
        self.tester_flag = 3
        PATH = 4
        self.path_flag = 4
        EXIT = 5
        self.exit_flag = 5
        
        self.map = [
            [INVD, INVD, TEST, TEST, TEST, TEST, TEST, INVD, INVD, TEST, TEST, TEST, TEST, TEST, INVD, INVD, INVD],
            [INVD, LIFT, PATH, PATH, PATH, PATH, PATH, PATH, LIFT, PATH, PATH, PATH, PATH, PATH, PATH, LIFT, INVD],
            [INVD, LIFT, INVD, INVD, INVD, INVD, INVD, INVD, LIFT, INVD, INVD, INVD, INVD, INVD, INVD, LIFT, INVD],
            [INVD, LIFT, TEST, TEST, TEST, TEST, TEST, INVD, LIFT, TEST, TEST, TEST, TEST, TEST, INVD, LIFT, INVD],
            [INVD, LIFT, PATH, PATH, PATH, PATH, PATH, PATH, LIFT, PATH, PATH, PATH, PATH, PATH, PATH, LIFT, INVD],
            [INVD, LIFT, INVD, INVD, INVD, INVD, INVD, INVD, LIFT, INVD, INVD, INVD, INVD, INVD, INVD, LIFT, INVD],
            [INVD, LIFT, TEST, TEST, TEST, TEST, TEST, INVD, LIFT, TEST, TEST, TEST, TEST, TEST, INVD, LIFT, INVD],
            [INVD, LIFT, PATH, PATH, PATH, PATH, PATH, PATH, LIFT, PATH, PATH, PATH, PATH, PATH, PATH, LIFT, INVD],
            [INVD, LIFT, INVD, INVD, INVD, INVD, INVD, INVD, LIFT, INVD, INVD, INVD, INVD, INVD, INVD, LIFT, INVD],
            [INVD, LIFT, TEST, TEST, TEST, TEST, TEST, INVD, LIFT, TEST, TEST, TEST, TEST, TEST, INVD, LIFT, INVD],
            [INVD, LIFT, PATH, PATH, PATH, PATH, PATH, PATH, LIFT, PATH, PATH, PATH, PATH, PATH, PATH, LIFT, INVD],
            [INVD, LIFT, INVD, INVD, INVD, INVD, INVD, INVD, LIFT, INVD, INVD, INVD, INVD, INVD, INVD, LIFT, INVD],
            [INVD, LIFT, TEST, TEST, TEST, TEST, TEST, INVD, LIFT, TEST, TEST, TEST, TEST, TEST, INVD, LIFT, INVD],
            [STRT, LIFT, PATH, PATH, PATH, PATH, PATH, PATH, LIFT, PATH, PATH, PATH, PATH, PATH, PATH, LIFT, EXIT],
        ]

        self.map.reverse()

        self.x_limit = len(self.map[0]) - 1
        self.y_limit = len(self.map) - 1
        self.agents = {}

        self.testers = {
            "a": {
                "x": [2, 3, 4, 5, 6],
                "y": [[0, 1], [3, 4], [6, 7], [9, 10], [12, 13]]
            }, 
            "b": {
                "x": [9, 10, 11, 12, 13],
                "y": [[0, 1], [3, 4], [6, 7], [9, 10], [12, 13]]
            }, 
        }

        self.lifts = {
            "a": {
                "x": [1],
                "y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            },
            "b": {
                "x": [8],
                "y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            },
            "c": {
                "x": [15],
                "y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            }
        }

        self.tester_mean = 50
        self.tester_std = 10

    def map_value(self, state):
        #print(state)
        if state is None:
            return "OUT"
        m = self.map[state[0]][state[1]]
        if m == self.invalid_flag:
            return "INVALID"
        elif m == self.start_flag:
            return "START"
        elif m == self.lift_flag:
            return "LIFT"
        elif m == self.tester_flag:
            return "TESTERS"
        elif m == self.path_flag:
            return "PATH"
        elif m == self.exit_flag:
            return "EXIT"

    def tester_status(self, tester_type):
        # 층별 테스터기 점유 상태
        counts = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]

        for agent_idx in self.agents:
            agent = self.agents[agent_idx]

            if agent.target is not None:
                if agent.target[0] == tester_type:
                    counts[agent.target[1]][agent.target[2]] += 1

        return counts
    


    def lift_status(self, lift_type):
        count = 0

        # 리프트 점유 상태
        for agent_idx in self.agents:
            agent = self.agents[agent_idx]

            s, _, _ = agent.location()
            if s == "LIFT":
                x = agent.state[1]

                if x in self.lifts[lift_type]["x"]:
                    count += 1

        return count

    def entrance_status(self):
        # 입구 점유 상태
        for agent_idx in self.agents:
            agent = self.agents[agent_idx]

            s, _, _ = agent.location()
            if s == "START":
                return True
        
        return False

    def is_occupied(self, state):
        # 셀 점유 상태
        for agent_idx in self.agents:
            agent = self.agents[agent_idx]

            if agent.state == state:
                return True
        
        return False

    def render(self, buffers=None, save=False, show=True, movie_name="movie_name"):
        def createBackground(ax):
            for floor, y in enumerate(self.map):
                for pos, value in enumerate(y):
                    p = [floor, pos]
                    v = self.map_value(p)

                    if v == "INVALID":
                        edgecolor=None
                        facecolor='white'
                    elif v == "START":
                        edgecolor = None
                        facecolor= 'blueviolet'
                    elif v == "LIFT":
                        # edgecolor = 'darkorange'
                        edgecolor = None
                        facecolor= 'navajowhite'
                    elif v == "PATH":
                        edgecolor = None
                        facecolor= 'powderblue'
                    elif v == "TESTERS":
                        # TODO : 테스터기별 컬러 다르게하기
                        if pos < 7:
                            edgecolor = 'steelblue'
                            facecolor= 'deepskyblue'
                        else:
                            edgecolor = 'darkseagreen'
                            facecolor= 'palegreen'
                    elif v == "EXIT":
                        edgecolor = None
                        facecolor= 'darkblue'
                    node = patches.Rectangle((pos, floor), 1, 1, fill=True, edgecolor=edgecolor, facecolor=facecolor)
                            
                    # node.set_width(0.5)
                    # node.set_height(0.5)
                    # node.set_xy([x,y])

                    ax.add_artist(node)

            return ax

        fig = plt.figure(figsize=((1 + len(buffers)) * self.x_limit / 3, self.y_limit / 2))

        axes = {}
        for i, buffer_type in enumerate(buffers):
            ax = fig.add_subplot(101+i+10*len(buffers), aspect='equal', autoscale_on=True)
            ax.title.set_text(buffer_type.upper())

            ax.set_xlim(0, self.x_limit + 1)
            ax.set_ylim(0, self.y_limit + 1)

            ax = createBackground(ax)

            axes[buffer_type] = ax

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        tact_time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

        if buffers is None == 0:
            
            for agent_idx in self.agents:
                agent = self.agents[agent_idx]

                p = agent.state
                if p is None:
                    p = (-100, -100)
                node = patches.Circle((p[1] + 0.5, p[0] + 0.5), radius=0.3, fill=True, facecolor="white")
                ax.add_artist(node)
                ax.annotate(str(agent.id), xy=(p[1] + 0.5, p[0] + 0.4), fontsize=8, ha="center")

            plt.show()     
        else: 
            pals = {}
            anns = {}
            for buffer_type in buffers:
                buffer = buffers[buffer_type]
                ax = axes[buffer_type]
                # Pallet 생성
                pals[buffer_type] = []
                anns[buffer_type] = []
                for agent_idx in buffer[0]:
                    agent = buffer[0][agent_idx]

                    # p = agent.state
                    # if p is None:
                    p = (-100, -100)
                    node = patches.Circle((p[1] + 0.5, p[0] + 0.5), radius=0.3, fill=True, facecolor="white")

                    pals[buffer_type].append(node)

                    ax.add_artist(node)
                    annotation = ax.annotate(str(agent.id), xy=(p[1] + 0.5, p[0] + 0.4), fontsize=8, ha="center")

                    anns[buffer_type].append(annotation)

            def init():
                """initialize animation"""
                time_text.set_text('')
                tact_time_text.set_text('')
                pallet_nodes = []
                annotations  = []

                for buffer_type in buffers:
                    pallet_nodes += pals[buffer_type]
                    annotations  += anns[buffer_type]

                # return tuple(pallet_nodes) + (time_text,) + (tact_time_text,) + tuple(annotations)
                return tuple(pallet_nodes) + tuple(annotations)

            def animate(i):
                pallet_nodes = []
                annotations  = []

                for buffer_type in buffers:
                    buffer = buffers[buffer_type]
                    b = buffer[i]

                    for pidx, agent_idx in enumerate(b):
                        agent = b[agent_idx]
                        
                        p = agent.state
                        if p is None:
                            p = (-100, -100)
                        # node = patches.Circle((p[1] + 0.5, p[0] + 0.5), radius=0.3, fill=True, facecolor="white")

                        pals[buffer_type][pidx].center = p[1] + 0.5, p[0] + 0.5
                        anns[buffer_type][pidx].set_position((p[1] + 0.5, p[0] + 0.4))
                        anns[buffer_type][pidx].xy = (p[1] + 0.5, p[0] + 0.4)

                for buffer_type in buffers:
                    pallet_nodes += pals[buffer_type]
                    annotations  += anns[buffer_type]

                return tuple(pallet_nodes) + tuple(annotations)

            interval = 0.1 * 1000
            anim = animation.FuncAnimation(fig, animate, frames=len(buffer),
                                            interval=interval, blit=True, init_func=init)

            if save == True:
                anim.save(movie_name+'.mp4')
            if show == True:
                plt.show()
