import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time

Orientation = 4
Grid_count = 5*5

class Grid(object):
    def __init__(self,  x:int = None,
                        y:int = None,
                        type:int = 0,
                        reward:float = 0.0,
                        value:float = 0.0):
        '''
        :param x:x轴坐标
        :param y: y轴坐标
        :param type: 格子类型（0代表能进，1代表不能进,2代表终点）
        :param reward: 奖励
        :param value: V值
        '''
        self.x = x
        self.y = y
        self.type = type
        self.reward =reward
        self.value = value
        self.name = None
        self._update_name()

    #显示已在第几个格子内
    def _update_name(self):
        self.name = "X{0}-Y{0}".format(self.x,self.y)

    def __str__(self):
        return "name:{4}, x{0}, y{1}, type:{2}, value{3}".format(self.x,
                                                                 self.y,
                                                                 self.type,
                                                                 self.value,
                                                                 self.name)

class GridMatrix(object):
    def __init__(self,  n_width:int,
                        n_height:int,
                        default_type:int = 0,
                        default_reward: float = 0.0,
                        default_value: float = 0.0):
        '''
        :param n_width: 格子有多少列
        :param n_height: 格子有多少行
        :param default_type: 同上
        :param default_reward: 同上
        :param default_value: 同上
        '''
        self.grids = None
        self.n_height = n_height
        self.n_width = n_width
        self.len = n_width * n_height
        self.default_reward = default_reward
        self.default_value = default_value
        self.default_type = default_type

        self.reset()

    def reset(self):
        self.grids = []
        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(x,
                                       y,
                                       self.default_type,
                                       self.default_reward,
                                       self.default_value))
    def get_grid(self,x,y=None):
        xx, yy = None, None
        #获得坐标（int则使用x,y tuple类型则只使用x）
        if isinstance(x, int):
            xx, yy = x, y
        elif isinstance(x, tuple):
            xx, yy = x[0], x[1]
        assert(xx>=0 and yy>=0 and xx<self.n_width and yy<self.n_height),\
            "坐标越界"
        #获得格子的在一维上的位置
        index = yy * self.n_width + xx
        return self.grids[index]

    def set_reward(self, x, y, reward):
        grid = self.get_grid(x,y)
        if grid is not None:
            grid.reward = reward
        else:
            raise("grid doesn't exist")

    def set_value(self, x, y, value):
        grid = self.get_grid(x,y)
        if grid is not None:
            grid.value = value
        else:
            raise("grid doesn't exist")

    def set_type(self, x, y, type):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.type = type
        else:
            raise ("grid doesn't exist")

    def get_reward(self, x, y):
        grid = self.get_grid(x,y)
        if grid is None:
            return None
        return grid.reward

    def get_value(self, x, y):
        grid = self.get_grid(x,y)
        if grid is None:
            return None
        return grid.value

    def get_type(self, x, y):
        grid = self.get_grid(x,y)
        if grid is None:
            return None
        return grid.type

class GirdWordEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    def __init__(self,  n_width:int=10,
                        n_height:int=7,
                        u_size = 40,
                        default_reward:float=0.0,
                        default_type = 0,
                        windy=False):
        '''
        :param n_width: 宽度默认为10
        :param n_height: 高度默认为7
        :param u_size: 每个格子的大小（pixels）
        :param default_reward: 略
        :param default_type: 略
        :param windy: 用于展示有风的情况
        '''

        self.u_size = u_size
        self.n_width = n_width
        self.n_height = n_height
        self.width = u_size * n_width   #实际宽度
        self.height = u_size * n_height #实际高度
        self.default_reward = default_reward
        self.default_type = default_type
        self._adjust_size()

        self.grids = GridMatrix(n_width = self.n_width,
                                n_height = self.n_height,
                                default_reward=self.default_reward,
                                default_type=self.default_type,
                                default_value=0.0)
        self.reward = 0
        self.action = None
        self.windy = windy

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.n_height * self.n_width)

        self.ends = [(7,3)]     #终止格子坐标
        self.start = (0,3)      #起始格子坐标
        self.types = []         #特殊格子坐标（3，2，1）代表坐标（3，2）的type为1

        self.rewards = []       #特殊奖励的格子

        self.refresh_setting()
        self.viewer = None      #图形接口对象
        self.seed()
        self.reset()

    def _adjust_size(self):
        pass

    def seed(self, seed=None):
        #产生随机数
        self.np_random, seed = seeding.np_random(seed)

#需要改进的地方：reward应该随动作而得到的
    def step(self, action):
        #判断action是否在space里
        assert self.action_space.contains(action),\
            "%r (%s) invalid" %(action,type(action))
        self.action = action
        #self.state是否有问题(没有事先说明state的值)：理解，self.state不必在__init__可以在其它函数中定义
        old_x, old_y = self._state_to_xy(self.state)
        new_x, new_y = old_x, old_y
        #有风的情况，一般不用
        if self.windy:
            if new_x in [3,4,5,8]:
                new_y += 1
            elif new_x in [6,7]:
                new_y += 2
        '''
        暂时只有4个方向
        0:left
        1:right
        2:up
        3:down
        '''
        if action == 0: new_x -= 1
        elif action == 1: new_x += 1
        elif action == 2: new_y += 1
        elif action == 3: new_y -= 1

        # #边界
        # if new_x < 0: new_x = 0
        # if new_x >= self.n_width: new_x = self.n_width-1
        # if new_y < 0: new_y = 0
        # if new_y >= self.n_height: new_y = self.n_height-1

        #障碍物
        if (new_x<0 or new_x >= self.n_width or new_y<0 or new_y >= self.n_width) or self.grids.get_type(new_x,new_y) == 1 :
            new_x, new_y = old_x, old_y
            self.reward = -1
        else:
            self.reward = self.grids.get_reward(new_x, new_y)

        done = self._is_end_state(new_x,new_y)
        self.state = self._xy_to_state(new_x,new_y)

        info = {"x":new_x, "y":new_y, 'grids':self.grids}
        return self.state, self.reward, done, info

    def _state_to_xy(self, s):
        x = s % self.n_width
        y = int((s-x)/self.n_width)
        return x,y

    def _xy_to_state(self, x, y = None):
        if isinstance(x, int):
            assert(isinstance(y, int)), "imcomplete Position info"
            return x + self.n_width * y
        elif isinstance(x, tuple):
            return x[0] + self.n_width * x[1]
        return -1

    def refresh_setting(self):
        #用于修改特殊格子
        for x,y,r in self.rewards:
            self.grids.set_reward(x,y,r)
        for x,y,t in self.types:
            self.grids.set_type(x,y,t)

    def reset(self):
        self.state = self._xy_to_state(self.start)
        return self.state

    def _is_end_state(self, x, y=None):
        if y is not None:
            xx,yy = x,y
        #y非空且x为int则说明输入的是state
        elif isinstance(x, int):
            xx, yy = self._state_to_xy(x)
        #否则输入的是tuple
        else:
            assert(isinstance(x, tuple)),"incomplete coordinate values"
            xx, yy = self._state_to_xy(x)
        for end in self.ends:
            if xx == end[0] and yy == end[1]:
                return True
        return False

    '''  
       在Viewer里绘制一个几何图像的步骤如下：
       the following steps just tells how to render an shape in the environment.
       1. 建立该对象需要的数据本身
       2. 使用rendering提供的方法返回一个geom对象
       3. 对geom对象进行一些对象颜色、线宽、线型、变换属性的设置（有些对象提供一些个
          性化的方法来设置属性，具体请参考继承自这些Geom的对象），这其中有一个重要的
          属性就是变换属性，
          该属性负责对对象在屏幕中的位置、渲染、缩放进行渲染。如果某对象
          在呈现时可能发生上述变化，则应建立关于该对象的变换属性。该属性是一个
          Transform对象，而一个Transform对象，包括translate、rotate和scale
          三个属性，每个属性都由以np.array对象描述的矩阵决定。
       4. 将新建立的geom对象添加至viewer的绘制对象列表里，如果在屏幕上只出现一次，
          将其加入到add_onegeom(）列表中，如果需要多次渲染，则将其加入add_geom()
       5. 在渲染整个viewer之前，对有需要的geom的参数进行修改，修改主要基于该对象
          的Transform对象
       6. 调用Viewer的render()方法进行绘制
   '''
    ''' 绘制水平竖直格子线，由于设置了格子之间的间隙，可不用此段代码
    for i in range(self.n_width+1):
        line = rendering.Line(start = (i*u_size, 0), 
                              end =(i*u_size, u_size*self.n_height))
        line.set_color(0.5,0,0)
        self.viewer.add_geom(line)
    for i in range(self.n_height):
        line = rendering.Line(start = (0, i*u_size),
                              end = (u_size*self.n_width, i*u_size))
        line.set_color(0,0,1)
        self.viewer.add_geom(line)
    '''

    def render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        zero = (0,0)
        u_size = self.u_size
        #设置格子的间隔
        m = 2

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)

            #绘制格子
            for x in range(self.n_width):
                for y in range(self.n_height):
                    v = [(x*u_size + m, y*u_size + m),
                         ((x+1)*u_size-m, y*u_size + m),
                         ((x+1)*u_size-m, (y+1)*u_size - m),
                         (x*u_size + m, (y+1)*u_size - m)]

                    rect = rendering.FilledPolygon(v)
                    t = self.grids.get_reward(x,y)/10

                    if t < 1:
                        rect.set_color(0.9-t, 0.9+t, 0.9+t)
                    elif t == 1:
                        rect.set_color(0.3, 0.5+t, 0.3)
                    else:
                        rect.set_color(0.9, 0.9, 0.9)
                    self.viewer.add_geom(rect)
                    #绘制边框
                    v_outline = [(x*u_size + m, y*u_size + m),
                                 ((x+1)*u_size - m, y*u_size + m),
                                 ((x+1)*u_size - m, (y+1)*u_size - m),
                                 (x*u_size + m, (y+1)*u_size - m)]
                    #False可能有问题
                    outline = rendering.make_polyline(v_outline)
                    outline.set_linewidth(3)

                    if self._is_end_state(x,y):
                        outline.set_color(0.9,0.9,0)
                        self.viewer.add_geom(outline)
                    if self.start[0] == x and self.start[1] == y:
                        outline.set_color(0.5, 0.5, 0.8)
                        self.viewer.add_geom(outline)
                    if self.grids.get_type(x,y) == 1:
                        rect.set_color(0.3,0.3,0.3)
                    else:
                        pass
            #可能有问题
            self.agent = rendering.make_circle(u_size/4, 30, True)
            self.agent.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)

        #更新个体位置
        x,y = self._state_to_xy(self.state)
        self.agent_trans.set_translation((x+0.5)*u_size, (y+0.5)*u_size)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

def largeGridWorld():
    env = GirdWordEnv(n_width=10,
                      n_height=10,
                      u_size=40,
                      default_reward=0,
                      default_type=0,
                      windy=False)
    env.start = (0,9)
    env.ends = [(5,4)]
    env.types = [(4, 2, 1), (4, 3, 1), (4, 4, 1), (4, 5, 1), (4, 6, 1), (4, 7, 1),
                 (1, 7, 1), (2, 7, 1), (3, 7, 1), (4, 7, 1), (6, 7, 1), (7, 7, 1),
                 (8, 7, 1)]
    env.rewards = [(3, 2, -1), (3, 6, -1), (5, 2, -1), (6, 2, -1), (8, 3, -1),
                   (8, 4, -1), (5, 4, 2), (6, 4, -1), (5, 5, -1), (6, 5, -1)]
    env.refresh_setting()
    return env

def createObstacle(obstacle,s,n_width):
    x = s % n_width
    y = int((s - x) / n_width)

    obstacle.append((x,y,1))

def createTrap(Trap,s,n_width):
    x = s % n_width
    y = int((s - x) / n_width)

    Trap.append((x,y,-1))

def RandomGridWorld(seed):
    np.random.seed(seed)

    env = GirdWordEnv(n_width=10,
                      n_height=10,
                      u_size=40,
                      default_reward=0,
                      default_type=0,
                      windy=False)
    env.start=(0,9)
    env.ends = [(9,0)]

    Obstacle = []
    Trap = []
    reward = [(env.ends[0][0],env.ends[0][1],2)]
    #Create Obstacle
    for i in range(20):
        state = np.random.randint(0,10*10)

        if state == env.start[0] + 10*env.start[1] or state == env.ends[0][0] + 10*env.ends[0][1]:
            continue
        else:
            createObstacle(Obstacle,state,10)

    #Create Trap
    for i in range(10):
        state = np.random.randint(0, 10 * 10)

        if state == env.start[0] + 10*env.start[1] or state == env.ends[0][0] + 10*env.ends[0][1]:
            continue
        else:
            createTrap(Trap, state, 10)

    env.rewards = reward + Trap
    env.types = Obstacle

    env.refresh_setting()

    return env

if __name__ == "__main__":
    env = RandomGridWorld(4)
    env.reset()
    nfs = env.observation_space
    nfa = env.action_space

    print("nfs:{}, nfa:{}".format(nfs,nfa))

    env.render()
    time.sleep(60)
    for _ in range(10):
        time.sleep(0.5)
        env.step(1)
        env.render()
    env.close()

