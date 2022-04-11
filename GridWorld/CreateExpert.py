# 利用q learning来解决迷宫问题
import math
import pickle

import numpy as np
import gym
from gym import wrappers
import time
from gridEnv import largeGridWorld,RandomGridWorld
from collections import deque
from numpy.linalg import norm

# 超参数
eps = 1.0
end_eps = 0.05
alpha = 0.1
gamma = 0.9
decay = 0.995
rList = deque(maxlen=100)
episodes = 5000
running_reward = None

def build_q_table(n_states, n_actions):
    q_table = np.zeros((n_states, n_actions))
    return q_table


def choose_action(state, q_table, eps=None, flag='test'):
    if flag == 'test':
        action = np.argmax(q_table[state, :])
    else:
        if np.random.uniform() < eps:
            action = np.random.choice(len(q_table[state, :]))
        else:
            action = np.argmax(q_table[state, :])
    return action


def display(action_list, env):
    env.reset()
    for action in action_list:
        env.render()
        env.step(action)
        time.sleep(0.2)


# Reinforcement Learning
def train(env,
          eps = eps,
          end_eps = end_eps,
          alpha = alpha,
          gamma = gamma,
          rList = rList,
          episodes = episodes,
          running_reward = running_reward,
          q_table = None,
          flag = 'train'):
    count = 0
    record = deque(maxlen=100)
    for i in range(1, episodes + 1):
        episode_time = time.time()
        state = env.reset()
        record.clear()
        rAll = 0
        for j in range(100):

            action = choose_action(state,q_table,eps,flag=flag)

            next_state, reward, done, _ = env.step(action)

            # 更新Q表
            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
            record.append(action)
            rAll += reward
            state = next_state

            if done == True:
                # if i>1000:
                # display(record,env)
                # count += 1
                break
        rList.append(rAll)
        # print(rList)
        # running_reward = rAll if running_reward is None else running_reward * 0.99 + rAll * 0.01
        # print('\rEpisode {}\t Average Score: {:.2f}'.format(i, np.mean(rList)), end="")
        if i % 10 == 0:
            eps = max(end_eps, eps*decay)
        if i % 1000 == 0:
            print('\rEpisode {}\tAverage Score:{:.2f}'.format(i, np.mean(rList)))
            # print(rList)
            # print(count,eps)
        # if i == episodes:
        #     display(record,env)
    # print(q_table)
    return q_table

def validate(env,
         q_table,
         flag='test',
         episodes = 5
        ):
    for i in range(episodes):
        state = env.reset()
        rAll = 0
        for j in range(100):
            action = choose_action(state,q_table,eps=None,flag=flag)
            time.sleep(0.1)
            env.render()
            next_state, reward, done, _ = env.step(action)
            # print('\r',reward,end='')
            state = next_state
            rAll += reward
            if done:
                # print('Episode {}/{}: reward is {}'.format(i,j,rAll))
                break

def validate_learnedQ(env,filename):
    file = open(filename,'rb')
    Q = pickle.load(file)
    file.close()
    validate(env,Q)

def difference_Expert_Inverse(env):
    # 用来比较
    print("Reinforcement Learning")
    validate_learnedQ(env, 'ExpertQ')
    time.sleep(3)
    print("Inverse Reinfocement Learning")
    validate_learnedQ(env, 'learnedQ')

if __name__ == '__main__':
    seed = 5
    env = RandomGridWorld(seed)
    q_table = build_q_table(env.observation_space.n, env.action_space.n)
    q_table = train(env,q_table=q_table)


    # filename = "ExpertQ_"+str(seed)
    # outfile = open(filename, 'wb')
    # pickle.dump(q_table, outfile)
    # outfile.close()

    validate(env,q_table=q_table)