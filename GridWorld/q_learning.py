#利用q learning来解决迷宫问题
import math
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import gym
import time
from gridEnv import largeGridWorld,RandomGridWorld
from collections import deque
from numpy.linalg import norm

#超参数
eps = 1.0
end_eps = 0.05
alpha = 0.1
gamma = 0.9
rList = deque(maxlen=100)
episodes = 1000
running_reward = None

#Inverse Reinforcement learning
def initialize_Q(Q,states, actions):
    for state in range(states):
        for action in range(actions):
            Q[state][action] = 0
    return Q
#注：如果state提取的feature是一个数，是否在算reward时会导致计算不准确
def getFeature(state):
    return 1./(1+math.exp(-state))

def getFeatureExpectation(Q,N=1000):
    observationSum = 0
    for i in range(N):
        state = env.reset()
        done = False
        cnt = 0

        for j in range(99):
            #选择max的action
            action = choose_action(state,Q)
            state, reward, done, _ = env.step(action)
            feature = getFeature(state)
            observationSum += (gamma**cnt)*feature

            if done:
                break
            cnt += 1
    featureExpectation = observationSum/N

    return featureExpectation

def irl_play_one_game(weight, Q, eps = 0.5):
    state = env.reset()
    done = False
    cnt = 0
    total_reward = 0

    for i in range(99):
        cnt += 1
        action = choose_action(state,Q,eps,'train')
        next_state, reward, done, _ = env.step(action)

        #reward获得
        feature = getFeature(state)
        reward = np.dot(weight, feature)
        total_reward += reward

        #存疑
        # if done and cnt < 200:
        #     reward = -1

        #选下一状态最大动作
        next_action = choose_action(next_state,Q)
        next_max_Q = Q[next_state][next_action]

        Q[state][action] += alpha*(reward + gamma*next_max_Q - Q[state][action])

        state = next_state
        action = next_action

        if done:
            break
    return total_reward, cnt

def irl_play_many_games(env,weight, N=1000):
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = build_q_table(n_states, n_actions)
    length = []
    reward = []
    eps = 1

    for n in range(N):
        eps *= 0.995

        episode_reward, episode_length = irl_play_one_game(weight,Q,eps)
        length.append(episode_length)
        reward.append(episode_reward)
    print("Avg Length %d"%(np.average(length)))
    return length, reward, Q

def irl_train(env,
              Q,
              expertQ,
              N = 10,
              termination = 0.000002,
              seed = None
              ):
    weight = []
    featureExpecation = []
    featureExpectationBar = []
    learnedQ = []
    margin = []
    avgEpisodeLength = []

    n_state = env.observation_space.n
    n_action = env.action_space.n
    expertExpectation = getFeatureExpectation(expertQ)

    # print(getFeatureExpectation(initialize_Q(Q,n_state,n_action)))
    # print(getFeatureExpectation(expertQ))
    for i in range(N):
        print('Iteration : ',i)
        if i == 0:
            initialQ = initialize_Q(Q,n_state,n_action)
            featureExpecation.append(getFeatureExpectation(initialQ))
            learnedQ.append(initialQ)
            weight.append(0)
            margin.append(1)
        else:
            if i == 1:
                featureExpectationBar.append(featureExpecation[i-1])
                weight.append(expertExpectation-featureExpecation[i-1])
                margin.append(math.sqrt((expertExpectation-featureExpecation[i-1])**2))

            else:
                A = featureExpectationBar[i-2]
                B = featureExpecation[i-1]-A
                C = expertExpectation - featureExpectationBar[i-2]
                featureExpectationBar.append(A+(np.dot(B,C)/np.dot(B,B))*(B))

                weight.append(expertExpectation-featureExpectationBar[i-1])
                margin.append(math.sqrt((expertExpectation-featureExpectationBar[i-1])**2))
                # print(featureExpecation,'\n',featureExpectationBar,'\n',weight,'\n',margin)

            if (margin[i] <= termination):
                break

            episode_lengths, episode_rewards, learnedQ_i = irl_play_many_games(env,weight[i],5000)
            learnedQ.append(learnedQ_i)
            avgEpisodeLength.append(episode_lengths)
            # print(featureExpecation)
            featureExpecation.append(getFeatureExpectation(learnedQ[i]))
            # print(featureExpecation)
    print("export trained IRL model...")
    filename = "learnedQ_"+str(seed)
    outfile = open(filename, 'wb')
    pickle.dump(learnedQ[-1],outfile)
    outfile.close()

    return learnedQ[-1]
    
def build_q_table(n_states,n_actions):
    q_table = np.zeros((n_states,n_actions))
    return q_table

def choose_action(state, q_table,eps=None,flag='test'):
    if flag=='test':
        action = np.argmax(q_table[state, :])
    else:
        if np.random.uniform() < eps:
            action = np.random.choice(len(q_table[state,:]))
        else:
            action = np.argmax(q_table[state, :])
    return action

def display(action_list,env):
    env.reset()
    for action in action_list:
        env.render()
        env.step(action)
        time.sleep(0.2)

def displayGraphDif(expertQ,learnedQ):
    expert_action_list = []
    learned_action_list = []
    for state in range(expertQ.shape[0]):
        expert_action_list.append(np.argmax(expertQ[state,:]))
        learned_action_list.append(np.argmax(learnedQ[state,:]))

    #展示1
    # x = range(100)
    # plt.scatter(x,expert_action_list,c='red',marker='o',label='Expert')
    # plt.scatter(x,learned_action_list,c='blue',marker='^',label='Learned')
    # plt.legend()
    # plt.show()
    #展示2
    ax = plt.subplot(111)

    for i in range(100):
        if expert_action_list[i] != learned_action_list[i]:
            coordinate_x = i % 10
            coordinate_y = int((i - coordinate_x) / 10)

            x = np.linspace(coordinate_x, coordinate_x + 1, 10)
            y1 = coordinate_y
            y2 = coordinate_y + 1
            ax.fill_between(x, y1, y2, facecolor='red')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    ax.xaxis.set_major_locator(MultipleLocator(1))  # 设置y主坐标间隔 1
    ax.yaxis.set_major_locator(MultipleLocator(1))  # 设置y主坐标间隔 1
    ax.xaxis.grid(True, which='major')  # major,color='black'
    ax.yaxis.grid(True, which='major')  # major,color='black'
    plt.show()

if __name__ == '__main__':
    seed = 5
    env = RandomGridWorld(seed)
    q_table = build_q_table(env.observation_space.n,env.action_space.n)

    #载入专家输入
    filename = 'ExpertQ_'+ str(seed)
    expertFile = open(filename,'rb')
    expertQ = pickle.load(expertFile)
    expertFile.close()

    learnedQ = irl_train(env,q_table,expertQ,seed=seed)
    # print(learnedQ)
    displayGraphDif(expertQ,learnedQ)
    # q_table = train(env,q_table=q_table)
    # # print(q_table)
    # validate(env,q_table=q_table)