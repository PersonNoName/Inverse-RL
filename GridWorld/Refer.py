import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
import pickle

ax = plt.subplot(111)  # 注意:一般都在ax中设置,不再plot中设置

filename1 = 'ExpertQ_1'
filename2 = 'learnedQ_1'

file = open(filename1,'rb')
ExpertQ = pickle.load(file)
file.close()

file = open(filename2,'rb')
LearnedQ = pickle.load(file)
file.close()

expert_action_list = []
learned_action_list = []
for state in range(ExpertQ.shape[0]):
    expert_action_list.append(np.argmax(ExpertQ[state,:]))
    learned_action_list.append(np.argmax(LearnedQ[state,:]))

for i in range(100):
    if expert_action_list[i] != learned_action_list[i]:
        coordinate_x = i % 10
        coordinate_y = int((i - coordinate_x) / 10)

        x = np.linspace(coordinate_x,coordinate_x+1,10)
        y1 = coordinate_y
        y2 = coordinate_y+1
        ax.fill_between(x,y1,y2,facecolor='red')
plt.xlim(0,10)
plt.ylim(0,10)
ax.xaxis.set_major_locator(MultipleLocator(1))  # 设置y主坐标间隔 1
ax.yaxis.set_major_locator(MultipleLocator(1))  # 设置y主坐标间隔 1
ax.xaxis.grid(True, which='major')  # major,color='black'
ax.yaxis.grid(True, which='major')  # major,color='black'
plt.show()
# for i in range(3):
#     x1 = np.linspace(30 * i, 30 * i + 30, 10)
#     ax.fill_between(x1, y1, y2, facecolor='green')
# #
# for i in range(6):
#     x2 = np.linspace(30 * i, 30 * i + 30, 10)
#     ax.fill_between(x2, y3, y4, facecolor='red')
#
# #
# plt.xlim(0, 900)
# plt.ylim(0, 800)
#
# ax.xaxis.set_major_locator(MultipleLocator(30))  # 设置y主坐标间隔 1
# ax.yaxis.set_major_locator(MultipleLocator(20))  # 设置y主坐标间隔 1
# ax.xaxis.grid(True, which='major')  # major,color='black'
# ax.yaxis.grid(True, which='major')  # major,color='black'
# plt.show()