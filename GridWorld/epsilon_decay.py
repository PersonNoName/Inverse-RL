import numpy as np
import math
from matplotlib import pyplot

eps1 = 1
eps2 = 1
eps3 = 1
end = 0.05

y_eps1 = []
y_eps2 = []
y_eps3 = []
episodes = 10000
for i in range(1,episodes):
    eps1 *= 0.9995
    eps2 = 1/np.sqrt(float(i))
    eps3 *= 0.995
    y_eps1.append(eps1)
    y_eps2.append(eps2)
    y_eps3.append(eps3)

pyplot.plot(range(1,episodes),y_eps1,label='0.9995')
pyplot.plot(range(1,episodes),y_eps2,label='1/sqrt(i)')
pyplot.plot(range(1,episodes),y_eps3,label='0.995')
pyplot.title('Epsilon Decay')
pyplot.xlabel('step')
pyplot.ylabel('epsilon')
pyplot.legend()
pyplot.show()