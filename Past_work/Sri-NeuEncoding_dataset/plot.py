import numpy as np
import matplotlib.pyplot as plt

s = np.loadtxt('texture_5_Ite0.txt')
ss = np.loadtxt('texture_spikes_5_Ite0.txt')

plt.figure()
plt.plot(s[:,10])

plt.figure()
plt.plot(ss[:,10])

plt.show()
