import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

from DataCollection import mapPoints
from neuron.neuron import runNeuronCustom, LIF, izhikevich

from DataCollection import mapPoints
from neuron1 import runLIFCustom, LIF

map = mapPoints('3_Bumps')    
pointmap = map.pointMap(1) # Dataset number

for key in pointmap.keys():
    for j in range(len(pointmap[key])):
        pointmap[key][j] = np.array(pointmap[key][j])
    pointmap[key] = np.array(pointmap[key])
    
def stackPlot(pointmap, col):
    figure, axis = plt.subplots(4, 4)

    for i in range(1):
        for j in range(1):
            # print(type(pointmap[str(i+1)][j]))
            axis[i//4, i%4].plot(np.abs(pointmap[str(i+1)][j][:,col]*20 - 2.44*19 - 2.5)+1)
            axis[i//4, i%4].plot(pointmap[str(i+1)][j][:,col-1])

            axis[i//4, i%4].set_title('Sensor: '+str(i+1))
            axis[i//4, i%4].grid()
    figure.tight_layout(pad=0.1)
    plt.show()


def Plot3DALL():
    ax = plt.axes(projection ='3d')
    for i in range(1,17):
        X = pointmap[str(i)][:,0]
        Y = pointmap[str(i)][:,1]
        Z = pointmap[str(i)][:,2]
        ax.plot3D(X, Y, Z,marker = 'o', markersize = 1)
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

    ax.set_title('3D line plot geeks for geeks')
    plt.show()

stackPlot(pointmap, 2)
# print(len(pointmap['1']))
# v = pointmap['6'][4][:,3]
# z = pointmap['6'][4][:,2]

# gain = 300
# t = pointmap['6'][4][:,4]
# for i in range(100,gain):
#     print(i)
#     v_scaled = np.abs(v*i - 2.44*(i-1) - 2.5)+1
#     # # exit()
#     neuron = izhikevich()
#     runNeuronCustom(neuron, t, v_scaled,z, True)
    # input()


