from os import error
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

from DataCollection import mapPoints
from neuron import runLIFCustom, LIF

actunum=0
finerror=100000000

textures=['3_Bumps','6_Bumps','4_Ridges','4_Waves','6_Waves','4_Blob','4_Ridges','6_Blob','Plain']
iterations=[1,2]
ans=0
for k in np.arange(1,200,0.5):
    for i in textures:
        for j in iterations:
            for l in range(1,17):
                for p in range(8):
                    map = mapPoints(i)  
                    pointmap = map.pointMap(j)
                    for key in pointmap.keys():
                        pointmap[key] = np.array(pointmap[key])
                    v = pointmap[str(l)][p][:,3]
                    v = np.abs(v*20 - 2.44*19 - 2.5)+1
                    t = pointmap[str(l)][p][:,4]
                    neuron = LIF()
                    neuron.C=k
                    v = runLIFCustom(neuron, t, v, True)
                    if i !='Plain':
                        actunum=int(i[0])
                    else:
                        actunum=0
                    num=neuron.num
                    error+=abs(num-actunum)
    if error<finerror:
        error=finerror
        ans=k

print(ans)
