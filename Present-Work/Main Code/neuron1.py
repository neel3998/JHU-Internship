import numpy as np
import matplotlib.pyplot as plt
from configs import LIFneuronConfig
import math
class LIF():
    def __init__(self):
        LIFneuronConfig.__init__(self,)
        self.v = self.v_base
        self.num=0
        self.isNoise = 0
    
    def noiseTerm(self,dt):
        sigma = self.noise_amp
        y = np.random.normal(0,1,1)
        return sigma*math.sqrt(dt)*y

    def generateSpiking(self, I, t, dt, vprev):

        self.v = self.v_base
        if t >= self.initRefrac:
            noise = 0
            if self.isNoise:
                noise = self.noiseTerm(dt)

            self.v = vprev + (-vprev + I*self.R) / self.tau_m * dt +noise
            if self.v >= self.v_thresh:
                self.num+=1
                self.v += self.v_spike
                self.initRefrac = t + self.refracTime
        return self.v

def runLIFSimple(model, t_span, dt, I):
    v = np.zeros_like(t_span)

    for i, t in enumerate(t_span):
        if i==0:
            v_prev = 0
        else:
            v_prev= v[i-1]
        v[i] = model.generateSpiking(I[i], t, dt, v_prev)

    
    if model.isPlot:
        plt.plot(t_span,v, label = 'V')
        plt.plot(t_span,I, label = 'I')
        plt.title('Leaky Integrate-and-Fire')
        plt.ylabel('Membrane Potential (V) and input current(I)')
        plt.xlabel('Time (msec)')
        plt.grid()
        plt.legend(loc="upper right")
        plt.show()
    return v

def runLIFCustom(model, t_span, I,z ,ifSec = False):
    v = np.zeros_like(t_span)

    if ifSec:
        t_span = t_span*1000

    for i, t in enumerate(t_span):
        if i==0:
            v_prev = 2.44
            v[i] = 2.44
            dt = t - 0
        else:
            v_prev= v[i-1]
            dt = t - t_span[i-1]
            v[i] = model.generateSpiking(I[i], t, dt, v_prev)
        

    
    if model.isPlot:
        plt.plot(t_span,v, label = 'V')
        plt.plot(t_span,I, label = 'I')
        plt.plot(t_span,z, label = 'Z')

        plt.title('Leaky Integrate-and-Fire')
        plt.ylabel('Membrane Potential (V) and input current(I)')
        plt.xlabel('Time (msec)')
        plt.grid()
        plt.legend(loc="upper right")
        plt.show()
    return v

if __name__=='__main__':
    t_tot = 500
    dt = 0.01
    t_span = np.arange(0, t_tot+dt, dt)
    I = [1 if 200/dt <= i <= 300/dt  else 10 for i in range(len(t_span))]
    neuron = LIF()
    v = runLIFSimple(neuron, t_span, dt, I)
    