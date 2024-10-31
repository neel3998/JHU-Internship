import numpy as np
from neuron.neuron import izhikevich

class neuronCluster():
    def __init__(self):
        self.neuron1 = izhikevich()
        self.neuron2 = izhikevich()
        self.neuron3 = izhikevich()

        self.gain1 = 50
        self.gain2 = 200
        self.gain3 = 1000

        self.dt = 1.39 #milli-seconds

        self.v1, self.v2, self.v3 = self.reset()

    def reset(self,):
        self.v1 = []
        self.v2 = []
        self.v3 = []

        return self.v1,self.v2,self.v3
    def _scaleGain(self,I):
        I1 = np.abs(I*self.gain1 - 2.44*(self.gain1-1) - 2.5)+1
        I2 = np.abs(I*self.gain2 - 2.44*(self.gain2-1) - 2.5)+1
        I3 = np.abs(I*self.gain3 - 2.44*(self.gain3-1) - 2.5)+1

        return I1, I2, I3

    def runCluster(self, I):
        self.reset()
        I1,I2,I3 = self._scaleGain(np.array(I))

        for i in range(len(I)):
            if i==0:
                self.v1.append(self.neuron1.c)
                self.v2.append(self.neuron2.c)
                self.v3.append(self.neuron3.c)

            else:
                self.v1.append(self.neuron1.generateSpiking(I1[i], self.dt))
                self.v2.append(self.neuron2.generateSpiking(I2[i], self.dt))
                self.v3.append(self.neuron3.generateSpiking(I3[i], self.dt))
        
        return np.array([self.v1, self.v2, self.v3])


