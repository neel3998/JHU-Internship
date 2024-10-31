import numpy as np

class LIFneuronConfig():
    def __init__(self,):
        self.R = 5 # resistance (k-Ohm)
        self.C = 3 # capacitance (u-F)

        self.v_thresh = 100
        self.v_spike = 10
        self.v_base = -1
        
        self.tau_m = self.R*self.C # time constant (msec)
        self.refracTime = 50 # refractory time (msec)
        self.initRefrac = 0

        self.noise_amp = 0.1

        self.isPlot = 1


    


