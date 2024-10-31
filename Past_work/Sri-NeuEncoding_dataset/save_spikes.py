import numpy as np
import matplotlib.pyplot as plt
import spiking_neurons as spkn
import time

numTextures = 5
numIte = 20
numTaxels = 16
dt = 1

for i in range(numTextures):
    for j in range(numIte):
        t0 = time.time()
        filename = 'texture_' + str(i+1) + '_Ite' + str(j) + '.txt'
        tactiledata = np.loadtxt(filename)
        # plt.figure(); plt.plot(tactiledata); plt.show()
        Im = [tactiledata[:,k] for k in range(numTaxels)]
        nrn = [spkn.model.izhikevich(d=8) for k in range(numTaxels)]
        simulObj = spkn.simulation(dt=1,t0=0,tf=len(tactiledata),I=Im,neurons=nrn)
        simulObj.run()
        spikeMatrix = [np.zeros(len(tactiledata)) for k in range(numTaxels)]
        for k in range(numTaxels):
            spikeMatrix[k][simulObj.spikes[k]] = 1
            # plt.figure(); plt.plot(spikeMatrix[k]); plt.show()

        newfile = 'texture_spikes_' + str(i+1) + '_Ite' + str(j) + '.txt'
        newf = open(newfile,'w')
        for z in range(len(tactiledata)):
            for k in range(numTaxels):
                newf.write(str(spikeMatrix[k][z]) + ' ')
            newf.write('\n')
        newf.close()

        tf = time.time()
        print(newfile,tf-t0)
