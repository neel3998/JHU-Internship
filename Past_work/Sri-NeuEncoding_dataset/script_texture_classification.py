# -*- coding: utf-8 -*-
'''
#-------------------------------------------------------------------------------
# NATIONAL UNIVERSITY OF SINGAPORE - NUS
# SINGAPORE INSTITUTE FOR NEUROTECHNOLOGY - SINAPSE
# Singapore
# URL: http://www.sinapseinstitute.org
#-------------------------------------------------------------------------------
# Neuromorphic Engineering and Robotics Group
#-------------------------------------------------------------------------------
'''
import numpy as np
import scipy.stats as stats
import spiking_neurons as spkn
import time
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets #kNN classifier
from sklearn.svm import SVC #SVM classifier
#-------------------------------------------------------------------------------
numTaxels = 16
numTextures = 5
numIte = 20
#-------------------------------------------------------------------------------
spikev = [[] for k in range(numTaxels)]
spike_features = [np.zeros((numTextures*numIte,2)) for k in range(numTaxels)]
spike_targets = [np.zeros(numTextures*numIte) for k in range(numTaxels)]
#-------------------------------------------------------------------------------
#load the data
t0 = time.time()
for i in range(numTextures):
    for j in range(numIte):
        filename = 'texture_spikes_' + str(i+1) + '_Ite' + str(j) + '.txt'
        spikedata = np.loadtxt(filename)
        for k in range(numTaxels):
            spikev[k].append(spikedata[:,k])
tf = time.time()
print('total time: ' + str(tf-t0) + ' seconds')
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#classification of each individual taxel separately
for k in range(numTaxels):
    #extract features from each spike train
    targetId = 1
    for i in range(numTextures*numIte):
        #measure the spiking (firing) rate
        average_spiking_rate = np.sum(spikev[k][i]) / (len(spikev[k][i])/1000)
        #retrieve the inter-spike interval (ISI) array
        isiv = np.where(spikev[k][i] == 1)[0]
        isiv = np.diff(isiv)
        #measure the coefficient of variation of the ISI
        cv_isi = np.std(isiv,ddof=1) / np.mean(isiv)
        #store the features
        spike_features[k][i,0] = average_spiking_rate
        spike_features[k][i,1] = cv_isi
        #store the target
        spike_targets[k][i] = targetId
        if (i+1)%numIte == 0:
            targetId += 1
    #normalize the features
    spike_features[k] = stats.zscore(spike_features[k],axis=0)
    #classification
    clfObj = SVC(gamma='auto')
    ret = spkn.classification.LOOCV(clfObj,spike_features[k],spike_targets[k])
    print(ret[0])
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
plt.figure()
for k in range(numTextures):
    plt.scatter(spike_features[10][k*numIte:k*numIte+numIte,0],spike_features[10][k*numIte:k*numIte+numIte,1])
plt.show()
#-------------------------------------------------------------------------------
