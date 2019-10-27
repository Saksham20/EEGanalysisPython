
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
from sklearn.decomposition import FastICA
from scipy import signal
from sklearn.neighbors import KNeighborsClassifier


q1test=np.load('q1_test_data.npy')
q1labels=np.load('q1_training_labels.npy')
q1training=np.load('q1_training_data.npy')

[lab1,lab2]=np.unique(q1labels)
trainshape=np.shape(q1training)
goodfeats=np.nan*np.ones([trainshape[0],trainshape[1]])
goodlabels=np.nan*np.ones([trainshape[0]])
N=trainshape[2]


transformer=FastICA(max_iter=1000)
sos = signal.butter(10, 1, 'hp', fs=256, output='sos')


for i in range(trainshape[0]):
    reshapeDat1=q1training[i,:,:].reshape(4,256)
    reshapeDat=reshapeDat1
    if ~np.any(np.isnan(reshapeDat.reshape(1,256*4))):
    #filtering step: 1:100Hz
        if q1labels[i]=='eyes_closed':
            goodlabels[i]=0
        elif q1labels[i]=='eyes_open':
            goodlabels[i]=1

        for t in range(trainshape[1]):
            reshapeDat[t,:] = signal.sosfilt(sos, reshapeDat1[t,:])

        reshapeDat=np.transpose(reshapeDat)
        icaDat=np.transpose(transformer.fit_transform(reshapeDat))
        reshapeDat=np.transpose(reshapeDat)

        for j in range(trainshape[1]):
            eofftica=fft(icaDat[j,:])
            # alpha power avg:
            goodfeats[i,j]=np.mean(np.power(((2/N)*np.abs(eofftica[8:13])),2))
