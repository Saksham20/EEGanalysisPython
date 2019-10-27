
# interaxon data exploration file

import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
from sklearn.decomposition import FastICA
from scipy import signal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

q2_eeg_data=np.load('q2_eeg_data.npy')
N=np.shape(q2_eeg_data)[0]
fs=256

fftdat=fft(q2_eeg_data)
# finding the dominat freq and plotting the freq response:
alpha_range=np.abs(fftdat[int(8/(fs*(1/N))):int(12/(fs*(1/N)))+1])
sm_window=100
alpha_range_sm=np.convolve(alpha_range,np.ones(sm_window)/sm_window,mode='same')
dom_freq=np.argmax(alpha_range_sm)
dom_freq=np.round(8+dom_freq*fs/N,1)
print(dom_freq)

fig,ax=plt.subplots(nrows=2,ncols=1)
fig2,ax2=plt.subplots()

ax[0].plot(np.array(range(int(1.5/(fs*(1/N))),N//2))*(fs/N),np.convolve((2/N)*np.abs(fftdat[int(1.5/(fs*(1/N))):N//2]),np.ones(sm_window)/sm_window,mode='same'))
ax[0].set_xlabel('freq(Hz)')
ax[0].set_ylabel('magnitude')


f, t, Sxx = signal.spectrogram(q2_eeg_data, fs)

index = (np.abs(f-dom_freq)).argmin()
index_base=(np.abs(f-5)).argmin()
print('nearestfreq',f[index],index_base,np.shape(Sxx)[0])
#Ans1. Max freq statistics over time:
ax[1].plot(t,Sxx[index,:])
ax2.pcolormesh(t, f[index_base:np.shape(Sxx)[0]], Sxx[index_base:np.shape(Sxx)[0],:])
ax[1].set_ylabel('Frequency {}Hz'.format(f[index]))
ax[1].set_xlabel('Time [sec]')


## computing the max freq over time: Part3
fig3,ax3=plt.subplots()
fig3,ax4=plt.subplots(nrows=4,ncols=1)

dom_freq_timewise=np.ones(np.size(t))
alphaid1=(np.abs(f-7)).argmin()
alphaid2=(np.abs(f-100)).argmin()
for times in range(np.size(t)):
    alpha_range_sm=np.convolve(Sxx[alphaid1:alphaid2+1,times],np.ones(sm_window)/sm_window,mode='same')
    dom_freq_temp=np.argmax(alpha_range_sm)
    dom_freq_timewise[times]=np.round(8+dom_freq_temp*fs/N,1)

ax3.plot((np.arange(np.size(t))+1)*(1000/256),dom_freq_timewise)
ax3.set_xlabel('time(ms)')
ax3.set_ylabel('magnitude')
ax3.set_label('Change in dominant freq over time')

ax4[0].plot(Sxx[alphaid1:alphaid2+1,(np.size(t)-1)*1//4])
ax4[1].plot(Sxx[alphaid1:alphaid2+1,(np.size(t)-1)*2//4])
ax4[2].plot(Sxx[alphaid1:alphaid2+1,(np.size(t)-1)*3//4])
ax4[3].plot(Sxx[alphaid1:alphaid2+1,(np.size(t)-1)*4//4])

plt.show()
