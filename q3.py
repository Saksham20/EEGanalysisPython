import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
from sklearn.decomposition import FastICA
from scipy import signal,stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

q2_eeg_data=np.load('q3_eeg_data.npy')
n_samples=np.shape(q2_eeg_data)[0]
fs=256
n_channels=np.shape(q2_eeg_data)[1]
#test
#initializing--
q2_eeg_data_filtered=q2_eeg_data
q2_eeg_data_filtered_ica=q2_eeg_data
q2_eeg_data_filtered_ica_fft=q2_eeg_data

transformer=FastICA(max_iter=1000)
sos = signal.butter(10, 1, 'hp', fs=256, output='sos')

for i in range(n_channels):
    q2_eeg_data_filtered[:,i] = signal.sosfilt(sos, q2_eeg_data[:,i])

q2_eeg_data_filtered_ica=transformer.fit_transform(q2_eeg_data_filtered)
for i in range(n_channels):
    q2_eeg_data_filtered_ica_fft[:,i]=fft(q2_eeg_data_filtered_ica[:,i])


# plotting channels, ica, spectrograms:
fig1,ax1=plt.subplots(nrows=4,ncols=1)
fig2,ax2=plt.subplots(nrows=4,ncols=1)
fig3,ax3=plt.subplots(nrows=4,ncols=1)
fig4,ax4=plt.subplots(nrows=4,ncols=1)

for i in range(n_channels):
    ax1[i].plot(np.arange(n_samples)*1000/256,q2_eeg_data_filtered[:,i])
    ax1[i].set_xlabel('Time(ms)')
    ax1[i].set_title('Filtered Signals')

    ax4[i].plot(np.arange(n_samples)*1000/256,q2_eeg_data[:,i])
    ax4[i].set_xlabel('Time(ms)')
    ax4[i].set_title('Raw_signals')

    ax2[i].plot(q2_eeg_data_filtered_ica[:,i])
    ax2[i].set_xlabel('Time(ms)')
    ax2[i].set_title('ICA components')

    ax3[i].plot(np.arange(n_samples//2)*fs/n_samples,(2/n_samples)*np.abs(q2_eeg_data_filtered_ica_fft[0:n_samples//2,i]))
    ax3[i].set_xlabel('Freq(Hz)')
    ax3[i].set_title('Spectogram of IC components')

plt.show()
