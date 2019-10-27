import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft
from sklearn.decomposition import FastICA
from scipy import signal,stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

q1test=np.load('q1_test_data.npy')
q1labels=np.load('q1_training_labels.npy')
q1training=np.load('q1_training_data.npy')

[lab1,lab2]=np.unique(q1labels)
trainshape=np.shape(q1training)
testshape=np.shape(q1test)
goodfeats=np.nan*np.ones([trainshape[0],trainshape[1]])
goodfeats_test=np.nan*np.ones([testshape[0],testshape[1]])
goodlabels=np.nan*np.ones([trainshape[0]])
N=trainshape[2]


transformer=FastICA(max_iter=1000)
sos = signal.butter(10, 1, 'hp', fs=256, output='sos')

# feature extraction (average alpha power)
for i in range(trainshape[0]):
    reshapeDat1=q1training[i,:,:].reshape(4,256)
    reshapeDat_training=reshapeDat1

    if ~np.any(np.isnan(reshapeDat_training.reshape(1,256*4))):# since there are NaNs in the dataset
        #updating labels with NaNs removed
        if q1labels[i]=='eyes_closed':
            goodlabels[i]=0
        elif q1labels[i]=='eyes_open':
            goodlabels[i]=1

        #filtering step: 1:100Hz
        for t in range(trainshape[1]):
            reshapeDat_training[t,:] = signal.sosfilt(sos, reshapeDat1[t,:])

        reshapeDat_training=np.transpose(reshapeDat_training)
        icaDat=np.transpose(transformer.fit_transform(reshapeDat_training))
        reshapeDat_training=np.transpose(reshapeDat_training)

        for j in range(trainshape[1]):
            eofftica=fft(icaDat[j,:])
            # alpha power avg:
            goodfeats[i,j]=np.mean(np.power(((2/N)*np.abs(eofftica[8:13])),2))

goodfeats=goodfeats[~np.isnan(goodfeats)].reshape(-1,4)
goodlabels=goodlabels[~np.isnan(goodlabels)]
n_ngh=20
training_accuracy = np.zeros([n_ngh])
test_accuracy = np.zeros([n_ngh])
neighbors_settings=range(1, n_ngh+1)
no_crossvalds=10
max_test_accuracy=np.zeros(no_crossvals)
max_neighbors=np.zeros(no_crossvals)

# training and testing of classification using K nearest neighbors:-----
for crosvall in range(no_crossvalds):
    X_train, X_test, y_train, y_test = train_test_split(goodfeats, goodlabels, test_size=0.1, random_state=crosvall)
    past_accuracy=0

    for n_neighbors in neighbors_settings:
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
        training_accuracy[n_neighbors-1]=clf.score(X_train, y_train)
        test_accuracy[n_neighbors-1]=clf.score(X_test, y_test)
        if test_accuracy[n_neighbors-1]>past_accuracy:
          clf_max=clf
          clf_max.fit(X_train, y_train)

        past_accuracy=test_accuracy[n_neighbors-1]

    max_test_accuracy[crosvall+1]=np.amax(test_accuracy)
    max_neighbors[crosvall+1]=np.argmax(test_accuracy)

# avg cross validated test accuracy:
print('avg cross validated test accuracy',np.mean(max_test_accuracy))
#max neighbors(optimal k nearest neighbors parameter)
print('max neighbors',stats.mode(max_neighbors))

#general plot showing accuracy for training and testing while varying the model parameter
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()



# predictions for the given test set:---------------------------------------
given_test_predictions=np.zeros()

for i in range(testshape[0]):
    reshapeDat2=q1test[i,:,:].reshape(4,256)
    reshapeDat_test=reshapeDat2

    if ~np.any(np.isnan(reshapeDat_test.reshape(1,256*4))):
        for t in range(testshape[1]):
          reshapeDat_test[t,:] = signal.sosfilt(sos, reshapeDat2[t,:])

        reshapeDat_test=np.transpose(reshapeDat_test)
        icaDat_test=np.transpose(transformer.fit_transform(reshapeDat_test))
        reshapeDat_test=np.transpose(reshapeDat_test)

        for j in range(testshape[1]):
              eofftica=fft(icaDat_test[j,:])
              # alpha power avg:
              goodfeats_test[i,j]=np.mean(np.power(((2/N)*np.abs(eofftica[8:13])),2))

goodfeats_test=goodfeats_test[~np.isnan(goodfeats_test)].reshape(-1,4)
test_predictions=clf_max.predict(goodfeats_test)
print('eyesCLosed Predictions:{},eyesOpen Predictions:{}'.format(sum(test_predictions==0),sum(test_predictions==1)))
