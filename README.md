# EEGanalysisPython
PythonImplementation

q1.py:
Using EEG data, developing a prediction model for eyes open and closed task. Data is raw at 256Hz,4 channels epoched and labelled. 
Various steps were used:
1) Data was cleaned, filtered and then ICA was done. 
2) Alpha power was extracted. 
3) KNN classifier was used with 10 fold cross validation to develop a classification model. 

q2.py:
1)Time -frequency decompostion of eeg data. 
2) Temporal analysis of the dominant freq and see how it evolves over the tasks. 

q3.py:
1) Charactorizing signal components into various artifacts like eye blinks, muscle activity, eye movement, eyes open/closed. 
2) Performing ICA and then analyzing its time-frequency activity to look for the standard signatures of the above artifacts. 
