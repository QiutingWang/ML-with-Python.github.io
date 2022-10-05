##Time series and ML primer##
#The components of time series data:1.an array of # represent the data itself;2.Another array contains a timestramp for each datapoint.
#Reading time series data in with pandas#
from ast import Or
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data.csv')
data.head() #each datapoint has a corresponding time point
#plot the time series data#
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6)) #create a figure and axis
data.plot('date', 'close', ax=ax)
ax.set(title="AAPL daily closing price")
#machine learning is to find the parterns of data when the data is too large or too complex to be processed by human;Also,ML can predict the furture
#Time series data always change over time,ML is a useful pattern to utilize. We use pipeline in the context of timeseries data
#Feature extraction(what kinds of special features leverage a signal changing over time)-->Model fitting-->Prediction & Validation

#to visualize your data:
# Using matplotlib
fig, ax = plt.subplots()
ax.plot(...)
# Using pandas
fig, ax = plt.subplots()
df.plot(..., ax=ax)
#Sklearn particular structure of data:(samples,features)
array.shape
(10,)
array.reshape([-1, 1]).shape
(10, 1)
#-1 fill that axis with remaining values
#Basic machine learnings:
# Import a support vector classifier, using SVM.
from sklearn.svm import LinearSVC
# Instantiate this model
model = LinearSVC()
# Fit the model on some data
model.fit(X, y)
# There is one coefficient per input feature
model.coef_
# Generate predictions
predictions = model.predict(X_test)


import librosa as lr
# `load` accepts a path to an audio file
audio, sfreq = lr.load('data/heartbeat-sounds/proc/files/murmur__201101051104.wav')
print(sfreq)
#create a time array: the sampling rate is fixed and no data points are lost
#create an array of indices,and divide by the sampling fresquency
indices = np.arange(0, len(audio)) #generate a range of indices from 0 to the number of datapoints in your audio file
time = indices / sfreq
#Alternitively:calculate the final timepoint of your audio data
final_time = (len(audio) - 1) / sfreq 
time = np.linspace(0, final_time, sfreq) #using the linspace function to generate evenly-spaced numbers between 0 and final timepoint

#using new york stock exchange dataset,detect any patterns in historical records that allows us to predict the value of companies in the future.
#take a look at the raw data
data = pd.read_csv('path/to/data.csv')
data.columns
data.head()
#investigate the type of the data of each column by accessing .dtypes
df['date'].dtypes
#convert a column to time series,using to_datetime() function
df['date'] = pd.to_datetime(df['date'])
df['date']
0   2017-01-01
1   2017-01-02
2   2017-01-03
Name: date, dtype: datetime64[ns]

##using heart beat disorder sounds dataset,inspecting time series classification data
import librosa as lr
from glob import glob #using glob to return a list of the .wav files
# List all the wav files in the folder
audio_files = glob(data_dir + '/*.wav')
# Read in the first audio file, create the time array
audio, sfreq = lr.load('./files/murmur__201108222238.wav')
time = np.arange(0, len(audio)) / sfreq
# Plot audio over time
fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
plt.show()

##inspecting regression data,using new york stock price dataset
# Read in the data
data = pd.read_csv('prices.csv', index_col=0)
# Convert the index of the DataFrame to datetime
data.index =pd.to_datetime(data.index)
print(data.head())
# Loop through each column, plot its values over time
fig, ax = plt.subplots()
for column in data:
    data[column].plot(ax=ax, label=column)
ax.legend()
plt.show()
