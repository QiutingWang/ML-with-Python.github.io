###Improving features for Classification###

#Smoothing Over Time:instead of averaging over all time,we can get a local average
#it removes short-term noise,while retaining the general pattern

#Calculating a rolling window statistics#use rolling()
# Audio is a Pandas DataFrame
print(audio.shape)
# (n_times, n_audio_files)
# Smooth our data by taking the rolling mean in a window of 50 samples
window_size = 50
windowed = audio.rolling(window=window_size)  #window parameter tells us how many timepoints to include in each window,the larger the window, the smoother the result will be.
audio_smooth = windowed.mean()
#Calculating the auditory envelope
audio_rectified = audio.apply(np.abs)  #we calculate the abs value of each timepoint-->rectification, we ensure that all time point>0
audio_envelope = audio_rectified.rolling(50).mean()  #calculate the rolling mean to smooth the signal

#Feature engineering the envelop,find a better feature
# Calculate several features of the envelope, one per sound
envelope_mean = np.mean(audio_envelope, axis=0)
envelope_std = np.std(audio_envelope, axis=0)
envelope_max = np.max(audio_envelope, axis=0)
# Create our training data for a classifier,using column stack() to put all the variables we care together~
X = np.column_stack([envelope_mean, envelope_std, envelope_max]) #combine them in a way
y = labels.reshape([-1, 1]) #preparing for sklearn
#use cross validation for classification,using cross_val_score for split,fit,and scoring data
from sklearn.model_selection import cross_val_score
model = LinearSVC()
scores = cross_val_score(model, X, y, cv=3)
print(scores)
#computing the tempogram
# Import librosa and calculate the tempo of a 1-D sound array
import librosa as lr  #librosa used to extract the tempogram from an audio array,the moment by moment tempo of the sound,calculate the features of the classifier
audio_tempo = lr.beat.tempo(audio, sr=sfreq,
                            hop_length=2**6, aggregate=None)
# Column stack all the features to create the X and y arrays
X = np.column_stack([means, stds, maxs, tempos_mean, tempos_std, tempos_max])
y = labels.reshape([-1, 1])
# Fit the model and score on testing data
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))



##Special case:spectrogram:special changes over time##
##The key part:the fourier transform(FFT):1.quickly-changing and slowly-changing things 2.we can describe the relative presence of fast- and slow- moving components

#SFFT-short time fourier transform:
#choosing a window size and shape;At a timepoint,calculated the FFT for this window;slide the window over by one;aggregate the result
#Calculating STFT:librosa,several parameters we can tweak(eg:window size),convert the output to decibels which normalizing the average values of all frequencies,use specshow() to visualize.
 
#Import the functions we'll use for the STFT
from librosa.core import stft, amplitude_to_db
from librosa.display import specshow
#Calculate our STFT,decide the window size use STFT
HOP_LENGTH = 2**4
SIZE_WINDOW = 2**7
audio_spec = stft(audio, hop_length=HOP_LENGTH, n_fft=SIZE_WINDOW)
#Convert into decibels for visualization
spec_db = amplitude_to_db(audio_spec) #amplitude to db function, ensure all values are positive,real numbers
#Visualize
specshow(spec_db, sr=sfreq, x_axis='time',
         y_axis='hz', hop_length=HOP_LENGTH)

#each time series have an unique spectral pattern to it,by analyzing spectrogram.eg:special banwidth and spectral centroids找到核心变量
#Calculating spectral features:
#Calculate the spectral centroid and bandwidth for the spectrogram
bandwidths = lr.feature.spectral_bandwidth(S=spec)[0]
centroids = lr.feature.spectral_centroid(S=spec)[0]
#Display these features on top of the spectrogram
ax = specshow(spec, x_axis='time', y_axis='hz', hop_length=HOP_LENGTH)
ax.plot(times_spec, centroids)
ax.fill_between(times_spec, centroids - bandwidths / 2,
                centroids + bandwidths / 2, alpha=0.5)

#combining spectual and temporal features in a classifier,initializing the features
centroids_all = []
bandwidths_all = []
#put them into for loop
for spec in spectrograms:
    bandwidths = lr.feature.spectral_bandwidth(S=lr.db_to_amplitude(spec))
    centroids = lr.feature.spectral_centroid(S=lr.db_to_amplitude(spec))
    # Calculate the mean spectral bandwidth
    bandwidths_all.append(np.mean(bandwidths))
    # Calculate the mean spectral centroid
    centroids_all.append(np.mean(centroids))

# Create our X matrix,combine each of the features mentioned before into a single matrix for our classifier
X = np.column_stack([means, stds, maxs, tempo_mean,
                     tempo_max, tempo_std, bandwidths_all, centroids_all]) #we include more complex features into our model,improve the model performance.
                   
