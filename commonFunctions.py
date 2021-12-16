import numpy as np
import pandas as pd# type: ignore
import librosa # type: ignore
#import pyAudioAnalysis # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.patches import Polygon # type: ignore

import warnings
warnings.filterwarnings('ignore')

subfolderName = 'izbrane_15/'
showResults = True

def displayFeature(data, name):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(data, y_axis=name, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title=name)
    plt.show()

# 199 audio features from librosa
def musicFeatureExtraction(filePath):
    
    y, sr = librosa.load(filePath, res_type='kaiser_fast')
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr) # returns: Normalized energy for each chroma bin at each frame.
    chroma_averaged = np.mean(chroma.T,axis=0)
    print("Chroma", len(chroma_averaged))
    
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr) # returns: Mel spectrogram
    melspectrogram_averaged = np.mean(melspectrogram.T,axis=0)
    print("melspectrogram", len(melspectrogram_averaged))
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40) # returns: MFCC sequence
    mfccs_averaged = np.mean(mfcc.T,axis=0)
    print("mfcc", len(mfccs_averaged))
    
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # returns: each row of spectral contrast values corresponds to a given octave-based frequency
    spectral_contrast_averaged = np.mean(spectral_contrast.T,axis=0)
    print("spectral_contrast", len(spectral_contrast_averaged))
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)  # returns: p’th-order spectral bandwidth.
    spectral_bandwidth_averaged = np.mean(spectral_bandwidth.T,axis=0)
    print("spectral_bandwidth", len(spectral_bandwidth_averaged))
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)  # returns: roll-off frequency.
    spectral_rolloff_averaged = np.mean(spectral_rolloff.T,axis=0)
    print("spectral_rolloff", len(spectral_rolloff_averaged))
    
    poly_features = librosa.feature.poly_features(y=y, sr=sr) # returns: coefficients of fitting an nth-order polynomial to the columns of a spectrogram.
    poly_features_averaged = np.mean(poly_features.T,axis=0)
    print("poly_features", len(poly_features_averaged))
    
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr) # returns: Tonal centroid features for each frame.
    tonnetz_averaged = np.mean(tonnetz.T,axis=0)
    print("tonnetz", len(tonnetz_averaged))
    
    zcr = librosa.feature.zero_crossing_rate(y=y)  # returns: zero-crossing rate of an audio time series.
    zcr_averaged = [np.mean(zcr)]
    print("zcr", len(zcr_averaged))
    
    bpm = librosa.beat.tempo(y=y, sr=sr)  # returns: bpm
    print("bpm", len(bpm))
    
    featureNumPy = mfccs_averaged
    featureNumPy = np.concatenate((featureNumPy,chroma_averaged))
    featureNumPy = np.concatenate((featureNumPy,melspectrogram_averaged))
    featureNumPy = np.concatenate((featureNumPy,spectral_contrast_averaged))
    featureNumPy = np.concatenate((featureNumPy,tonnetz_averaged))
    
    # dodane potem
    featureNumPy = np.concatenate((featureNumPy,poly_features_averaged))
    featureNumPy = np.concatenate((featureNumPy,spectral_rolloff_averaged))
    featureNumPy = np.concatenate((featureNumPy,spectral_bandwidth_averaged))
    featureNumPy = np.concatenate((featureNumPy,zcr_averaged))
    featureNumPy = np.concatenate((featureNumPy,bpm))
    
    #print('size:', featureNumPy.size)
    
    return featureNumPy


def featureTranslationOrg(featureIndex):
    featDict = [ 'Chroma B', 'Chroma A#', 'Chroma A', 'Chroma G#', 'Chroma G', 'Chroma F#', 'Chroma F', 'Chroma E', 'Chroma D#', 'Chroma D', 'Chroma C#', 'Chroma C', 'MEL spectrogram 1', 'MEL spectrogram 2', 'MEL spectrogram 3', 'MEL spectrogram 4', 'MEL spectrogram 5', 'MEL spectrogram 6', 'MEL spectrogram 7', 'MEL spectrogram 8', 'MEL spectrogram 9', 'MEL spectrogram 10', 'MEL spectrogram 11', 'MEL spectrogram 12', 'MEL spectrogram 13', 'MEL spectrogram 14', 'MEL spectrogram 15', 'MEL spectrogram 16', 'MEL spectrogram 17', 'MEL spectrogram 18', 'MEL spectrogram 19', 'MEL spectrogram 20', 'MEL spectrogram 21', 'MEL spectrogram 22', 'MEL spectrogram 23', 'MEL spectrogram 24', 'MEL spectrogram 25', 'MEL spectrogram 26', 'MEL spectrogram 27', 'MEL spectrogram 28', 'MEL spectrogram 29', 'MEL spectrogram 30', 'MEL spectrogram 31', 'MEL spectrogram 32', 'MEL spectrogram 33', 'MEL spectrogram 34', 'MEL spectrogram 35', 'MEL spectrogram 36', 'MEL spectrogram 37', 'MEL spectrogram 38', 'MEL spectrogram 39', 'MEL spectrogram 40', 'MEL spectrogram 41', 'MEL spectrogram 42', 'MEL spectrogram 43', 'MEL spectrogram 44', 'MEL spectrogram 45', 'MEL spectrogram 46', 'MEL spectrogram 47', 'MEL spectrogram 48', 'MEL spectrogram 49', 'MEL spectrogram 50', 'MEL spectrogram 51', 'MEL spectrogram 52', 'MEL spectrogram 53', 'MEL spectrogram 54', 'MEL spectrogram 55', 'MEL spectrogram 56', 'MEL spectrogram 57', 'MEL spectrogram 58', 'MEL spectrogram 59', 'MEL spectrogram 60', 'MEL spectrogram 61', 'MEL spectrogram 62', 'MEL spectrogram 63', 'MEL spectrogram 64', 'MEL spectrogram 65', 'MEL spectrogram 66', 'MEL spectrogram 67', 'MEL spectrogram 68', 'MEL spectrogram 69', 'MEL spectrogram 70', 'MEL spectrogram 71', 'MEL spectrogram 72', 'MEL spectrogram 73', 'MEL spectrogram 74', 'MEL spectrogram 75', 'MEL spectrogram 76', 'MEL spectrogram 77', 'MEL spectrogram 78', 'MEL spectrogram 79', 'MEL spectrogram 80', 'MEL spectrogram 81', 'MEL spectrogram 82', 'MEL spectrogram 83', 'MEL spectrogram 84', 'MEL spectrogram 85', 'MEL spectrogram 86', 'MEL spectrogram 87', 'MEL spectrogram 88', 'MEL spectrogram 89', 'MEL spectrogram 90', 'MEL spectrogram 91', 'MEL spectrogram 92', 'MEL spectrogram 93', 'MEL spectrogram 94', 'MEL spectrogram 95', 'MEL spectrogram 96', 'MEL spectrogram 97', 'MEL spectrogram 98', 'MEL spectrogram 99', 'MEL spectrogram 100', 'MEL spectrogram 101', 'MEL spectrogram 102', 'MEL spectrogram 103', 'MEL spectrogram 104', 'MEL spectrogram 105', 'MEL spectrogram 106', 'MEL spectrogram 107', 'MEL spectrogram 108', 'MEL spectrogram 109', 'MEL spectrogram 110', 'MEL spectrogram 111', 'MEL spectrogram 112', 'MEL spectrogram 113', 'MEL spectrogram 114', 'MEL spectrogram 115', 'MEL spectrogram 116', 'MEL spectrogram 117', 'MEL spectrogram 118', 'MEL spectrogram 119', 'MEL spectrogram 120', 'MEL spectrogram 121', 'MEL spectrogram 122', 'MEL spectrogram 123', 'MEL spectrogram 124', 'MEL spectrogram 125', 'MEL spectrogram 126', 'MEL spectrogram 127', 'MEL spectrogram 128', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5', 'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10', 'MFCC 11', 'MFCC 12', 'MFCC 13', 'MFCC 14', 'MFCC 15', 'MFCC 16', 'MFCC 17', 'MFCC 18', 'MFCC 19', 'MFCC 20', 'MFCC 21', 'MFCC 22', 'MFCC 23', 'MFCC 24', 'MFCC 25', 'MFCC 26', 'MFCC 27', 'MFCC 28', 'MFCC 29', 'MFCC 30', 'MFCC 31', 'MFCC 32', 'MFCC 33', 'MFCC 34', 'MFCC 35', 'MFCC 36', 'MFCC 37', 'MFCC 38', 'MFCC 39', 'MFCC 40', 'Spectral Contrast 1', 'Spectral Contrast 2', 'Spectral Contrast 3', 'Spectral Contrast 4', 'Spectral Contrast 5', 'Spectral Contrast 6', 'Spectral Contrast 7', 'Spectral Bandwidth', 'Spectral Rolloff', 'Poly order 0', 'Poly order 0-1', 'Tonnetz M3y', 'Tonnetz M3x', 'Tonnetz m3y', 'Tonnetz m3x', 'Tonnetz 5y', 'Tonnetz 5x', 'Zero Crossing Rate', 'BPM' ]
    translated = []
    for index in featureIndex:
        translated.append(featDict[index])
    return translated


def featuresTranslation(featureIndexs):
    translated = []
    for index in featureIndexs:
        translated.append(featureTranslation(index))
    return translated

def featureTranslation(featureIndex):
    featDict = [ 'Chroma B', 'Chroma A#', 'Chroma A', 'Chroma G#', 'Chroma G', 'Chroma F#', 'Chroma F', 'Chroma E', 'Chroma D#', 'Chroma D', 'Chroma C#', 'Chroma C', 'MEL spectrogram 1', 'MEL spectrogram 2', 'MEL spectrogram 3', 'MEL spectrogram 4', 'MEL spectrogram 5', 'MEL spectrogram 6', 'MEL spectrogram 7', 'MEL spectrogram 8', 'MEL spectrogram 9', 'MEL spectrogram 10', 'MEL spectrogram 11', 'MEL spectrogram 12', 'MEL spectrogram 13', 'MEL spectrogram 14', 'MEL spectrogram 15', 'MEL spectrogram 16', 'MEL spectrogram 17', 'MEL spectrogram 18', 'MEL spectrogram 19', 'MEL spectrogram 20', 'MEL spectrogram 21', 'MEL spectrogram 22', 'MEL spectrogram 23', 'MEL spectrogram 24', 'MEL spectrogram 25', 'MEL spectrogram 26', 'MEL spectrogram 27', 'MEL spectrogram 28', 'MEL spectrogram 29', 'MEL spectrogram 30', 'MEL spectrogram 31', 'MEL spectrogram 32', 'MEL spectrogram 33', 'MEL spectrogram 34', 'MEL spectrogram 35', 'MEL spectrogram 36', 'MEL spectrogram 37', 'MEL spectrogram 38', 'MEL spectrogram 39', 'MEL spectrogram 40', 'MEL spectrogram 41', 'MEL spectrogram 42', 'MEL spectrogram 43', 'MEL spectrogram 44', 'MEL spectrogram 45', 'MEL spectrogram 46', 'MEL spectrogram 47', 'MEL spectrogram 48', 'MEL spectrogram 49', 'MEL spectrogram 50', 'MEL spectrogram 51', 'MEL spectrogram 52', 'MEL spectrogram 53', 'MEL spectrogram 54', 'MEL spectrogram 55', 'MEL spectrogram 56', 'MEL spectrogram 57', 'MEL spectrogram 58', 'MEL spectrogram 59', 'MEL spectrogram 60', 'MEL spectrogram 61', 'MEL spectrogram 62', 'MEL spectrogram 63', 'MEL spectrogram 64', 'MEL spectrogram 65', 'MEL spectrogram 66', 'MEL spectrogram 67', 'MEL spectrogram 68', 'MEL spectrogram 69', 'MEL spectrogram 70', 'MEL spectrogram 71', 'MEL spectrogram 72', 'MEL spectrogram 73', 'MEL spectrogram 74', 'MEL spectrogram 75', 'MEL spectrogram 76', 'MEL spectrogram 77', 'MEL spectrogram 78', 'MEL spectrogram 79', 'MEL spectrogram 80', 'MEL spectrogram 81', 'MEL spectrogram 82', 'MEL spectrogram 83', 'MEL spectrogram 84', 'MEL spectrogram 85', 'MEL spectrogram 86', 'MEL spectrogram 87', 'MEL spectrogram 88', 'MEL spectrogram 89', 'MEL spectrogram 90', 'MEL spectrogram 91', 'MEL spectrogram 92', 'MEL spectrogram 93', 'MEL spectrogram 94', 'MEL spectrogram 95', 'MEL spectrogram 96', 'MEL spectrogram 97', 'MEL spectrogram 98', 'MEL spectrogram 99', 'MEL spectrogram 100', 'MEL spectrogram 101', 'MEL spectrogram 102', 'MEL spectrogram 103', 'MEL spectrogram 104', 'MEL spectrogram 105', 'MEL spectrogram 106', 'MEL spectrogram 107', 'MEL spectrogram 108', 'MEL spectrogram 109', 'MEL spectrogram 110', 'MEL spectrogram 111', 'MEL spectrogram 112', 'MEL spectrogram 113', 'MEL spectrogram 114', 'MEL spectrogram 115', 'MEL spectrogram 116', 'MEL spectrogram 117', 'MEL spectrogram 118', 'MEL spectrogram 119', 'MEL spectrogram 120', 'MEL spectrogram 121', 'MEL spectrogram 122', 'MEL spectrogram 123', 'MEL spectrogram 124', 'MEL spectrogram 125', 'MEL spectrogram 126', 'MEL spectrogram 127', 'MEL spectrogram 128', 'MFCC 1', 'MFCC 2', 'MFCC 3', 'MFCC 4', 'MFCC 5', 'MFCC 6', 'MFCC 7', 'MFCC 8', 'MFCC 9', 'MFCC 10', 'MFCC 11', 'MFCC 12', 'MFCC 13', 'MFCC 14', 'MFCC 15', 'MFCC 16', 'MFCC 17', 'MFCC 18', 'MFCC 19', 'MFCC 20', 'MFCC 21', 'MFCC 22', 'MFCC 23', 'MFCC 24', 'MFCC 25', 'MFCC 26', 'MFCC 27', 'MFCC 28', 'MFCC 29', 'MFCC 30', 'MFCC 31', 'MFCC 32', 'MFCC 33', 'MFCC 34', 'MFCC 35', 'MFCC 36', 'MFCC 37', 'MFCC 38', 'MFCC 39', 'MFCC 40', 'Spectral Contrast 1', 'Spectral Contrast 2', 'Spectral Contrast 3', 'Spectral Contrast 4', 'Spectral Contrast 5', 'Spectral Contrast 6', 'Spectral Contrast 7', 'Spectral Bandwidth', 'Spectral Rolloff', 'Poly order 0', 'Poly order 0-1', 'Tonnetz M3y', 'Tonnetz M3x', 'Tonnetz m3y', 'Tonnetz m3x', 'Tonnetz 5y', 'Tonnetz 5x', 'Zero Crossing Rate', 'BPM' ]
    return featDict[featureIndex]



def drawTriangle(ax, pts, color):
    p = Polygon(pts, closed=False, facecolor = color, alpha=.7)
    ax = plt.gca()
    ax.add_patch(p)

def displayVAgraph(valence, arousal, names, min, max):
    fig = plt.figure()
    fig.subplots_adjust(top=0.8)
    ax = fig.add_subplot()
    ax.set_ylabel('Arousal')
    ax.set_xlabel('Valence')
    ax.set_xlim([min,max])
    ax.set_ylim([min,max])
    center = (max+min)/2
    drawTriangle(ax, np.array([[max,max], [center,max], [center,center]]), 'orange')
    drawTriangle(ax, np.array([[max,max], [max,center], [center,center]]), 'yellow')
    drawTriangle(ax, np.array([[max,min], [max,center], [center,center]]), 'lawngreen')
    drawTriangle(ax, np.array([[max,min], [center,min], [center,center]]), 'lime')
    drawTriangle(ax, np.array([[min,min], [center,min], [center,center]]), 'dodgerblue')
    drawTriangle(ax, np.array([[min,min], [min,center], [center,center]]), 'cyan')
    drawTriangle(ax, np.array([[min,max], [min,center], [center,center]]), 'violet')
    drawTriangle(ax, np.array([[min,max], [center,max], [center,center]]), 'red')
    ax.axvline(x=0, linewidth=1, color='k')
    ax.axhline(y=0, linewidth=1, color='k')
    
    ax.plot(valence, arousal, 'ko')
    
    if names:
        for i,name in enumerate(names):
            plt.annotate(name, # this is the text
                    (valence[i],arousal[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0,5), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center
    plt.show()

def normalizacija(array, faktor, inMin, inMax):
    return 2.*(array - inMin)/inMax-faktor


# function to parse CSV file of valence and arousal
def parseDEAM(pathEmotions):
    #pathToEmotions = os.path.join(dirname,pathEmotions)
    emotions = pd.read_csv(pathEmotions, index_col=0, sep=',')
    return emotions

def rezultatiTestData(predictions_valence, predictions_arousal):
    from commonFunctions import parseDEAM
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import r2_score
    from sklearn.metrics import max_error
    
    emotions = parseDEAM('dataset2/15_pravilne.csv')
    Y_valence = np.array(emotions['valence'].tolist())
    Y_arousal = np.array(emotions['arousal'].tolist())
    
    print('MSE:',round(mean_squared_error(Y_valence, np.array(predictions_valence)),4), round(mean_squared_error(Y_arousal, np.array(predictions_arousal)),4))
    print('MAE:',round(mean_absolute_error(Y_valence, np.array(predictions_valence)),4), round(mean_absolute_error(Y_arousal, np.array(predictions_arousal)),4))
    print('R2 :',round(r2_score(Y_valence, np.array(predictions_valence)),4), round(r2_score(Y_arousal, np.array(predictions_arousal)),4))
    print('EVS:',round(explained_variance_score(Y_valence, np.array(predictions_valence)),4), round(explained_variance_score(Y_arousal, np.array(predictions_arousal)),4))
    print('MXE:',round(max_error(Y_valence, np.array(predictions_valence)),4), round(max_error(Y_arousal, np.array(predictions_arousal)),4))



# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()