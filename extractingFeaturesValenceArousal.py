import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
from commonFunctions import musicFeatureExtraction
from commonFunctions import displayVAgraph
from commonFunctions import printProgressBar

dirname = os.path.dirname(__file__)

# function to parse CSV file of valence and arousal
def parseDEAM(pathEmotions):
    emotions = pd.read_csv(pathEmotions, index_col=0, sep=',')
    return emotions

subfolder = 'Dataset/'
#subfolder = 'dataset_small/'
#emotions = parseDEAM(subfolder+'emotions/filtered_annotations_vse.csv')
emotions = parseDEAM(subfolder+'emotions/valence_arousal_vse_normalized.csv')
#displayVAgraph(emotions['valence_mean'], emotions['arousal_mean'], False, 1, 9)

allFeatures = []
l = len(emotions)
counter = 0
printProgressBar(counter, l, prefix = '      Progress:', suffix = 'Complete', length = 120)
for fileName, row in emotions.iterrows():
    print(fileName)
    fullFilePath = subfolder+'audio/'+str(fileName)+'.mp3'
    data = musicFeatureExtraction(fullFilePath)
    allFeatures.append([fileName, data, emotions.loc[fileName,'valence_mean'], emotions.loc[fileName,'arousal_mean']])
    
    counter += 1
    printProgressBar(counter, l, prefix = '      Progress:', suffix = 'Complete', length = 120)

featuresdf = pd.DataFrame(allFeatures, columns=['id','features','valence', 'arousal'])
featuresdf = featuresdf.set_index(['id'])
print(featuresdf)
featuresdf.to_pickle(subfolder+'pickle/more_exported_features_valence_arousal2021.pkl')