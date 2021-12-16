import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import joblib
import csv

from sklearn.model_selection import train_test_split
from commonFunctions import normalizacija

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import max_error


def main():
    indexesFeatures = 5 # from 5 to 80 -> 90 is special for all features
    VA = ['valence', 'arousal']
    #regressionType = "RF"
    regressionType = "SVR"
    selectionType = "reliefF"
    seed = 0
    
    subfolder = 'Dataset/'
    featuresdf = pd.read_pickle(subfolder+'Pickle/199_exported_features_valence_arousal2021.pkl')
    
    f = open('results/'+regressionType+'_results.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['MSE', 'MAE', 'R2', 'EVS', 'MXE', 'noFeat', 'VA'])
    
    
    while indexesFeatures < 100:
        for selectVA in VA:
            X = np.array(featuresdf['features'].tolist())
            y = np.array(featuresdf[selectVA].tolist())
            X_train, X_features, y_train, y_features = train_test_split(X,y, test_size = 0.2, random_state = seed) # remove items used for RReliefF
            y_train_norm = normalizacija(y_train, 1, 1, 9)
            
            if(indexesFeatures <= 80):
                selectedFeatIds = pd.read_pickle(subfolder+'indexes/'+selectionType+'_'+str(indexesFeatures)+'_'+selectVA+'_indexes.pkl')
                indexes = np.array(selectedFeatIds['indexes'].tolist()) # read indexes file
                X_train_filtered = X_train[:, indexes]# filter features by index
                X_train = X_train_filtered
            else:
                indexesFeatures = 199
                
            print()
            print(regressionType+" "+selectVA+" "+str(indexesFeatures))
            
            model, results = trainModelKfold(X_train, y_train_norm, regressionType, seed)
            results.append(indexesFeatures)
            results.append(selectVA)
            writer.writerow(results)
            
            joblib.dump(model, "./models/model"+regressionType+"_"+selectVA+"_"+str(indexesFeatures)+".joblib") # save model
            
        
        if(indexesFeatures == 5):
            indexesFeatures+=5
        else:
            indexesFeatures+=10
    f.close()


def trainModelKfold(X, y, typeReg, seed):
    allMSE = []
    allMAE = []
    allR2 = []
    allEVS = []
    allMXE = []
    
    
    kf = KFold(n_splits=10, shuffle = True, random_state = seed)
    best = None
    bestModel = None
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        regrModel = None
        if(typeReg=="RF"):
            regrModel = RForest(X_train, y_train, seed)
        elif(typeReg=="SVR"):
            regrModel = SVRegression(X_train, y_train, seed)
        elif(typeReg=="NN"):
            regrModel = NNRegression(X_train, X_test, y_train, y_test, True)
        
        predictions = regrModel.predict(X_test)
        MSE = mean_squared_error(y_test, predictions)
        allMSE.append(MSE)
        allMAE.append(mean_absolute_error(y_test, predictions))
        allR2.append(r2_score(y_test, predictions))
        allEVS.append(explained_variance_score(y_test, predictions))
        allMXE.append(max_error(y_test, predictions))
        
        print('MSE:', round(MSE,5))
        
        if(best == None or best > MSE):
            best = MSE
            bestModel = regrModel
        
    MSE = np.round(np.mean(allMSE), 5)
    MAE = np.round(np.mean(allMAE), 5)
    R2  = np.round(np.mean(allR2 ), 5)
    EVS = np.round(np.mean(allEVS), 5)
    MXE = np.round(np.mean(allMXE), 5)
    print()
    print('MSE:',MSE)
    print('MAE:',MAE)
    print('R2: ',R2 )
    print('EVS:',EVS)
    print('MXE:',MXE)
    return bestModel, [MSE, MAE, R2, EVS, MXE]


def RForest(X_train, y_train, seed):
    from sklearn.ensemble import RandomForestRegressor
    modelRF = RandomForestRegressor(criterion = 'mse', n_estimators = 300, random_state = seed)
    modelRF.fit(X_train, y_train)
    return modelRF

def SVRegression(X_train, y_train, seed):
    from sklearn.svm import SVR
    modelSVR = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    modelSVR.fit(X_train, y_train)
    return modelSVR


if __name__ == '__main__':
    main()