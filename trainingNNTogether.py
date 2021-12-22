import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from commonFunctions import normalizacija
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import max_error
from pickle import load

regressionType = "NN_both"
full = "FULL"

selectionType = "reliefF"
#selectionType = "corelation"

seed = 0

def main():
    scaler = load(open('scaler.pkl', 'rb'))

    indexesFeatures = 5 # from 5 to 80 -> 90 is special for all features
    
    subfolder = 'Dataset/'
    featuresdf = pd.read_pickle(subfolder+'Pickle/199_exported_features_valence_arousal2021.pkl')
    
    f = open('results/'+regressionType+'_'+selectionType+'_results.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['MSE', 'MAE', 'R2', 'EVS', 'MXE_valence', 'MXE_arousal', 'noFeat'])
    
    while indexesFeatures < 100:
        X = np.array(featuresdf['features'].tolist())
        
        # combine VA
        y_valence = np.array(featuresdf['valence'].tolist())
        y_arousal = np.array(featuresdf['arousal'].tolist())
        y_valence = normalizacija(y_valence, 1, 1, 9)
        y_arousal = normalizacija(y_arousal, 1, 1, 9)
        y = np.array([y_valence, y_arousal])
        y = y.T
        
        X = scaler.transform(X)
        
        X_train = X
        y_train = y
        #X_train, X_features, y_train, y_features = train_test_split(X,y, test_size = 0.2, random_state = seed) # remove items used for RReliefF
        
        if(indexesFeatures <= 80):
            selectVA = 'arousal'
            selectedFeatIds = pd.read_pickle(subfolder+'indexes/'+selectionType+'_'+str(indexesFeatures)+'_'+selectVA+'_indexes.pkl')
            indexes = np.array(selectedFeatIds['indexes'].tolist()) # read indexes file
            X_train_filtered = X_train[:, indexes]# filter features by index
            X_train = X_train_filtered
        else:
            indexesFeatures = 199
        
        print("\n"+regressionType+" "+str(indexesFeatures))
        
        # model, results = trainModelKfold(X_train, y_train_norm, seed)
        X_train_NN, X_test, y_train_NN, y_test = train_test_split(X_train,y_train, test_size = 0.2, random_state = seed)
        model = NNRegression(X_train_NN, X_test, y_train_NN, y_test, indexesFeatures, True)
        
        y_pred_best = model.predict(X_test)
        MSE = np.round(mean_squared_error(y_test, y_pred_best), 5)
        MAE = np.round(mean_absolute_error(y_test, y_pred_best), 5)
        R2  = np.round(r2_score(y_test, y_pred_best), 5)
        EVS = np.round(explained_variance_score(y_test, y_pred_best), 5)
        MXE_valence = np.round(max_error(y_test[0], y_pred_best[0]), 5)
        MXE_arousal = np.round(max_error(y_test[1], y_pred_best[1]), 5)
        print()
        print('MSE:',MSE)
        print('MAE:',MAE)
        print('R2: ',R2 )
        print('EVS:',EVS)
        print('MXE_valence:',MXE_valence)
        print('MXE_arousal:',MXE_arousal)
        
        results = [MSE, MAE, R2, EVS, MXE_valence, MXE_arousal, indexesFeatures]
        writer.writerow(results)
        
        model.save('models/NNtogether/model_'+regressionType+'_'+selectionType+'_'+str(indexesFeatures)+'_all')
        
        if(indexesFeatures == 5):
            indexesFeatures+=5
        else:
            indexesFeatures+=10
    f.close()

def trainModelKfold(X, y, seed):
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

def NNRegression(X_train, X_test, y_train, y_test, indexesFeatures, earlyStop=False):
    model = buildModel(indexesFeatures)
    from keras.callbacks import EarlyStopping
    
    epochs = 30
    if(earlyStop):
        early_callback = EarlyStopping(monitor='val_loss', patience=10 )
        history = model.fit( x = X_train, y = y_train, epochs=epochs, validation_data = (X_test, y_test) , verbose=0, callbacks=[early_callback] )
    else:
        history = model.fit( x = X_train, y = y_train, epochs=epochs, validation_data = (X_test, y_test) , verbose=0 )
    
    # evaluate the keras model
    loss = model.evaluate(X_test, y_test)
    #print('Validation loss: %.2f' % (loss))
    showLoss(history, indexesFeatures)
    return model

def buildModel(noFeatures):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import BatchNormalization
    from keras.layers import Dropout
    
    model = Sequential()
    model.add(BatchNormalization())
    dropRate = .2
    if(noFeatures == 199):
        model.add(Dense( 199, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 100, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 80, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 50, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 10, activation='relu' ))
    
    elif(noFeatures == 80):
        model.add(Dense( 80, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 70, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 60, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 20, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 10, activation='relu' ))
    
    elif(noFeatures == 70):
        model.add(Dense( 70, activation='relu' ))
        model.add(Dense( 60, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 50, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 20, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 10, activation='relu' ))
    
    elif(noFeatures == 60):
        model.add(Dense( 60, activation='relu' ))
        model.add(Dense( 50, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 40, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 20, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 10, activation='relu' ))
    
    elif(noFeatures == 50):
        model.add(Dense( 50, activation='relu' ))
        model.add(Dense( 40, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 20, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 10, activation='relu' ))
    
    elif(noFeatures == 40):
        model.add(Dense( 40, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 20, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 10, activation='relu' ))
    
    elif(noFeatures == 30):
        model.add(Dense( 30, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 20, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 10, activation='relu' ))
    
    elif(noFeatures == 20):
        model.add(Dense( 20, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 10, activation='relu' ))
        
    elif(noFeatures == 10):
        model.add(Dense( 10, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 10, activation='relu' ))
     
    elif(noFeatures == 5):
        model.add(Dense( 5, activation='relu' ))
        model.add(Dropout(rate = dropRate))
        model.add(Dense( 5, activation='relu' ))
    
    model.add(Dropout(rate = dropRate))
    model.add(Dense(  2, activation='linear' ))
    
    model.compile( loss = "mean_squared_error", optimizer = 'adam')
    return model

def showLoss(history, indexesFeatures, show=False):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    ax = plt.gca()
    ax.set_ylim([0.04, 0.25])
    if show:
        plt.show()
    else:
        plt.savefig('results/NNmodelLoss/NN_both_'+str(indexesFeatures)+'_'+selectionType+'.png')

if __name__ == '__main__':
    main()