import pandas as pd
import math
import numpy as np
import scipy.spatial
import timeit
from sklearn import model_selection
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer

# --------------------------------------------------------------------------------------
# From hw05

# Paste your readData, foldsTest, kFoldCVManual, and findNearestHOF functions from hw05 in here.
def readData(numRows = None):
    inputCols = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "Diluted", "Proline"]
    outputCol = 'Class'
    colNames = [outputCol] + inputCols  # concatenate two lists into one
    wineDF = pd.read_csv("data/wine.data", header=None, names=colNames, nrows = numRows)
    
    return wineDF, inputCols, outputCol


def foldsTest(trainInputDF, trainOutputSeries, testInputDF, testOutputSeries):
    print("================================")
    # For the training set, we're converting to a list so that all the contents are nicely printed out.
    # This isn't necessary for the testing set, because those indices are always contiguous.
    print("Train input:\n", list(trainInputDF.index))
    print("Train output:\n", list(trainOutputSeries.index))
    
    print("Test input:\n", testInputDF.index)
    print("Test output:\n", testOutputSeries.index)
    
    return trainInputDF.index[-1] # just returning an arbitrary, made up value here - we'll do something real in a future assignment


def kFoldCVManual(k, inputDF, outputSeries, testFunc):
    avg = 0
    foldSize = len(outputSeries) / k
    
    for i in range(0, k, 1):
        
        start = int (i*foldSize) 
        upToNotIncluding = int (((i+1)*foldSize))
         
        trainInputDF = pd.concat([inputDF.loc[:start,:].iloc[:-1],inputDF.loc[upToNotIncluding:, :]]) 
        trainOutputSeries =  pd.concat([outputSeries.loc[:start].iloc[:-1],outputSeries.loc[upToNotIncluding:]])
        testInputDF = inputDF.loc[start:upToNotIncluding-1]
        testOutputSeries = outputSeries.loc[start:upToNotIncluding-1]
        avg = avg + testFunc(trainInputDF, trainOutputSeries, testInputDF, testOutputSeries)
    return(avg/k)
   
def findNearestHOF(df, testRow):
    s = df.apply(lambda row: scipy.spatial.distance.euclidean(row.loc[:], testRow), axis = 1)
    minID = s.idxmin()
    return minID




# ---------------------------------
# Problem 2

# Given
def partialOneNNTest():
    df, inputCols, outputCol = readData()
    
    # Just arbitrarily choose a small training and testing set from the entire df, for easy testing
    # Note how I'm chaining a .loc with a .iloc so I can use the indexers I want for row and column
    trainInputDF      = df.loc[:, inputCols].iloc[125:131, :]
    trainOutputSeries = df.loc[:, outputCol].iloc[125:131]
    testInputDF       = df.loc[:, inputCols].iloc[131:135, :]
    testOutputSeries  = df.loc[:, outputCol].iloc[131:135]
    
    return oneNNTest(trainInputDF, trainOutputSeries, testInputDF, testOutputSeries)

def oneNNTest(trainInputDF, trainOutputSeries, testInputDF, testOutputSeries):
    # a is for testing series
    # b is for training series
    #have row a, compaing a to every b in training
    s = testInputDF.apply(lambda row: findNearestHOF(trainInputDF,row), axis =1)
    s2 = s.map(lambda nearestTrainIndx: trainOutputSeries.loc[nearestTrainIndx])
    return accuracyOfActualVsPredicted(s2, testOutputSeries)
                          
    
        
# Given
def accuracyOfActualVsPredicted(actualOutputSeries, predOutputSeries):
    compare = (actualOutputSeries == predOutputSeries).value_counts()
    
    # actualOutputSeries == predOutputSeries makes a Series of Boolean values.
    # So in this case, value_counts() makes a Series with just two elements:
    # - with index "False" is the number of times False appears in the Series
    # - with index "True" is the number of times True appears in the Series

    # print("compare:", compare, type(compare), sep='\n', end='\n\n')
    
    # if there are no Trues in compare, then compare[True] throws an error. So we have to check:
    if (True in compare):
        accuracy = compare[True] / actualOutputSeries.size
    else:
        accuracy = 0
    
    return accuracy





# --------------------------------------------------------------------------------------
# Problems 4-6

# Given
def testTheClass():
    fullDF, inputCols, outputCol = readData()
    
    # Use rows 0:100 for the training set
    trainDF = fullDF.iloc[0:100, :]
    
    # Set up the inputs and outputs for the training set
    trainInputDF = trainDF.loc[:, inputCols]
    trainOutputSeries = trainDF.loc[:, outputCol]
    
    # Make the classifier object and "fit" the training set
    alg = OneNNClassifier()
    alg.fit(trainInputDF, trainOutputSeries)
    
    # Use rows 100:200 for the testing set
    testDF = fullDF.iloc[100:200, :]
    
    # Set up the inputs for the testing set
    testInputDF = testDF.loc[:, inputCols]
    
    # Predict outputs for just a single row (a Series)
    print("Series:", alg.predict(testInputDF.iloc[0, :]), end='\n\n')
    
    # Predict outputs for the entire testing set (a DataFrame)
    print("DF:", alg.predict(testInputDF), sep='\n')

class OneNNClassifier(BaseEstimator, ClassifierMixin):
    def _init_(self):
        self.inputsDF = None
        self.outputSeries = None
        self.scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True) 
    
    def fit(self, inputsDF, outputSeries):
        self.inputsDF = inputsDF
        self.outputSeries = outputSeries
        return self
    
    def predict(self, testInput):
        if isinstance(testInput, pd.core.series.Series):
            # testInput is a Series, so predict for just this one row
            return findNearestHOF(testInput, self)
        else:
            return testInput.apply(lambda row: findNearestHOF(testInput,row), axis =1)



   
# --------------------------------------------------------------------------------------
# Problem 9

# Given
def testBuiltIn():
    fullDF, inputCols, outputCol = readData()
    result = kFoldCVBuiltIn(3, fullDF.loc[:, inputCols], fullDF.loc[:, outputCol])
    print(result)

# Given
def compareManualAndBuiltIn(k=10):
    df, inputCols, outputCol = readData()
    
    inputDF = df.loc[:, inputCols]
    outputSeries = df.loc[:, outputCol]
    
    results = kFoldCVBuiltIn(k, inputDF, outputSeries)
    print("Built-in:", results)
    
    results = kFoldCVManual(k, inputDF, outputSeries, oneNNTest)
    print("Manual:", results)










# --------------------------------------------------------------------------------------
# Given
def test06():
    df, inputCols, outputCol = readData()
    alg = OneNNClassifier()
    alg.fit(df.loc[:, inputCols], df.loc[:, outputCol])
    print(model_selection.cross_val_score(alg, df.loc[:, inputCols], df.loc[:, outputCol], cv=10, scoring=alg.scorer).mean())
