from .realDataSets import *
from .DataSet import *

from sklearn import preprocessing
import numpy as np

def exportData(XTrain, yTrain, XTest, yTest, fileNamePrefix):
    trainFeaturesFileName = fileNamePrefix + '-train.data'
    trainLabelsFileName = fileNamePrefix + '-train.labels'
    if np.issubdtype(yTrain.dtype, int) or np.issubdtype(yTrain.dtype, float):
        trainLabels = yTrain
        testLabels = yTest
    else:
        le = preprocessing.LabelEncoder()
        le.fit(np.append(yTrain, yTest))

        trainLabels = le.transform(yTrain)
        testLabels = le.transform(yTest)

    if issubclass(XTrain.dtype.type, np.integer):
        # print 'save int'
        # df = pd.DataFrame(self.TrainSamplesLst[i])
        # df.to_csv(trainFeaturesFileName, header=False, sep=' ')
        # df = pd.DataFrame(trainLabels)
        # df.to_csv(trainLabelsFileName, header=False, sep=' ')

        np.savetxt(trainFeaturesFileName, XTrain, fmt='%i', comments='')
        np.savetxt(trainLabelsFileName, trainLabels, fmt='%i', comments='')
    else:
        np.savetxt(trainFeaturesFileName, XTrain, fmt='%.18g', comments='')
        np.savetxt(trainLabelsFileName, trainLabels, fmt='%i', comments='')

    if len(testLabels) > 0:
        testFeaturesFileName = fileNamePrefix + '-test.data'
        testLabelsFileName = fileNamePrefix + '-test.labels'
        if issubclass(XTest.dtype.type, np.integer):
            np.savetxt(testFeaturesFileName, XTest, fmt='%i', comments='')
            np.savetxt(testLabelsFileName, testLabels, fmt='%i', comments='')
        else:
            np.savetxt(testFeaturesFileName, XTest, fmt='%.7f', comments='')
            np.savetxt(testLabelsFileName, testLabels, fmt='%i', comments='')
        return trainFeaturesFileName, trainLabelsFileName, testFeaturesFileName, testLabelsFileName
    else:
        return trainFeaturesFileName, trainLabelsFileName, '', ''

def getDataSet(name):
    metaData = None
    if name == 'outdoorStream':
        samples, labels = getOutdoorStream()
    elif name == 'outdoor':
        trainSamples, trainLabels, testSamples, testLabels = getOutdoor()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'border':
        trainSamples, trainLabels, testSamples, testLabels = getBorder()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'overlap':
        trainSamples, trainLabels, testSamples, testLabels = getOverlap()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)
    elif name == 'coil':
        trainSamples, trainLabels, testSamples, testLabels = getCOIL()
        return TrainTestDataSet(name, trainSamples, trainLabels, testSamples, testLabels)

    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    return DataSet(name, samples, labels.astype(np.int32), metaData=metaData)


def isStationary(dataSetName):
    if dataSetName in ['border', 'overlap', 'coil', 'USPS', 'DNA',
                        'letter', 'isolet', 'gisette',
                        'mnist', 'outdoor', 'satImage',
                        'penDigits', 'HAR', 'optDigits', 'news20', 'noise', 'stopping', 'passStopTurn', 'adult', 'covTypeTT', 'susy']:
        return True
    elif dataSetName in [dsWeather, dsAirline, dsLedDriftSmall, 'elec', 'spam', 'covType', 'poker',
                         'movingRBF', 'sea', 'rotatingHyp',
                         'interRBF', 'movingSquares', 'rbfGradual', 'rectGradual',
                          'transientChessb', 'chessIID', 'chessIIDLarge', 'outdoorStream', 'rialto', 'mixedDrift',
                          'keystroke', 'news20_1000', 'news20Virtual', 'news20Virtual2', 'news20Abrupt', 'testUniform', 'mnistStream',
                         'rtg', 'rtgLarge', 'susyStream', 'santander', 'kdd', 'airlines', 'gmsc', 'skinNonSkin', 'mnist8m', 'epsilon', 'rbfLarge']:
        return False
    else:
        raise Exception('unknown dataset')