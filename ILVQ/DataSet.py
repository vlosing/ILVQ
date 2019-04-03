import numpy as np
import logging

from sklearn.datasets import load_svmlight_file
from . import Paths
import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
#from DataGeneration.dsPaths import *

dsSea = 'sea'
dsWeather = 'weather'
dsRialto = 'rialto'
dsOutdoor = 'outdoor'
dsPoker = 'poker'
dsRtgLarge = 'rtgLarge'
dsRbfLarge = 'rbfLarge'
dsChessIID= 'chessIID'
dsCovType = 'covType'
dsMnist8m = 'mnist8m'
dsMnist = 'mnist'
dsHiggs = 'higgs'
dsAirline = 'airline'
dsLedDrift = 'ledDrift'
dsPamap = 'pamap'
dsRotatingHyp = 'rotatingHyp'
dsMovingRBF = 'movingRBF'
dsInterRBF = 'interRBF'
dsInterRBF20 = 'interRBF20'
dsMovingSquares = 'movingSquares'
dsTransientChessb = 'transientChessb'
dsMixedDrift = 'mixedDrift'
dsRtg = 'rtg'
dsLedDriftSmall = 'ledDriftSmall'
dsElec = 'elec'
dsSusy = 'susy'
dsKdd = 'kdd'
dsSpam = 'spam'
dsNews20 = 'news20'
dsNews20Switched = 'news20Switched'
dsNews20Sorted = 'news20Sorted'

dsBorder = 'border'
dsOverlap = 'overlap'

r_loadType = 'loadType'
r_seperated = 'seperated'
r_single = 'single'
r_filePath = 'filePath'
r_arff = 'arff'
r_svmlight = 'svmlight'
r_labelsPath = 'labelsPath'
r_samplesPath = 'samplesPath'
r_delimiter = 'delimiter'
r_skipRows = 'skipRows'

class DataSet(object):
    def __init__(self, name, samples, labels, metaData=None):
        self.name = name
        self.samples = samples
        self.labels = labels
        self.metaData = metaData
        if self.metaData is None:
            self.dimensions = samples.shape[1]
        else:
            self.dimensions = len(samples[0].dtype.names)


        self.testSamples = None
        self.testLabels = None

    @staticmethod
    def getLabelDistributions(labels, classes):
        distribution = []
        for label in classes:
            numberOfSamples = len(np.where(labels == label)[0])
            distribution.append([label, numberOfSamples, numberOfSamples/float(len(labels))])
        return distribution

    def loggDSInformation(self):
        DataSet._loggDSInformation(self.name, self.dimensions, self.labels, np.unique(self.labels))

    @staticmethod
    def _loggDSInformation(name, dimensions, labels, classes):
        logging.info('name %s' %(name))
        logging.info('dimensions %s' %(dimensions))
        logging.info('samples %d' %(len(labels)))
        logging.info('classes %d' %(len(classes)))
        try:
            np.set_printoptions(precision=3, suppress=True)
            logging.info('labelDistr \n%s' %(np.array(DataSet.getLabelDistributions(labels, classes=classes)).astype(np.float)))
        finally:
            #XXVL Thats really ugly as set back to default
            np.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)


    @staticmethod
    def getIDStr(name, permutate):
        return name + '_perm(' + str(permutate) + ')'

def getDatasetProperties(name):
    dict = {
        dsSea: {r_loadType: r_seperated, r_samplesPath: seaFeaturesPath, r_labelsPath: seaLabelsPath, r_skipRows: 0, r_delimiter: ','},
        dsWeather: {r_loadType: r_seperated, r_samplesPath: weatherFeaturesPath, r_labelsPath: weatherLabelsPath, r_skipRows: 0, r_delimiter: ','},
        dsElec: {r_loadType: r_seperated, r_samplesPath: elecFeaturesPath,
                    r_labelsPath: elecLabelsPath, r_skipRows: 0, r_delimiter: ' ', 'excludeFeatures': [0,3]},
        dsRialto: {r_loadType: r_seperated, r_samplesPath: rialtoSamplesPath,
                 r_labelsPath: rialtoLabelsPath, r_skipRows: 0, r_delimiter: ' '},
        dsKdd: {r_loadType: r_arff, r_filePath: kddArffPath},

        dsOutdoor: {r_loadType: r_seperated, r_samplesPath: outdoorStreamSamplesPath,
                   r_labelsPath: outdoorStreamLabelsPath, r_skipRows: 0, r_delimiter: ' '},
        dsPoker: {r_loadType: r_arff, r_filePath: pokerArffPath},
        dsMovingRBF: {r_loadType: r_seperated, r_samplesPath: movingRBFSamplesPath,
                      r_labelsPath: movingRBFLabelsPath, r_skipRows: 0, r_delimiter: ' '},
        dsRotatingHyp: {r_loadType: r_seperated, r_samplesPath: rotatingHypSamplesPath,
                      r_labelsPath: rotatingHypLabelsPath, r_skipRows: 0, r_delimiter: ' '},
        dsMovingSquares: {r_loadType: r_seperated, r_samplesPath: movingSquaresSamplesPath,
                        r_labelsPath: movingSquaresLabelsPath, r_skipRows: 0, r_delimiter: ' '},
        dsTransientChessb: {r_loadType: r_seperated, r_samplesPath: transientChessboardSamplesPath,
                          r_labelsPath: transientChessboardLabelsPath, r_skipRows: 0, r_delimiter: ' '},
        dsMixedDrift: {r_loadType: r_seperated, r_samplesPath: mixedDriftSamplesPath,
                            r_labelsPath: mixedDriftLabelsPath, r_skipRows: 0, r_delimiter: ' '},
        dsInterRBF: {r_loadType: r_seperated, r_samplesPath: interchangingRBFSamplesPath,
                          r_labelsPath: interchangingRBFLabelsPath, r_skipRows: 0, r_delimiter: ' '},
        dsInterRBF20: {r_loadType: r_arff, r_filePath: interchangingRBF20ArffPath},
        dsAirline: {r_loadType: r_arff, r_filePath: airlinesArffPath},
        dsLedDrift: {r_loadType: r_arff, r_filePath: ledDriftArffPath},
        dsLedDriftSmall: {r_loadType: r_arff, r_filePath: ledDriftSmallArffPath},
        dsRtgLarge: {r_loadType: r_arff, r_filePath: rtgLargeArffPath},
        dsRtg: {r_loadType: r_seperated, r_samplesPath: rtgSamplesPath,
                     r_labelsPath: rtgLabelsPath, r_skipRows: 0, r_delimiter: ' '},
        dsRbfLarge: {r_loadType: r_single, r_filePath: rbfLargeCsvPath, 'labelIdx': 100, r_delimiter: ','},
        dsChessIID: {r_loadType: r_seperated, r_samplesPath: chessIIDSamplesPath, r_labelsPath: chessIIDLabelsPath, r_skipRows: 0, r_delimiter: ' '},
        'chessIIDLarge': {r_loadType: r_seperated, r_samplesPath: chessIIDLargeSamplesPath,
                     r_labelsPath: chessIIDLargeLabelsPath, r_skipRows: 0, r_delimiter: ' '},
        dsCovType: {r_loadType: r_arff, r_filePath: covTypeArffPath},
        'epsilon': {r_loadType: r_svmlight, r_filePath: epsilonPath, 'numParts': 2},
        #'mnist8m': {r_loadType: r_svmlight, r_filePath: MNist8MSVMLightPath, 'dtype':np.int16},
        #'mnist8m': {r_loadType: r_arff, r_filePath: MNist8MArffPath},
        dsMnist8m: {r_loadType: r_single, r_filePath: MNist8MPath, 'labelIdx': 784, r_delimiter: ',', 'dtype': np.int16},
        dsMnist: {r_loadType: r_seperated, r_samplesPath: MNistSamplesPath, r_labelsPath: MNistLabelsPath, r_skipRows: 0, r_delimiter: ' '},
        'adult': {r_loadType: r_svmlight, r_filePath: adultPath},
        #'higgs': {r_loadType: r_arff, r_filePath: higgsArffPath},
        dsHiggs: {r_loadType: r_single, r_filePath: higgsCSVPath, 'labelIdx': 0, r_delimiter: ','},
        dsPamap: {r_loadType: r_single, r_filePath: pamapPath, 'labelIdx': 0, r_delimiter: ' '},
        'activity': {r_loadType: r_single, r_filePath: activityPath, 'labelIdx': 3, r_delimiter: ' '},
        dsSusy: {r_loadType: r_single, r_filePath: susyCSVPath, 'labelIdx': 0, r_delimiter: ','},
        #dsSpam: {r_loadType: r_seperated, r_samplesPath: spamSamplesPath,
        #                    r_labelsPath: spamLabelsPath, r_skipRows: 0, r_delimiter: ' '},
        dsSpam: {r_loadType: r_svmlight, r_filePath: spamPath, 'dtype': np.bool},
        dsNews20: {r_loadType: r_svmlight, r_filePath: news20Path, 'dtype': np.int8},
        dsNews20Switched: {r_loadType: r_arff, r_filePath: news20SwitchedPath, 'dtype': np.int8},
        dsNews20Sorted: {r_loadType: r_svmlight, r_filePath: news20SortedPath, 'dtype': np.int8}
            }


    return dict[name]


def isinteger(x):
    return np.all(np.equal(np.mod(x, 1), 0))

def permutateDataset(X, Y):
    indices = np.random.permutation(len(Y))
    return X[indices], Y[indices]

def getBootstrapSample(X, Y):
    indices = np.random.randint(len(Y), size=len(Y))
    indices = np.sort(indices)
    return X[indices], Y[indices]

class DataSetChunkWise(object):
    def __init__(self, name, chunkSize=None, permutate=False, bootstrap=False):
        self.name = name
        self.numParts = 1
        self.chunkSize = chunkSize

        self.partIdx = 0
        self.currPartDataIdx = 0
        self.chunkIdx = 0
        self.totalDataIdx = 0

        self.permutate = permutate
        self.bootstrap = bootstrap
        self.dataProps = getDatasetProperties(name)
        if 'numParts' in self.dataProps:
            self.numParts = self.dataProps['numParts']
        #self.partX, self.partY, self.classes, self.metaData = self.getNextPart()
        self.getNextChunk()
        self.maxFloatPrecision = None
        if 'maxFloatPrecision' in self.dataProps:
            self.maxFloatPrecision = self.dataProps['maxFloatPrecision']

        self.numExamples = self.chunkX.shape[0]
        if self.metaData is None:
            self.dimensions = self.chunkX.shape[1]
        else:
            self.dimensions = len(self.chunkX[0].dtype.names)

    def reset(self):
        self.partIdx = 0
        self.currPartDataIdx = 0
        self.chunkIdx = 0
        self.getNextChunk()

    def loadSeperated(self, samplesPath, labelsPath, delimiter=',', partIdx=None, skiprows=0, dtype=None):
        _samplesPath = Paths.getLocalGlobalPath(self.getFilePathWithIdx(samplesPath, partIdx=partIdx))
        _labelsPath = Paths.getLocalGlobalPath(self.getFilePathWithIdx(labelsPath, partIdx=partIdx))
        if dtype is None:
            X = pd.read_csv(_samplesPath, header=None, delimiter=delimiter, skiprows=skiprows).values
        else:
            X = pd.read_csv(_samplesPath, header=None, delimiter=delimiter, skiprows=skiprows, dtype=dtype).values
        Y = pd.read_csv(_labelsPath, header=None, delimiter=delimiter, skiprows=skiprows, dtype=np.int16).values.ravel()
        return X, Y

    def loadSingle(self, filePath, labelIdx, delimiter=',', partIdx=None, skiprows=0, dtype=None):
        _filePath = Paths.getLocalGlobalPath(self.getFilePathWithIdx(filePath, partIdx=partIdx))
        print(_filePath)
        if dtype is None:
            data = pd.read_csv(_filePath, header=None, delimiter=delimiter, skiprows=skiprows).values
        else:
            data = pd.read_csv(_filePath, header=None, delimiter=delimiter, skiprows=skiprows, dtype=dtype).values
        #data = np.loadtxt(_filePath, skiprows=skiprows, delimiter=delimiter)
        if isinstance(data[0, labelIdx], int):
            Y = data[:, labelIdx].astype(np.int16)
        else:
            Y = data[:, labelIdx]
        X = np.delete(data, labelIdx, axis=1)

        return X, Y

    def loadSingleNP(self, filePath, labelIdx, delimiter=',', partIdx=None, skiprows=0):
        _filePath = Paths.getLocalGlobalPath(self.getFilePathWithIdx(filePath, partIdx=partIdx))
        data = np.loadtxt(_filePath, skiprows=skiprows, delimiter=delimiter, dtype=str)
        if isinstance(data[0, labelIdx], int):
            Y = data[:, labelIdx].astype(np.int16)
        else:
            Y = data[:, labelIdx]
        X = np.delete(data, labelIdx, axis=1).astype(np.float)
        return X, Y

    def getFilePathWithIdx(self, filePath, partIdx=None):
        _filePath = filePath
        if self.numParts > 1 and partIdx is not None:
            _filePath, suffix = _filePath.split('.')
            _filePath = _filePath + str(partIdx) + '.' + suffix
        return _filePath

    def loadSVMLight(self, filePath, partIdx=None, dtype=None):
        _filePath = Paths.getLocalGlobalPath(self.getFilePathWithIdx(filePath, partIdx=partIdx))
        if dtype is None:
            X, Y = load_svmlight_file(_filePath)
        else:
            X, Y = load_svmlight_file(_filePath, dtype=dtype)
        X = X.toarray()
        if isinstance(Y[0], int):
            Y = (Y).astype(np.int16)[:]
        return X, Y

    def loadArff(self, filePath, partIdx=None):
        _filePath = Paths.getLocalGlobalPath(self.getFilePathWithIdx(filePath, partIdx=partIdx))
        f = open(_filePath, 'r')
        data, metaData = arff.loadarff(f)
        X = data[list(data.dtype.names[:-1])]


        Y = data[data.dtype.names[-1]]
        if isinstance(Y[0], int):
            Y = (Y).astype(np.int16)[:]
        return X, Y, metaData

    def getNextPart(self):
        X = None
        Y = None
        if self.partIdx < self.numParts:
            metaData = None
            dtype = np.float32
            if 'dtype' in self.dataProps:
                dtype = self.dataProps['dtype']
            if self.dataProps[r_loadType] == r_seperated:
                X, Y = self.loadSeperated(samplesPath=self.dataProps[r_samplesPath], labelsPath=self.dataProps[r_labelsPath], skiprows=self.dataProps[r_skipRows], delimiter=self.dataProps[r_delimiter],
                                                   partIdx=None if self.numParts == 1 else self.partIdx)
            elif self.dataProps[r_loadType] == r_svmlight:
                X, Y = self.loadSVMLight(filePath=self.dataProps[r_filePath], partIdx=self.partIdx, dtype=dtype)
                #X = X[:1500,:]
                #Y = Y[:1500]
            elif self.dataProps[r_loadType] == r_arff:
                X, Y, metaData = self.loadArff(filePath=self.dataProps[r_filePath], partIdx=self.partIdx)
            elif self.dataProps[r_loadType] == r_single:
                X, Y = self.loadSingle(filePath=self.dataProps[r_filePath], labelIdx=self.dataProps['labelIdx'], delimiter=self.dataProps[r_delimiter], partIdx=self.partIdx, dtype=dtype)
            elif self.dataProps[r_loadType] == 'singleNP':
                self.X, self.Y = self.loadSingleNP(filePath=self.dataProps[r_filePath], labelIdx=self.dataProps['labelIdx'], delimiter=self.dataProps[r_delimiter], partIdx=self.partIdx)
            if 'excludeFeatures' in self.dataProps:
                X = np.delete(X, self.dataProps['excludeFeatures'], axis=1)
            if self.partIdx == 0:
                self.labelEncoder = preprocessing.LabelEncoder()
                self.labelEncoder.fit(Y)
                Y = self.labelEncoder.transform(Y)
                classes = self.labelEncoder.transform(self.labelEncoder.classes_)
                self.classes = classes
                self.metaData = metaData
            self.partIdx += 1
        if self.permutate and X is not None:
            X, Y = permutateDataset(X, Y)
        if self.bootstrap and X is not None:
            X, Y = getBootstrapSample(X, Y)
        self.partX = X
        self.partY = Y
        return self.partX, self.partY, self.classes, self.metaData

    def getNextChunk(self):
        X = None
        Y = None
        if self.currPartDataIdx == 0:
            if not self.getNextPart()[0] is None:
                X = self.partX[self.currPartDataIdx:self.currPartDataIdx + self.chunkSize]
                Y = self.partY[self.currPartDataIdx:self.currPartDataIdx + self.chunkSize]
                self.currPartDataIdx += self.chunkSize
                self.totalDataIdx += self.chunkSize
            self.chunkIdx += 1
        elif self.partX is None:
            True #XXVL refactor!
        elif self.currPartDataIdx + self.chunkSize > self.partX.shape[0]:
            X = self.partX[self.currPartDataIdx:]
            Y = self.partY[self.currPartDataIdx:]
            delta = self.partX.shape[0] - self.currPartDataIdx
            self.totalDataIdx += delta
            if not self.getNextPart()[0] is None:
                self.currPartDataIdx = self.chunkSize - delta
                self.totalDataIdx += self.currPartDataIdx
                X = np.vstack([self.chunkX, self.partX[:self.currPartDataIdx]])
                Y = np.append(self.chunkY, self.partY[:self.currPartDataIdx])
            else:
                self.currPartDataIdx += delta
                if X.shape[0] == 0:
                    X = None
                    Y = None
            self.chunkIdx += 1
        else:
            X = self.partX[self.currPartDataIdx:self.currPartDataIdx + self.chunkSize]
            Y = self.partY[self.currPartDataIdx:self.currPartDataIdx + self.chunkSize]
            self.currPartDataIdx += self.chunkSize
            self.totalDataIdx += self.chunkSize
            self.chunkIdx += 1
        self.chunkX = X
        self.chunkY = Y
        return self.chunkX, self.chunkY, self.classes, self.metaData

    def printDSInformation(self):
        DataSet._loggDSInformation(self.name, self.dimensions, self.partY, self.classes)

class TrainTestDataSet(DataSet):
    def __init__(self, name, trainSamples, trainLabels, testSamples, testLabels):
        super(TrainTestDataSet, self).__init__(name, trainSamples, trainLabels)
        self.testSamples = testSamples
        self.testLabels = testLabels

