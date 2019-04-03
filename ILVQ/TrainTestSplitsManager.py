__author__ = 'vlosing'

import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler


def getIndicesOrderedByLabel(labels):
    indices = np.array([])
    uniqueLabels = np.unique(labels)
    uniqueLabels = uniqueLabels[np.random.permutation(len(uniqueLabels))]
    for label in uniqueLabels:
        labelIndices = np.where(labels == label)[0]
        indices = np.append(indices, labelIndices)
    return indices


def getOrderedIndices(dataOrder, labels, chunkSize):
    indices = np.array([])
    if dataOrder == 'random':
        indices = np.random.permutation(len(labels))
    elif dataOrder == 'original':
        indices = np.arange(len(labels))
    elif dataOrder == 'orderedByLabel':
        #indices = np.argsort(labels)
        indices = getIndicesOrderedByLabel(labels)
    elif dataOrder == 'chunksRandom':
        '''self.checkForBalance(labels)
        indices = np.argsort(labels)
        numberOfChunks = len(indices) / self.chunkSize
        shuffledChunkIndices = np.random.permutation(numberOfChunks)
        indices = np.split(indices, numberOfChunks)
        indices = indices[shuffledChunkIndices]
        indices = np.concatenate(indices)'''
        uniqueLabels = np.unique(labels)
        splitIndices = []
        for label in uniqueLabels:
            labelIndices = np.where(labels == label)[0]
            numberOfChunks = len(labelIndices) / chunkSize

            splitIndices = splitIndices + np.array_split(labelIndices, numberOfChunks)
        splitIndices = np.array(splitIndices)
        splitIndices = splitIndices[np.random.permutation(len(splitIndices))]
        indices = np.concatenate(splitIndices)
    else:
        raise NameError('unknown data-order ' + dataOrder)
    indices = indices.astype(int)
    return indices



class TrainTestSplitsManager(object):
    def __init__(self, samples, labels, dataOrder='random', chunkSize=1, shuffle=True, stratified=False, splitType='simple', numberOfFolds=2, trainSetSize=0.5, testSamples=None, testLabels=None):
        self.samples = samples
        self.labels = labels
        self.TrainLabelsLst = []
        self.TrainSamplesLst = []
        self.TestLabelsLst = []
        self.TestSamplesLst = []
        self.TrainFileNames = []
        self.TestFileNames = []
        self.TrainShuffledIndices = []
        self.clearBeforeSplit = True
        self.dataOrder = dataOrder
        self.chunkSize = chunkSize
        self.shuffle = shuffle
        self.stratified = stratified
        self.splitType = splitType
        self.numberOfFolds = numberOfFolds
        self.trainSetSize = trainSetSize
        self.testSamples = testSamples
        self.testLabels = testLabels

    def clear(self):
        self.TrainLabelsLst = []
        self.TrainSamplesLst = []
        self.TestLabelsLst = []
        self.TestSamplesLst = []
        self.TrainFileNames = []
        self.TestFileNames = []
        self.TrainShuffledIndices = []

    def scaleData(self, maxSamples=None, maxProportion=None):
        if maxSamples and maxProportion:
            numScalingSamples = min(len(self.TrainLabelsLst[0]) * maxProportion, maxSamples)
        elif maxSamples:
            numScalingSamples = maxSamples
        elif maxProportion:
            numScalingSamples = len(self.TrainLabelsLst[0]) * maxProportion
        else:
            numScalingSamples = len(self.TrainLabelsLst[0])

        #scaler = StandardScaler().fit(self.TrainSamplesLst[0][:numScalingSamples, :])
        scaler = MinMaxScaler(feature_range=(-1,1)).fit(self.TrainSamplesLst[0][:numScalingSamples, :])

        self.TrainSamplesLst[0] = scaler.transform(self.TrainSamplesLst[0])

        if len(self.TestSamplesLst[0]) > 0:
            self.TestSamplesLst[0] = scaler.transform(self.TestSamplesLst[0])

    '''@staticmethod
    def getEdges(samples, bins=10):
        edges = np.empty(shape=(0 ,bins+1))
        for d in range(samples.shape[1]):
            minValue = np.min(samples[:, d])
            maxValue = np.max(samples[:, d])
            if minValue == maxValue:
                dimEdges = np.repeat(minValue, bins+1)
            else:
                stepSize = (maxValue - minValue) / float(bins)
                dimEdges = np.append(np.arange(minValue, maxValue, stepSize), maxValue)
            edges = np.vstack([edges, dimEdges])
        return edges



    @staticmethod
    def discretizeSamples(samples, edges=None, bins=10):
        if edges ==None:
            #h, _edges = np.histogramdd(samples, bins=bins)
            _edges = TrainTestSplitsManager.getEdges(samples, bins=bins)
        else:
            _edges = edges
        newSamples = np.empty(shape=samples.shape)
        for i in range(samples.shape[0]):
            for d in range(samples.shape[1]):

                indices = np.where(samples[i,d] >= _edges[d])[0]
                if len(indices) == 0:
                   idx = 0
                else:
                    idx = indices[-1]
                #newSamples[i, d] = _edges[d][idx]
                newSamples[i, d] = idx
        return newSamples, _edges

    @staticmethod
    def binarizeSamples(samples):
        lbin = LabelBinarizer()
        for k in range(np.size(samples, 1)):
            if k==0:
                samplesBinarized = lbin.fit_transform(samples[:,k])
            else:
                samplesBinarized = np.hstack((samplesBinarized, lbin.fit_transform(samples[:,k])))
        return samplesBinarized'''

    def shuffleOrg(self):
        if self.shuffle:
            shuffledIndices = np.random.permutation(len(self.labels))
            self.samples = self.samples[shuffledIndices]
            self.labels = self.labels[shuffledIndices]

    def generateSplits(self):
        if self.clearBeforeSplit: #XXVL kann vllt weg
            self.clear()
        samples = self.samples
        labels = self.labels
        if self.shuffle:
            shuffledIndices = np.random.permutation(len(labels))
            samples = samples[shuffledIndices]
            labels = labels[shuffledIndices]
        indices = getOrderedIndices(self.dataOrder, labels, self.chunkSize)
        samples = samples[indices]
        labels = labels[indices]
        if self.splitType == 'simple':
            dataSetSplitter = TrainTestSplitter(samples, labels, self.stratified, self.trainSetSize)
        elif self.splitType == 'kFold':
            dataSetSplitter = KFoldSplitter(samples, labels, self.stratified, self.numberOfFolds)
        elif self.splitType == None:
            dataSetSplitter = DummySplitter(samples, labels, self.testSamples, self.testLabels)
        else:
            raise NameError('unknown splitType ' + self.splitType)
        self.TrainSamplesLst, self.TrainLabelsLst, self.TestSamplesLst, self.TestLabelsLst = dataSetSplitter.getSplits()

    def exportToORFFormat(self, fileNamePrefix):
        print(fileNamePrefix)
        trainFeaturesFileNames = []
        trainLabelsFileNames = []
        testFeaturesFileNames = []
        testLabelsFileNames = []
        for i in range(len(self.TrainSamplesLst)):
            fileNamePrefix_ = fileNamePrefix + '_' + str(i)
            trainFeaturesFileName, trainLabelsFileName, testFeaturesFileName, testLabelsFileName =  TrainTestSplitsManager.exportData(self.TrainSamplesLst[i], self.TrainLabelsLst[i], self.TestSamplesLst[i], self.TestLabelsLst[i], fileNamePrefix_)
            trainFeaturesFileNames.append(trainFeaturesFileName)
            trainLabelsFileNames.append(trainLabelsFileName)
            if testFeaturesFileName != '':
                testFeaturesFileNames.append(testFeaturesFileName)
                testLabelsFileNames.append(testLabelsFileName)
        return trainFeaturesFileNames, trainLabelsFileNames, testFeaturesFileNames, testLabelsFileNames

class DataSetSplitter(object):
    def __init__(self, samples, labels, stratified):
        self.samples = samples
        self.labels = labels
        self.stratified = stratified
    def getSplits(self):
        raise NotImplementedError()


class DummySplitter(DataSetSplitter):
    def __init__(self, trainSamples, trainLabels, testSamples, testLabels):
        self.trainSamplesLst = []
        self.testSamplesLst = []
        self.trainLabelsLst = []
        self.testLabelsLst = []
        self.trainSamplesLst.append(trainSamples)
        self.trainLabelsLst.append(trainLabels)
        self.testSamplesLst.append(testSamples)
        self.testLabelsLst.append(testLabels)

    def getSplits(self):
        return self.trainSamplesLst, self.trainLabelsLst, self.testSamplesLst, self.testLabelsLst


class TrainTestSplitter(DataSetSplitter):
    def __init__(self, samples, labels, stratified, trainSetSize):
        super(TrainTestSplitter, self).__init__(samples, labels, stratified)
        self.trainSetSize = trainSetSize

    def getSplits(self):
        trainSamplesLst = []
        testSamplesLst = []
        trainLabelsLst = []
        testLabelsLst = []

        if self.stratified:
            numberOfSamplesPerClass = int(len(self.labels) * self.trainSetSize/ len(np.unique(self.labels)))
            trainIndices = []
            testIndices = []
            for className in np.unique(self.labels):
                indices = np.where(self.labels==className)[0]
                trainIndices = np.append(trainIndices, indices[0:numberOfSamplesPerClass])
                testIndices = np.append(testIndices, indices[numberOfSamplesPerClass:])
            trainIndices = trainIndices.astype(np.int)
            testIndices = testIndices.astype(np.int)
        else:
            indices = np.arange(len(self.labels))
            trainIndices = indices[0:int(len(indices) * self.trainSetSize)]
            testIndices = indices[len(trainIndices):len(indices)]
        trainSamplesLst.append(self.samples[trainIndices])
        testSamplesLst.append(self.samples[testIndices])
        trainLabelsLst.append(self.labels[trainIndices])
        testLabelsLst.append(self.labels[testIndices])
        return trainSamplesLst, trainLabelsLst, testSamplesLst, testLabelsLst


class KFoldSplitter(DataSetSplitter):
    def __init__(self, samples, labels, stratified, numberOfFolds):
        super(KFoldSplitter, self).__init__(samples, labels, stratified)
        self.numberOfFolds = numberOfFolds

    def getSplits(self):
        trainSamplesLst = []
        testSamplesLst = []
        trainLabelsLst = []
        testLabelsLst = []
        if self.stratified:
            kFold = model_selection.StratifiedKFold(self.labels, n_folds=self.numberOfFolds, shuffle=False, random_state=None)
        else:
            kFold = model_selection.KFold(len(self.labels), n_folds=self.numberOfFolds, shuffle=False, random_state=None)

        for train_index, test_index in kFold:
            trainSamplesLst.append(self.samples[train_index, :])
            testSamplesLst.append(self.samples[test_index, :])
            trainLabelsLst.append(self.labels[train_index])
            testLabelsLst.append(self.labels[test_index])
        return trainSamplesLst, trainLabelsLst, testSamplesLst, testLabelsLst


