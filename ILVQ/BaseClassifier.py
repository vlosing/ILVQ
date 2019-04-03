__author__ = 'vlosing'
# Copyright (C)
# Honda Research Institute Europe GmbH
# Carl-Legien-Str. 30
# 63073 Offenbach/Main
# Germany
#
# UNPUBLISHED PROPRIETARY MATERIAL.
# ALL RIGHTS RESERVED.


import numpy as np
from sklearn.metrics import log_loss
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import time

class BaseClassifier(object):
    def __init__(self):
        pass

    def fit(self, samples, labels, epochs=1):
        raise NotImplementedError()

    def partial_fit(self, samples, labels, classes, metaData=None):
        raise NotImplementedError()

    def testTrainLearning(self, samples, labels, classes, metaData=None):
        raise NotImplementedError()

    #def alternateFitPredict(self, samples, labels, classes, metaData=None):
    #    raise NotImplementedError()

    def predict(self, samples):
        raise NotImplementedError()

    def predict_proba(self, samples, classes=None):
        raise NotImplementedError()

    def getInfos(self):
        raise NotImplementedError()

    def getComplexity(self):
        raise NotImplementedError()

    def getComplexityNumParameterMetric(self):
        raise NotImplementedError()

    @staticmethod
    def getLabelAccuracy(groundTruthY, predictedY):
        lblAccuracies = []
        for label in np.unique(groundTruthY):
            labelIndices = np.where(groundTruthY == label)[0].astype(int)
            labelAcc = np.sum(np.array(predictedY)[labelIndices] == np.array(groundTruthY)[labelIndices])
            labelAcc /= float(len(labelIndices))
            lblAccuracies.append([label, labelAcc])
        return lblAccuracies


    def getAccuracy(self, samples, labels, scoreType='acc', labelSpecific=False):
        if len(samples) == 0:
            logging.warning('no samples to predict')
        elif scoreType == 'acc':
            classifiedLabels = np.array(self.predict(samples))
            acc = np.sum(classifiedLabels == labels)/float(len(labels))
            if labelSpecific:
                BaseClassifier.getLabelAccuracy(classifiedLabels, labels)
                return acc
            else:
                return acc

        elif scoreType == 'logLoss':
            classifiedLabels = self.predict_proba(samples)
            return log_loss(labels, classifiedLabels, eps=1e-15, normalize=True, sample_weight=None)
        else:
            raise NameError('unknown scoreType ' + scoreType)

    def getCostValues(self, samples, labels):
        return -1

    def trainFromFileAndEvaluate(self, trainFeaturesFileName, trainLabelsFileName,
                                 testFeaturesFileName,
                                 evaluationStepSize, evalDstPathPrefix, splitTestfeatures):

        trainFeatures = pd.read_csv(trainFeaturesFileName, sep=' ', header=None, skiprows=1).values
        trainLabels = pd.read_csv(trainLabelsFileName, header=None, skiprows=1).values.ravel()


        if testFeaturesFileName != '':
            testFeatures = pd.read_csv(testFeaturesFileName, sep=' ', header=None, skiprows=1).values

        self.trainAndEvaluate(trainFeatures, trainLabels, testFeatures, evaluationStepSize, splitTestfeatures)

    def trainAndEvaluate(self, trainFeatures, trainLabels,
                        testFeatures,
                        evaluationStepSize, classes, streamSetting=False, metaData=None):
        tic = time.time()
        splitIndices = np.arange(evaluationStepSize, len(trainLabels), evaluationStepSize)
        trainFeaturesChunks = np.array_split(trainFeatures, splitIndices)
        trainLabelsChunks = np.array_split(trainLabels, splitIndices)
        complexities = []
        complexityNumParameterMetric = []
        allPredictedTrainLabels = []
        allPredictedTestLabels = []
        for i in range(len(trainFeaturesChunks)):
            print("chunk %d/%d %s" % (i+1, len(trainFeaturesChunks), datetime.now().strftime('%H:%M:%S')))
            if streamSetting:
                YTrainPred = self.partial_fit(trainFeaturesChunks[i], trainLabelsChunks[i], classes, metaData=metaData)
                allPredictedTrainLabels.append(YTrainPred)
            else:
                self.partial_fit(trainFeaturesChunks[i], trainLabelsChunks[i], classes, metaData=metaData)

            if len(testFeatures) > 0:
                predictedLabels = self.predict(testFeatures)
                allPredictedTestLabels.append(predictedLabels)
            complexities.append(self.getComplexity())
            complexityNumParameterMetric.append(self.getComplexityNumParameterMetric())
        logging.info('%.2f seconds' % round(time.time() - tic, 2))
        return allPredictedTestLabels, allPredictedTrainLabels, complexities, complexityNumParameterMetric, splitIndices


    def trainOnline(self, X, y, classes, metaData=None, chunkSize=1):
        if chunkSize == 1:
            predictedLabels = self.testTrainLearning(X, y, classes, metaData=metaData)
        else:
            predictedLabels = []
            for i in range(0, X.shape[0], chunkSize):
                if i % chunkSize == 0:
                    predictedLabels += list(self.predict(X[i:i + chunkSize, :]))
                    self.partial_fit(X[i:i + chunkSize, :], y[i:i + chunkSize], classes, metaData=metaData)
            predictedLabels = np.array(predictedLabels)
        return predictedLabels, self.getComplexity(), self.getComplexityNumParameterMetric()
