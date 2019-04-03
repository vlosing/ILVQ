import logging
from ILVQ.TrainTestSplitsManager import TrainTestSplitsManager
from ILVQ.hyperParameterFactory import getHyperParams
from ILVQ.auxiliaryFunctions import trainClassifier, updateClassifierEvaluations, getTrainSetCfg
from ILVQ.DataSetFactory import isStationary, getDataSet
from ILVQ.classifierCommon import *
from ILVQ.DataSet import *

from comparisonPlot import doComparisonPlot
import matplotlib.pyplot as plt
import json
import os

def permutateDataset(samples, labels):
    indices = np.random.permutation(len(labels))
    return samples[indices], labels[indices]

def doComparison(dataSetName, classifierNames, iterations, criteria, permutate=False):
    trainSetCfg = getTrainSetCfg(dataSetName)
    dataSet = getDataSet(trainSetCfg['dsName'])
    dataSet.loggDSInformation()
    streamSetting = not isStationary(dataSetName)
    logging.info('stream setting' if streamSetting  else 'train/test setting')
    trainSamples = dataSet.samples
    testSamples = dataSet.testSamples
    trainTestSplitsManager = TrainTestSplitsManager(trainSamples, dataSet.labels,
                                                 dataOrder=trainSetCfg['trainOrder'], chunkSize=trainSetCfg['chunkSize'],
                                                 shuffle=trainSetCfg['shuffle'], stratified=trainSetCfg['stratified'],
                                                 splitType=trainSetCfg['splitType'], numberOfFolds=trainSetCfg['folds'],
                                                 trainSetSize=trainSetCfg['trainSetSize'], testSamples=testSamples, testLabels=dataSet.testLabels)
    classifierEvaluations = {'meta':{'dataSetName':dataSetName, 'numTrainSamples': len(dataSet.labels), 'streamSetting':streamSetting}, 'values':{}}

    for iteration in np.arange(iterations):
        logging.info('iteration %d/%d'% (iteration+1, iterations))
        trainTestSplitsManager.generateSplits()
        trainSamples = trainTestSplitsManager.TrainSamplesLst[0]
        trainLabels = trainTestSplitsManager.TrainLabelsLst[0]

        if permutate:
            trainSamples, trainLabels = permutateDataset(trainSamples, trainLabels)
        numTrainSamples = len(trainLabels)

        logging.info('train-samples %d' % numTrainSamples)
        if iteration == 0:
            hyperParams = {}
            for classifierName in classifierNames:
                hyperParams[classifierName] = getHyperParams(trainSetCfg['dsName'], classifierName, 0, False)

        for criterion in criteria:
            chunkSize = criterion
            splits = int(np.ceil(numTrainSamples / float(chunkSize)))
            logging.info('evaluationStepSize %d splits %d' % (chunkSize, splits))

            for classifierName in classifierNames:
                classifierParams = hyperParams[classifierName]
                logging.info(classifierName + ' ' + str(classifierParams))
                dataSetIDStr = DataSet.getIDStr(dataSetName, permutate)
                allPredictedTestLabels, allPredictedTrainLabels, complexities, complexityNumParameterMetric, splitIndices = trainClassifier(classifierName, classifierParams, dataSetIDStr, trainSamples, trainLabels, trainTestSplitsManager.TestSamplesLst[0], trainTestSplitsManager.TestLabelsLst[0], dataSet.metaData, chunkSize, streamSetting)


                classifierEvaluations = updateClassifierEvaluations(classifierEvaluations, classifierName, trainLabels, trainTestSplitsManager.TestLabelsLst[0], allPredictedTestLabels, allPredictedTrainLabels, complexities, complexityNumParameterMetric, criterion, streamSetting, splitIndices)

    json.encoder.FLOAT_REPR = lambda o: format(o, '.5f')
    dataSetIDStr = DataSet.getIDStr(dataSetName, permutate)
    evalDir = './tmp/'
    if not os.path.exists(evalDir):
        os.makedirs(evalDir)
    filePath = os.path.join(evalDir, dataSetIDStr + 'evaluations.json')
    json.dump(classifierEvaluations, open(filePath, 'w'))
    #doComparisonPlot(dataSetName, classifierNames, dataSetIDStr, filePath=filePath)
    #plt.show()

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    #randomState = 0
    #np.random.seed(randomState)

    criteria = [10000]
    classifierNames = [cILVQ]


    iterations = 1
    permutate=True
    useAutomaticHyperParamFiles=False

    ###stationary###
    doComparison(dsBorder, classifierNames, iterations,  criteria, permutate=True)
    doComparison(dsOverlap, classifierNames, iterations, criteria, permutate=True)
    doComparison('outdoor', classifierNames, iterations, criteria, permutate=True)
    doComparison('coil', classifierNames, iterations, criteria, permutate=True)



