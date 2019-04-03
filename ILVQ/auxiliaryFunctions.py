import numpy as np
from sklearn.metrics import accuracy_score
from .DataSetFactory import isStationary
from sklearn.neighbors import KNeighborsClassifier

from .core import getBaseClassifier

import logging
import time
def getTrainSetCfg(dataSetName):
    trainSetSize = 1
    if isStationary(dataSetName):
        #trainOrder = 'original'
        trainOrder = 'chunksRandom'
        splitType = None
        shuffle = False
    else:
        #trainOrder = 'original'
        trainOrder = 'chunksRandom'
        splitType = 'simple'
        #splitType = None
        shuffle = False

    trainingSetCfg = {'dsName': dataSetName, 'splitType': splitType, 'folds': 2, 'trainOrder': trainOrder, 'stratified': False,
                      'shuffle': shuffle, 'chunkSize': 10, 'trainSetSize': trainSetSize}
    return trainingSetCfg

def KNearestLeaveOneOut(samples, labels):
    accuracies = []
    classifier = KNeighborsClassifier(n_neighbors=5)
    for i in range(len(labels)):
        trainSamples = np.delete(samples, i, 0)
        trainLabels = np.delete(labels, i, 0)
        classifier.fit(trainSamples, trainLabels)
        predLabel = classifier.predict(samples[i, :].reshape(1, -1))
        accuracies.append(predLabel == labels[i])
    print(np.mean(accuracies))


def getClassifier(classifierName, classifierParams, listener=[]):
    if classifierName == 'meta':
        return MetaClassifier(baseClassifier=classifierParams['classifierName'], growingParam=classifierParams['growingParam'], numClassifier=classifierParams['numClassifier'], SEM=classifierParams['SEM'], listener=listener)
    else:
        return getBaseClassifier(classifierName, classifierParams, listener=[])

def trainClassifierOnline(classifierName, classifiers, classifierParams, dataSetIDStr, X, Y, classes, metaData, maxFloatPrecision=None, listener=[]):
    if classifierName in classifiers:
        classifier = classifiers[classifierName]
    elif classifierName in moaClassifier:
        predictedLabels, complexity, complexityNumParameterMetric, dummy, runTime, ramHours, dummy, ensembleLabels, nInstances, runTimes = \
            trainMOAAlgorithmOnline(classifierName, classifierParams, dataSetIDStr, X, Y, len(Y)/20, metaData=metaData,
                                    maxFloatPrecision=maxFloatPrecision, writeStatistics=True)
        return predictedLabels, complexity, complexityNumParameterMetric, classifiers, runTime, ramHours, ensembleLabels, nInstances, runTimes
    else:
        classifier = getClassifier(classifierName, classifierParams, listener=listener)
    classifiers[classifierName] = classifier
    startTime = time.time()
    '''if metaData is not None:
        X2 = []
        for sample in X:
            # print(sample)
            X2.append(np.array([int(feature) for feature in sample]))
        X2 = np.array(X2)
        predictedLabels, complexity, complexityNumParameterMetric = classifier.trainOnline(X2, Y, classes, metaData=metaData)
    else:'''
    predictedLabels, complexity, complexityNumParameterMetric = classifier.trainOnline(X, Y, classes, metaData=metaData, chunkSize=1)
    runTime = time.time() - startTime
    return predictedLabels, complexity, complexityNumParameterMetric, classifiers, runTime, 0, None, None, None

def updateClassifierEvaluationsOnline(classifierEvaluations, classifierName, iterationIdx, labels, predictedTestLabels, complexity, complexityNumParameterMetric, totalIdx, runTime, ramHours, nInstances, runTimes, storeLabels=False, ensembleLabels=None):

    accuracy = accuracy_score(labels, predictedTestLabels)
    logging.info("error %.2f" % ((1-accuracy)*100))
    weight = len(predictedTestLabels)
    data = classifierEvaluations['values']
    if not classifierName in data:
        data[classifierName] = []
    if len(data[classifierName]) <= iterationIdx:
        data[classifierName].append({})
        data[classifierName][iterationIdx]['accuracies'] = []
        data[classifierName][iterationIdx]['complexity'] = []
        data[classifierName][iterationIdx]['complexitiesNumParamMetric'] = []
        data[classifierName][iterationIdx]['weights'] = []
        data[classifierName][iterationIdx]['idx'] = []
        data[classifierName][iterationIdx]['Y'] = []
        data[classifierName][iterationIdx]['YPred'] = []
        data[classifierName][iterationIdx]['YPredEnsemble'] = []

    data[classifierName][iterationIdx]['accuracies'].append(accuracy)
    data[classifierName][iterationIdx]['complexity'].append(complexity)
    data[classifierName][iterationIdx]['complexitiesNumParamMetric'].append(complexityNumParameterMetric)
    data[classifierName][iterationIdx]['weights'].append(weight)
    data[classifierName][iterationIdx]['idx'].append(totalIdx)
    if storeLabels:
        data[classifierName][iterationIdx]['Y'] += labels.tolist()
        data[classifierName][iterationIdx]['YPred'] += predictedTestLabels.tolist()
    if ensembleLabels is not None:
        data[classifierName][iterationIdx]['YPredEnsemble'] = ensembleLabels.tolist()
    data[classifierName][iterationIdx]['runTime'] = runTime
    data[classifierName][iterationIdx]['ramHours'] = ramHours
    if nInstances is not None:
        data[classifierName][iterationIdx]['nInstances'] = nInstances.tolist()
    if runTimes is not None:
        data[classifierName][iterationIdx]['runTimes'] = runTimes.tolist()

    return classifierEvaluations

def getSummaryString(evaluationResults, groupByClassifier=True, extended=False):

    summaryString = ''
    resultsGrouped = {}
    if groupByClassifier:
        classifierSummaries = {}
        for result in evaluationResults:
            if not result['classifierName'] in classifierSummaries:
                classifierSummaries[result['classifierName']] = []
            classifierSummaries[result['classifierName']].append(result)

            if not result['classifierName'] in resultsGrouped:
                resultsGrouped[result['classifierName']] = {}
                resultsGrouped[result['classifierName']]['errorRate'] = []
                resultsGrouped[result['classifierName']]['runTime'] = []
                resultsGrouped[result['classifierName']]['ramHours'] = []
            resultsGrouped[result['classifierName']]['errorRate'].append(100 - result['acc'])
            resultsGrouped[result['classifierName']]['runTime'].append(result['runTime'])
            resultsGrouped[result['classifierName']]['ramHours'].append(result['ramHours'])
        print(evaluationResults)
        print(classifierSummaries)
        for classifier in classifierSummaries:
            summaryString += '%s \n' % classifier
            for result in classifierSummaries[classifier]:
                summaryString += '%s acc %.2f  error %.2f runTime %.2f ramHours %.7f \n' % (
                result['datasetName'], result['acc'], 100 - result['acc'], result['runTime'], result['ramHours'])
            if extended:
                summaryString += 'error  \n'
                for result in classifierSummaries[classifier]:
                    summaryString += '%.2f \n' % (100 - result['acc'])
                summaryString += 'runTime \n'
                for result in classifierSummaries[classifier]:
                    summaryString += '%.2f \n' % (result['runTime'])
                summaryString += 'ramHours \n'
                for result in classifierSummaries[classifier]:
                    summaryString += '%.7f \n' % (result['ramHours'])

    for classifierName in resultsGrouped:
        summaryString += '%s avg error %.2f runTime total %.2f avg %.2f, ' \
                         'ramHours total %.7f avg %.7f \n' % (classifierName,
                                                           np.mean(resultsGrouped[classifierName]['errorRate']),
                                                           np.sum(resultsGrouped[classifierName]['runTime']),
                                                           np.mean(resultsGrouped[classifierName]['runTime']),
                                                           np.sum(resultsGrouped[classifierName]['ramHours']),
                                                           np.mean(resultsGrouped[classifierName]['ramHours']))
    return summaryString

def summarizeDatasetEvaluations(classifierEvaluations):
    results = []
    data = classifierEvaluations['values']
    for classifierName in data.keys():
        iterationIdx = 0
        avg = np.round(np.average(data[classifierName][iterationIdx]['accuracies'], weights=data[classifierName][iterationIdx]['weights']), 4)
        result = {'datasetName': classifierEvaluations['meta']['dataSetName'], 'classifierName': classifierName, 'acc': 100 * avg, 'complexity': data[classifierName][iterationIdx]['complexity'][-1],
                  'runTime': data[classifierName][iterationIdx]['runTime'], 'ramHours': data[classifierName][iterationIdx]['ramHours']}
        results.append(result)
    return results

def trainClassifier(classifierName, classifierParams, dataSetIDStr, trainFeatures, trainLabels, testFeatures, testLabels, metaData, evaluationStepSize, streamSetting, ):
    if classifierName in ['LVGB', 'KNNPaw', 'HoeffAdwin', 'DACC', 'SAMKNNJ', 'HoeffTreeJ']:
        return trainMOAAlgorithm(classifierName, classifierParams, dataSetIDStr, trainFeatures, trainLabels, testFeatures, testLabels, evaluationStepSize, streamSetting)
    else:
        classifier = getClassifier(classifierName, classifierParams)
    classes = np.unique(np.append(trainLabels, testLabels))
    allPredictedTestLabels, allPredictedTrainLabels, complexities, complexityNumParameterMetric, splitIndices = classifier.trainAndEvaluate(trainFeatures, trainLabels,
                                    testFeatures,
                                    evaluationStepSize,
                                    classes,
                                    streamSetting=streamSetting,
                                    metaData=metaData)
    return allPredictedTestLabels, allPredictedTrainLabels, complexities, complexityNumParameterMetric, splitIndices

def updateClassifierEvaluations(classifierEvaluations, classifierName, trainLabels, testLabels, allPredictedTestLabels, allPredictedTrainLabels, complexities, complexityNumParameterMetric,
                                criterion, streamSetting, splitIndices):
    if not streamSetting:
        testAccuracyScores = []

        for i in range(len(allPredictedTestLabels)):
            allPredictedTestLabels[i] = [int(i) for i in allPredictedTestLabels[i]]

        for classifiedLabels in allPredictedTestLabels:
            classifiedLabels = np.array(classifiedLabels).astype(np.int)
            testAccuracyScores.append(accuracy_score(testLabels, classifiedLabels))

        print(testAccuracyScores, np.mean(testAccuracyScores))
        print(complexities, complexityNumParameterMetric)


        classifierEvaluations['values'][classifierName] = {}
        classifierEvaluations['values'][classifierName]['Y'] = testLabels.tolist()
        classifierEvaluations['values'][classifierName]['YPred'] = allPredictedTestLabels
        classifierEvaluations['values'][classifierName]['times'] = splitIndices.tolist()
        classifierEvaluations['values'][classifierName]['times'].append(len(trainLabels))
    else:
        trainPredictionAccuracyScores = []
        idx = 0
        weights = []
        for classifiedLabels in allPredictedTrainLabels:
            trainPredictionAccuracyScores.append(accuracy_score(trainLabels[idx:idx+len(classifiedLabels)], classifiedLabels))
            idx += len(classifiedLabels)
            weights.append(len(classifiedLabels))
        if classifierName in ['LPP', 'LPPNSE', 'IELM'] and len(trainPredictionAccuracyScores) > 0:
            trainPredictionAccuracyScores[0] = 0.5*trainPredictionAccuracyScores[1]
        meanAcc = np.average(trainPredictionAccuracyScores, weights=weights)
        print(np.round(trainPredictionAccuracyScores, 4), np.round(meanAcc, 4))
        print(complexities)
        if classifierName in classifierEvaluations['values']:
            if criterion in classifierEvaluations['values'][classifierName]:
                if 'trainPredictionAccuracies' in classifierEvaluations['values'][classifierName][criterion]:
                    classifierEvaluations['values'][classifierName][criterion]['trainPredictionAccuracies'].append(trainPredictionAccuracyScores)
                    classifierEvaluations['values'][classifierName][criterion]['finalACC'].append(meanAcc)
                else:
                    classifierEvaluations['values'][classifierName][criterion]['trainPredictionAccuracies'] = [trainPredictionAccuracyScores]
                    classifierEvaluations['values'][classifierName][criterion]['finalACC'] = [meanAcc]
            else:
                classifierEvaluations['values'][classifierName][criterion] = {}
                classifierEvaluations['values'][classifierName][criterion]['trainPredictionAccuracies'] = [trainPredictionAccuracyScores]
                classifierEvaluations['values'][classifierName][criterion]['finalACC']= [meanAcc]
        else:
            classifierEvaluations['values'][classifierName] = {}
            classifierEvaluations['values'][classifierName][criterion] = {}
            classifierEvaluations['values'][classifierName][criterion]['trainPredictionAccuracies'] = [trainPredictionAccuracyScores]
            classifierEvaluations['values'][classifierName][criterion]['finalACC'] = [meanAcc]

    if classifierName in classifierEvaluations['values']:
        if criterion in classifierEvaluations['values'][classifierName]:
            if 'complexities' in classifierEvaluations['values'][classifierName][criterion]:
                classifierEvaluations['values'][classifierName][criterion]['complexities'].append(complexities)
            else:
                classifierEvaluations['values'][classifierName][criterion]['complexities'] = [complexities]
        else:
            classifierEvaluations['values'][classifierName][criterion] = {}
            classifierEvaluations['values'][classifierName][criterion]['complexities'] = [complexities]
    else:
        classifierEvaluations['values'][classifierName] = {}
        classifierEvaluations['values'][classifierName][criterion] = {}
        classifierEvaluations['values'][classifierName][criterion]['complexities'] = [complexities]

    if classifierName in classifierEvaluations['values']:
        if criterion in classifierEvaluations['values'][classifierName]:
            if 'complexitiesNumParamMetric' in classifierEvaluations['values'][classifierName][criterion]:
                classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'].append(complexityNumParameterMetric)
                classifierEvaluations['values'][classifierName][criterion]['finalComplexitiesNumParamMetric'].append(complexityNumParameterMetric[-1])
            else:
                classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'] = [complexityNumParameterMetric]
                classifierEvaluations['values'][classifierName][criterion]['finalComplexitiesNumParamMetric'] = [complexityNumParameterMetric[-1]]
        else:
            classifierEvaluations['values'][classifierName][criterion] = {}
            classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'] = [complexityNumParameterMetric]
            classifierEvaluations['values'][classifierName][criterion]['finalComplexitiesNumParamMetric'] = [complexityNumParameterMetric[-1]]
    else:
        classifierEvaluations['values'][classifierName] = {}
        classifierEvaluations['values'][classifierName][criterion] = {}
        classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'] = [complexityNumParameterMetric]
        classifierEvaluations['values'][classifierName][criterion]['finalComplexitiesNumParamMetric'] = [complexityNumParameterMetric[-1]]
    return classifierEvaluations




