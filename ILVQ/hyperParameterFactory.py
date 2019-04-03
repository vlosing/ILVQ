from .hyperParamsScaled import getHyperParamsStationaryScaled, getHyperParamsNonStationaryScaled
from .hyperParamsUnscaled import getHyperParamsStationaryUnscaled, getHyperParamsNonStationaryUnscaled
from .classifierCommon import *
import logging
import os
import json
import copy

def getDefaultHyperParams(classifierName):
    params = {}
    params[cILVQ] = {'classifierType': 'LVQ', 'netType': 'GLVQ', 'activFct': 'logistic', 'retrainFreq': 0,
         'learnRatePerProto': True, 'learnRateInitial': 1, 'learnRateAnnealingSteps': 5000,
         'metricLearnRate': 0.03,
         'maxSize': 200, 'name': 'NoName', 'LIRAMLVQDimensions': '1',
         'insertionStrategy': 'samplingCost', 'insertionTiming': 'errorCount', 'insertionTimingThresh': 30,
         'sampling': 'random', 'protoAdds': 1, 'deletionStrategy': None, 'driftStrategy': None}

    params[cILVQCluster] = copy.deepcopy(params[cILVQ])
    params[cILVQCluster]['insertionStrategy'] = 'Cluster'

    params[cILVQClosest] = copy.deepcopy(params[cILVQ])
    params[cILVQClosest]['insertionStrategy'] = 'Closest'

    params[cILVQVoronoi] = copy.deepcopy(params[cILVQ])
    params[cILVQVoronoi]['insertionStrategy'] = 'Voronoi'

    return params[classifierName]

def getHyperParams(datasetName, classifierName, scaled, useAutomaticHyperParamFiles=False):
    params = getDefaultHyperParams(classifierName)
    if classifierName in ['LPPNSE', 'GNB', 'SAMKNN', 'KNNPaw', 'DACC', 'HoeffAdwin', 'LVGB', 'LVGBSAM']:
        return params

    specificParams = {}
    if scaled:
        datasetSpecificParams = getHyperParamsStationaryScaled()
        datasetSpecificParams2 = getHyperParamsNonStationaryScaled()
    else:
        datasetSpecificParams = getHyperParamsStationaryUnscaled()
        datasetSpecificParams2 = getHyperParamsNonStationaryUnscaled()
    fileParams = datasetSpecificParams.copy()
    fileParams.update(datasetSpecificParams2)
    if datasetName in fileParams and classifierName in fileParams[datasetName]:
        specificParams = fileParams[datasetName][classifierName]
        logging.info('%s: specific params from manual hypFile.' %classifierName)
    if specificParams == {}:
        logging.info('%s: No dataset specific hyperparams.' %classifierName)

    for key in specificParams.keys():
        params[key] = specificParams[key]

    return params
