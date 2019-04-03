__author__ = 'viktor'
from .classifierCommon import *
def getHyperParamsStationaryUnscaled():
    return {
        'border': {
            cILVQ:  {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 9, 'metricLearnRate': 0.01, 'insertionTimingThresh': 10},
            cILVQClosest: {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 9, 'metricLearnRate': 0.01,                'insertionTimingThresh': 10},
            cILVQCluster: {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 9, 'metricLearnRate': 0.01,                    'insertionTimingThresh': 10},
            cILVQVoronoi: {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 9, 'metricLearnRate': 0.01,                    'insertionTimingThresh': 10},
            'ISVM':  {'C': 2**8, 'sigma': 50},
            'LASVM': {'C': 10000, 'kGamma': 0.04},
            'ORF':   {'numTrees': 10, 'numRandomTests': 250, 'counterThreshold': 20},
            'IELM':  {'numHiddenNeurons': 150},
            'LPP':   {'classifierPerChunk': 3},
            'SGD':   {'eta0': 0.005},
            'LVGB':  {'splitConfidence': 0.25, 'tieThresh': 0.15, 'gracePeriod': 25, 'numClassifier': 10}},
        'borderGen': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 9, 'metricLearnRate': 0.01, 'insertionTimingThresh': 2},
            'ISVM':  {'C': 2**8, 'sigma': 50},
            'ORF':   {'numTrees': 10, 'numRandomTests': 250, 'counterThreshold': 20},
            'IELM':  {'numHiddenNeurons': 150},
            'LPP':   {'classifierPerChunk': 3},
            'SGD':   {'eta0': 0.005},
            'LVGB':  {'splitConfidence': 0.25, 'tieThresh': 0.15, 'gracePeriod': 25, 'numClassifier': 10}},
        'overlap': {
            'ILVQ': {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 0.05, 'metricLearnRate': 0.01,
                     'insertionTimingThresh': 30},
            'ISVM':  {'C': 2**4, 'sigma': 0.50},
            'LASVM': {'C': 10, 'kGamma': 0.07},
            'ORF':   {'numTrees': 10, 'numRandomTests': 250, 'counterThreshold': 50},
            'IELM':  {'numHiddenNeurons': 150},
            'LPP':   {'classifierPerChunk': 1},
            'SGD':   {'eta0': 0.0001},
            'LVGB':  {'splitConfidence': 0.05, 'tieThresh': 0.1, 'gracePeriod': 30, 'numClassifier': 10}},
        'overlapGen': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 0.05, 'metricLearnRate': 0.01, 'insertionTimingThresh': 30},
            'ISVM':  {'C': 2**4, 'sigma': 0.50},
            'ORF':   {'numTrees': 10, 'numRandomTests': 250, 'counterThreshold': 50},
            'IELM':  {'numHiddenNeurons': 150},
            'LPP':   {'classifierPerChunk': 1},
            'SGD':   {'eta0': 0.0001},
            'LVGB':  {'splitConfidence': 0.05, 'tieThresh': 0.1, 'gracePeriod': 30, 'numClassifier': 10}},
        'noise': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 0.05, 'metricLearnRate': 0.01, 'insertionTimingThresh': 30},
            'ISVM':  {'C': 2**4, 'sigma': 0.50},
            'ORF':   {'numTrees': 10, 'numRandomTests': 250, 'counterThreshold': 50},
            'IELM':  {'numHiddenNeurons': 150},
            'LPP':   {'classifierPerChunk': 1},
            'SGD':   {'eta0': 0.0001},
            'LVGB':  {'splitConfidence': 0.05, 'tieThresh': 0.1, 'gracePeriod': 30, 'numClassifier': 10}},
        'coil': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 0.003, 'metricLearnRate': 0.03, 'insertionTimingThresh': 1, 'protoAdds': 5},
            'ISVM':  {'C': 2**10, 'sigma': 0.05},
            'LASVM': {'C': 100, 'kGamma': 20},
            'ORF':   {'numTrees': 50, 'numRandomTests': 250, 'counterThreshold': 5},
            'IELM':  {'numHiddenNeurons': 200},
            'LPP':   {'classifierPerChunk': 4},
            'SGD':   {'eta0': 0.08},
            'LVGB':  {'splitConfidence': 0.45, 'tieThresh': 0.4, 'gracePeriod': 15, 'numClassifier': 10}},
        'outdoor': {
            'ILVQ': {'netType': 'GMLVQ', 'activFct': 'linear', 'learnRateInitial': 0.0005, 'metricLearnRate': 0.0005,
                     'insertionTimingThresh': 1, 'protoAdds': 5},
            cILVQClosest: {'netType': 'GLVQ', 'activFct': 'linear', 'learnRateInitial': 0.0005, 'metricLearnRate': 0.0005,
                    'insertionTimingThresh': 1, 'protoAdds': 3},
            cILVQCluster: {'netType': 'GLVQ', 'activFct': 'linear', 'learnRateInitial': 0.0005, 'metricLearnRate': 0.0005,
                    'insertionTimingThresh': 1, 'protoAdds': 3},
            cILVQVoronoi: {'netType': 'GLVQ', 'activFct': 'linear', 'learnRateInitial': 0.0005, 'metricLearnRate': 0.0005,
                    'insertionTimingThresh': 1, 'protoAdds': 3},
            'ISVM':  {'C': 2**7, 'sigma': 0.05},
            'LASVM': {'C': 50, 'kGamma': 5},
            'ORF':   {'numTrees': 30, 'numRandomTests': 250, 'counterThreshold': 30},
            'IELM':  {'numHiddenNeurons': 200},
            'LPP':   {'classifierPerChunk': 5},
            'SGD':   {'eta0': 0.02},
            'LVGB':  {'splitConfidence': 0.4, 'tieThresh': 0.25,'gracePeriod': 40, 'numClassifier': 10}},
        'USPS': {
            'ILVQ':  {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 90, 'metricLearnRate': 0.01, 'insertionTimingThresh': 10},
            'ISVM':  {'C': 2**2, 'sigma': 50},
            'LASVM': {'C': 100, 'kGamma': 0.01},
            'ORF':   {'numTrees': 30, 'numRandomTests': 250, 'counterThreshold': 19},
            'IELM':  {'numHiddenNeurons': 400},
            'LPP':   {'classifierPerChunk': 5},
            'SGD':   {'eta0': 0.005},
            'LVGB':  {'splitConfidence': 0.4, 'tieThresh': 0.3, 'gracePeriod': 85, 'numClassifier': 10}},
        'DNA': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'linear', 'learnRateInitial': 0.001, 'metricLearnRate': 0.004, 'insertionTimingThresh': 500},
            'ISVM':  {'C': 2**3, 'sigma': 50},
            'LASVM': {'C': 10, 'kGamma': 0.01},
            'ORF':   {'numTrees': 20, 'numRandomTests': 500, 'counterThreshold': 20},
            'IELM':  {'numHiddenNeurons': 400},
            'LPP':   {'classifierPerChunk': 5},
            'SGD':   {'eta0': 0.01},
            'LVGB':  {'splitConfidence': 0.05, 'tieThresh': 0.07, 'gracePeriod': 50, 'numClassifier': 10}},
        'isolet': {
            'ILVQ':  {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 110,  'insertionTimingThresh': 100},
            'ISVM':  {'C': 2**8, 'sigma': 500},
            'LASVM': {'C': 100, 'kGamma': 0.005},
            'ORF':   {'numTrees': 100, 'numRandomTests': 300, 'counterThreshold': 75},
            'IELM':  {'numHiddenNeurons': 500},
            'LPP':   {'classifierPerChunk': 5},
            'SGD':   {'eta0': 0.005},
            'LVGB':  {'splitConfidence': 0.35, 'tieThresh': 0.35, 'gracePeriod': 160, 'numClassifier': 10}},
        'letter': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 20, 'metricLearnRate': 0.03, 'insertionTimingThresh': 1, 'protoAdds': 1},
            'ISVM':  {'C': 2**5, 'sigma': 40},
            'LASVM': {'C': 100, 'kGamma': 0.03},
            'ORF':   {'numTrees': 35, 'numRandomTests': 300, 'counterThreshold': 11},
            'IELM':  {'numHiddenNeurons': 200},
            'LPP':   {'classifierPerChunk': 5},
            'SGD':   {'eta0': 0.001},
            'LVGB':  {'splitConfidence': 0.45, 'tieThresh': 0.2, 'gracePeriod': 25, 'numClassifier': 10}},
        'gisette': {
            'ILVQ':  {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 100000000, 'insertionTimingThresh': 10},
            'ISVM':  {'C': 2**2, 'sigma': 500000000},
            'LASVM': {'C': 100, 'kGamma': 0.000000001},
            'ORF':   {'numTrees': 20, 'numRandomTests': 300, 'counterThreshold': 100},
            'IELM':  {'numHiddenNeurons': 500},
            'LPP':   {'classifierPerChunk': 3},
            'SGD':   {'eta0': 0.3},
            'LVGB':  {'splitConfidence': 0.3, 'tieThresh': 0.1, 'gracePeriod': 300, 'numClassifier': 5}},
        'mnist': {
            'ILVQ':  {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 3500000, 'insertionTimingThresh': 10},
            'ISVM':  {'C': 2**7, 'sigma': 100000000},
            'LASVM': {'C': 100, 'kGamma': 0.0000002},
            'ORF':   {'numTrees': 50, 'numRandomTests': 300, 'counterThreshold': 100},
            'IELM':  {'numHiddenNeurons': 500},
            'LPP':   {'classifierPerChunk': 3},
            'SGD':   {'eta0': 0.005},
            'LVGB':  {'splitConfidence': 0.4, 'tieThresh': 0.2, 'gracePeriod': 220, 'numClassifier': 5}},
        'satImage': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 0.001, 'metricLearnRate': 0.01, 'insertionTimingThresh': 2},
            'ISVM':  {'C': 2**0, 'sigma': 1150},
            'ORF':   {'numTrees': 20, 'numRandomTests': 100, 'counterThreshold': 5},
            'IELM':  {'numHiddenNeurons': 65},
            'LPP':   {'classifierPerChunk': 7},
            'SGD':   {'eta0': 0.0004},
            'LVGB':  {'splitConfidence': 0.4, 'tieThresh': 0.3, 'gracePeriod': 80, 'numClassifier': 10}},
        'penDigits': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 4723, 'metricLearnRate': 0.03, 'insertionTimingThresh': 3},
            'ISVM':  {'C': 2**10, 'sigma': 7500},
            'ORF':   {'numTrees': 20, 'numRandomTests': 100, 'counterThreshold': 8},
            'IELM':  {'numHiddenNeurons': 115},
            'LPP':   {'classifierPerChunk': 6},
            'SGD':   {'eta0': 0.00005},
            'LVGB':  {'splitConfidence': 0.4, 'tieThresh': 0.15, 'gracePeriod': 10, 'numClassifier': 10}},
        'HAR': {
            'ILVQ':  {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 22, 'insertionTimingThresh': 1000},
            'ISVM':  {'C': 2**7, 'sigma': 75},
            'ORF':   {'numTrees': 20, 'numRandomTests': 100, 'counterThreshold': 22},
            'IELM':  {'numHiddenNeurons': 95},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.08},
            'LVGB':  {'splitConfidence': 0.35, 'tieThresh': 0.1, 'gracePeriod': 40, 'numClassifier': 10}},
        'news20': {
            'ILVQ':  {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 4500, 'metricLearnRate': 0.01, 'insertionTimingThresh': 500},
            'ISVM':  {'C': 2**8, 'sigma': 0.75},
            'ORF':   {'numTrees': 10, 'numRandomTests': 500, 'counterThreshold': 13},
            'IELM':  {'numHiddenNeurons': 90},
            'LPP':   {'classifierPerChunk': 6},
            'SGD':   {'eta0': 0.002},
            'LVGB':  {'splitConfidence': 0.25, 'tieThresh': 0.15, 'gracePeriod': 22, 'numClassifier': 20}},
        'covTypeTT': {
            'ILVQ': {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 0.5, 'metricLearnRate': 0.01,
                     'insertionTimingThresh': 250},
            'ISVM': {'C': 2 ** 0, 'sigma': 1.7},
            #'LASVM': {'C': 10, 'kGamma': 0.000007},
            'LASVM': {'C': 10, 'kGamma': 15},
            'ORF': {'numTrees': 10, 'numRandomTests': 100, 'counterThreshold': 25},
            'IELM': {'numHiddenNeurons': 50},
            'LPP': {'classifierPerChunk': 3},
            'SGD':   {'eta0': 0.0001},
            'LVGB': {'splitConfidence': 0.45, 'tieThresh': 0.1, 'gracePeriod': 200, 'numClassifier': 5}},
        'susy': {
            'ILVQ': {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 2, 'metricLearnRate': 0.02,
                     'insertionTimingThresh': 1000},
            'ISVM': {'C': 2 ** 0, 'sigma': 1.7},
            'LASVM': {'C': 10, 'kGamma': 0.07},
            'ORF': {'numTrees': 10, 'numRandomTests': 100, 'counterThreshold': 1000},
            'IELM': {'numHiddenNeurons': 100},
            'LPP': {'classifierPerChunk': 1},
            'SGD': {'eta0': 0.001},
            'LVGB': {'splitConfidence': 0.45, 'tieThresh': 0.1, 'gracePeriod': 200, 'numClassifier': 5}},
    }




def getHyperParamsNonStationaryUnscaled():
    return {
        'elec': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial':1, 'metricLearnRate': 0.005, 'insertionTimingThresh': 50},
            'ISVM':  {'C': 2**7, 'sigma': 5000000},
            'LASVM': {'C': 10, 'kGamma': 50},
            'ORF':   {'numTrees': 20, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 88},
            'LPP':   {'classifierPerChunk': 5},
            'SGD':   {'eta0':  1.05},
            'LVGB':  {'splitConfidence': 0.004, 'tieThresh': 0.1, 'gracePeriod': 200, 'numClassifier': 10}},
        'weather': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 148, 'insertionTimingThresh': 23},
            'ISVM':  {'C': 2**0, 'sigma': 500},
            'ORF':   {'numTrees': 30, 'numRandomTests': 100, 'counterThreshold': 10},
            'IELM':  {'numHiddenNeurons': 120},
            'LPP':   {'classifierPerChunk': 7},
            'SGD':   {'eta0': 0.001},
            'LVGB':  {'splitConfidence': 0.3, 'tieThresh': 0.1, 'gracePeriod': 200, 'numClassifier': 10}},
        'sea': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 1.8, 'metricLearnRate': 0.01, 'insertionTimingThresh': 100000},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 20, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 57},
            'LPP':   {'classifierPerChunk': 7},
            'SGD':   {'eta0': 0.002},
            'LVGB':  {'splitConfidence': 0.003, 'tieThresh': 0.1, 'gracePeriod': 200, 'numClassifier': 10}},
        'souza2CDT': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 5.2, 'metricLearnRate': 0.01, 'insertionTimingThresh': 55},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 40, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 50},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.008},
            'LVGB':  {'splitConfidence': 0.1, 'tieThresh': 0.1, 'gracePeriod': 200, 'numClassifier': 10}},
        'souza4CREV1': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 9, 'metricLearnRate': 0.01, 'insertionTimingThresh': 27},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 40, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 50},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.13},
            'LVGB':  {'splitConfidence': 0.35, 'tieThresh': 0.1, 'gracePeriod': 200, 'numClassifier': 10}},
        'souzaFG2C2D': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 1.9, 'metricLearnRate': 0.01, 'insertionTimingThresh': 15},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 20, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 50},
            'LPP':   {'classifierPerChunk': 5},
            'SGD':   {'eta0': 0.01},
            'LVGB':  {'splitConfidence': 0.1, 'tieThresh': 0.1, 'gracePeriod': 200, 'numClassifier': 10}},
        'souzaGears2C2D': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 0.42, 'metricLearnRate': 0.01, 'insertionTimingThresh': 5},#1
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 40, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 100},
            'LPP':   {'classifierPerChunk': 3},
            'SGD':   {'eta0':  0.05},
            'LVGB':  {'splitConfidence': 0.0004, 'tieThresh': 0.1, 'gracePeriod': 200, 'numClassifier': 10}},
        'souza2CHT': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 20, 'metricLearnRate': 0.01, 'insertionTimingThresh': 5}, #200
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 40, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 50},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.009},
            'LVGB':  {'splitConfidence': 0.1, 'tieThresh': 0.1, 'gracePeriod': 200, 'numClassifier': 10}},
        'hyperplaneSlow': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 0.12, 'metricLearnRate': 0.01, 'insertionTimingThresh': 5},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 40, 'numRandomTests': 100, 'counterThreshold': 11},
            'IELM':  {'numHiddenNeurons': 50},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.06},
            'LVGB':  {'splitConfidence': 0.35, 'tieThresh': 0.3, 'gracePeriod': 200, 'numClassifier': 10}},
        'movingRBF': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 4.3, 'metricLearnRate': 0.005, 'insertionTimingThresh': 5},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 40, 'numRandomTests': 100, 'counterThreshold': 10},
            'IELM':  {'numHiddenNeurons': 50},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.1},
            'LVGB':  {'splitConfidence': 0.44, 'tieThresh': 0.3, 'gracePeriod': 100, 'numClassifier': 10}},
        'interRBF': {
            'ILVQ': {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 4.3, 'metricLearnRate': 0.005,
                     'insertionTimingThresh': 5},
            'ISVM': {'C': 2 ** 0, 'sigma': 1.7},
            'ORF': {'numTrees': 40, 'numRandomTests': 100, 'counterThreshold': 10},
            'IELM': {'numHiddenNeurons': 50},
            'LPP': {'classifierPerChunk': 8},
            'SGD': {'eta0': 0.1},
            'LVGB': {'splitConfidence': 0.44, 'tieThresh': 0.3, 'gracePeriod': 100, 'numClassifier': 10}},
        'keystroke': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 0.25, 'metricLearnRate': 0.02, 'insertionTimingThresh': 2},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 30, 'numRandomTests': 100, 'counterThreshold': 10},
            'IELM':  {'numHiddenNeurons': 180},
            'LPP':   {'classifierPerChunk': 7},
            'SGD':   {'eta0': 0.004},
            'LVGB':  {'splitConfidence': 0.35, 'tieThresh': 0.1, 'gracePeriod': 200, 'numClassifier': 10}},
        'spam': {
            'ILVQ':  {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 20000, 'metricLearnRate': 0.01, 'insertionTimingThresh': 200},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 20, 'numRandomTests': 100, 'counterThreshold': 4},
            'IELM':  {'numHiddenNeurons': 100},
            'LPP':   {'classifierPerChunk': 1},
            'SGD':   {'eta0': 0.02},
            'LVGB':  {'splitConfidence': 0.1, 'tieThresh': 0.1, 'gracePeriod': 200, 'numClassifier': 10}},
        'covType': {
            'ILVQ':  {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 5000000, 'metricLearnRate': 0.2, 'insertionTimingThresh': 50},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 10, 'numRandomTests': 100, 'counterThreshold': 10},
            'IELM':  {'numHiddenNeurons': 100},
            'LPP':   {'classifierPerChunk': 1},
            'SGD':   {'eta0': 0.36},
            'LVGB':  {'splitConfidence': 0.45, 'tieThresh': 0.1, 'gracePeriod': 200, 'numClassifier': 5}},
    }
