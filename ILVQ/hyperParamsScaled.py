import logging

def getHyperParamsStationaryScaled():
    return {
        'border': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 0.13, 'metricLearnRate': 0.01, 'insertionTimingThresh': 2},
            'ISVM':  {'C': 2**8, 'sigma': 0.75},
            'ORF':   {'numTrees': 10, 'numRandomTests': 250, 'counterThreshold': 10},
            'IELM':  {'numHiddenNeurons': 90},
            'LPP':   {'classifierPerChunk': 6},
            'SGD':   {'eta0': 0.003},
            'LVGB':  {'splitConfidence': 0.4}},
        'noise': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 0.13, 'metricLearnRate': 0.01, 'insertionTimingThresh': 30},
            'ISVM':  {'C': 2**8, 'sigma': 0.75},
            'ORF':   {'numTrees': 10, 'numRandomTests': 250, 'counterThreshold': 10},
            'IELM':  {'numHiddenNeurons': 90},
            'LPP':   {'classifierPerChunk': 6},
            'SGD':   {'eta0': 0.003},
            'LVGB':  {'splitConfidence': 0.16}},
        'overlap': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 0.1, 'metricLearnRate': 0.01, 'insertionTimingThresh': 30},
            'ISVM':  {'C': 2**4, 'sigma': 0.05},
            'ORF':   {'numTrees': 10, 'numRandomTests': 250, 'counterThreshold': 52},
            'IELM':  {'numHiddenNeurons': 70},
            'LPP':   {'classifierPerChunk': 4},
            'SGD':   {'eta0': 0.14},
            'LVGB':  {'splitConfidence': 0.16}},
        'coil': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 0.003, 'metricLearnRate': 0.03, 'insertionTimingThresh': 1, 'protoAdds': 5},
            'ISVM':  {'C': 2**10, 'sigma': 10},
            'ORF':   {'numTrees': 50, 'numRandomTests': 250, 'counterThreshold': 5},
            'IELM':  {'numHiddenNeurons': 200},
            'LPP':   {'classifierPerChunk': 4},
            'SGD':   {'eta0': 0.08},
            'LVGB':  {'splitConfidence': 0.45, 'tieThresh': 0.4, 'gracePeriod': 50}},
        'outdoor': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'linear', 'learnRateInitial': 0.07, 'metricLearnRate': 0.007, 'insertionTimingThresh': 1, 'protoAdds': 3},
            'ISVM':  {'C': 2**7, 'sigma': 8},
            'ORF':   {'numTrees': 30, 'numRandomTests': 200, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 450},
            'LPP':   {'classifierPerChunk': 5},
            'SGD':   {'eta0': 0.01},
            'LVGB':  {'splitConfidence': 0.2, 'tieThresh': 0.25,'gracePeriod': 100}},
        'USPS': {
            'ILVQ':  {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 275, 'metricLearnRate': 0.01, 'insertionTimingThresh': 10},
            'ISVM':  {'C': 2**2, 'sigma': 110},
            'ORF':   {'numTrees': 30, 'numRandomTests': 200, 'counterThreshold': 19},
            'IELM':  {'numHiddenNeurons': 450},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.004},
            'LVGB':  {'splitConfidence': 0.2, 'tieThresh': 0.2, 'gracePeriod': 85}},
        'DNA': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'linear', 'learnRateInitial': 0.1, 'metricLearnRate': 0.004, 'insertionTimingThresh': 500},
            'ISVM':  {'C': 2**3, 'sigma': 130},
            'ORF':   {'numTrees': 20, 'numRandomTests': 500, 'counterThreshold': 20},
            'IELM':  {'numHiddenNeurons': 300},
            'LPP':   {'classifierPerChunk': 4},
            'SGD':   {'eta0': 0.005},
            'LVGB':  {}},
        'isolet': {
            'ILVQ':  {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 600,  'insertionTimingThresh': 1000},
            'ISVM':  {'C': 2**8, 'sigma': 300},
            'ORF':   {'numTrees': 100, 'numRandomTests': 300, 'counterThreshold': 18},
            'IELM':  {'numHiddenNeurons': 450},
            'LPP':   {'classifierPerChunk': 7},
            'SGD':   {'eta0': 0.0002},
            'LVGB':  {'splitConfidence': 0.2, 'tieThresh': 0.2, 'gracePeriod': 160}},
        'letter': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 3.7, 'metricLearnRate': 0.03, 'insertionTimingThresh': 1},
            'ISVM':  {'C': 2**5, 'sigma': 8}, #XXVL to tune
            'ORF':   {'numTrees': 35, 'numRandomTests': 300, 'counterThreshold': 11},
            'IELM':  {'numHiddenNeurons': 480},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.03},
            'LVGB':  {'splitConfidence': 0.1, 'tieThresh': 0.1}},
        'gisette': {
            'ILVQ':  {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 1900, 'insertionTimingThresh': 400},
            'ISVM':  {'C': 2**2, 'sigma': 5000},
            'ORF':   {'numTrees': 20, 'numRandomTests': 300, 'counterThreshold': 50},
            'IELM':  {'numHiddenNeurons': 500},
            'LPP':   {'classifierPerChunk': 4},
            'SGD':   {'eta0': 0.005},
            'LVGB':  {}},
        'mnist': {
            'ILVQ':  {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 200, 'insertionTimingThresh': 10},
            'ISVM':  {'C': 2**7, 'sigma': 600},
            'ORF':   {'numTrees': 50, 'numRandomTests': 300, 'counterThreshold': 100},
            'IELM':  {'numHiddenNeurons': 420},
            'LPP':   {'classifierPerChunk': 3},
            'SGD':   {'eta0': 0.0008},
            'LVGB':  {'splitConfidence': 0.3, 'tieThresh': 0.2}},
        'satImage': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 5.5, 'metricLearnRate': 0.01, 'insertionTimingThresh': 2},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 20, 'numRandomTests': 100, 'counterThreshold': 5},
            'IELM':  {'numHiddenNeurons': 440},
            'LPP':   {'classifierPerChunk': 7},
            'SGD':   {'eta0': 0.02},
            'LVGB':  {'splitConfidence': 0.1}},
        'penDigits': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 4.4, 'metricLearnRate': 0.03, 'insertionTimingThresh': 3},
            'ISVM':  {'C': 2**10, 'sigma': 30},
            'ORF':   {'numTrees': 20, 'numRandomTests': 100, 'counterThreshold': 8},
            'IELM':  {'numHiddenNeurons': 490},
            'LPP':   {'classifierPerChunk': 6},
            'SGD':   {'eta0': 0.065},
            'LVGB':  {'splitConfidence': 0.1}},
        'HAR': {
            'ILVQ':  {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 200, 'insertionTimingThresh': 1000},
            'ISVM':  {'C': 2**7, 'sigma': 260},
            'ORF':   {'numTrees': 20, 'numRandomTests': 100, 'counterThreshold': 22},
            'IELM':  {'numHiddenNeurons': 490},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.004},
            'LVGB':  {'splitConfidence': 0.1}},
        'news20': {
            'ILVQ':  {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 4500, 'metricLearnRate': 0.01, 'insertionTimingThresh': 500},
            'ISVM':  {'C': 2**8, 'sigma': 0.75},
            'ORF':   {'numTrees': 10, 'numRandomTests': 500, 'counterThreshold': 13},
            'IELM':  {'numHiddenNeurons': 90},
            'LPP':   {'classifierPerChunk': 6},
            'SGD':   {'eta0': 0.00001},
            'LVGB':  {'splitConfidence': 0.3, 'tieThresh': 0.1, 'gracePeriod': 200, 'numClassifier': 5}}
    }


def getHyperParamsNonStationaryScaled():
    return {
        'elec': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 12, 'metricLearnRate': 0.001, 'insertionTimingThresh': 5},
            'ISVM':  {'C': 2**7, 'sigma': 5000000},
            'ORF':   {'numTrees': 45, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 88},
            'LPP':   {'classifierPerChunk': 5},
            'SGD':   {'eta0':  1.05},
            'LVGB':  {'splitConfidence': 0.08}},
        'weather': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 2.63, 'insertionTimingThresh': 23},
            'ISVM':  {'C': 2**0, 'sigma': 500},
            'ORF':   {'numTrees': 30, 'numRandomTests': 100, 'counterThreshold': 10},
            'IELM':  {'numHiddenNeurons': 120},
            'LPP':   {'classifierPerChunk': 7},
            'SGD':   {'eta0': 0.2},
            'LVGB':  {'splitConfidence': 0.02}},
        'sea': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 0.2, 'metricLearnRate': 0.01, 'insertionTimingThresh': 100000},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 20, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 57},
            'LPP':   {'classifierPerChunk': 7},
            'SGD':   {'eta0': 0.004},
            'LVGB':  {'splitConfidence': 0.001, 'numClassifier': 2}},
        'souza2CDT': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 5.2, 'metricLearnRate': 0.01, 'insertionTimingThresh': 55},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 40, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 50},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.008},
            'LVGB':  {'splitConfidence': 0.1, 'tieThresh': 0.1, 'gracePeriod': 200, 'numClassifier': 10}},
        'souza4CREV1': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 2.0, 'metricLearnRate': 0.01, 'insertionTimingThresh': 0},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 40, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 50},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.28},
            'LVGB':  {}},
        'souzaFG2C2D': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 1.9, 'metricLearnRate': 0.01, 'insertionTimingThresh': 15},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 20, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 50},
            'LPP':   {'classifierPerChunk': 5},
            'SGD':   {'eta0': 0.01},
            'LVGB':  {}},
        'souzaGears2C2D': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 1.7, 'metricLearnRate': 0.01, 'insertionTimingThresh': 40},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 40, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 100},
            'LPP':   {'classifierPerChunk': 3},
            'SGD':   {'eta0': 0.03},
            'LVGB':  {'splitConfidence': 0.01}},
        'souza2CHT': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 2.2, 'metricLearnRate': 0.01, 'insertionTimingThresh': 200},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 40, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 50},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.007},
            'LVGB':  {'splitConfidence': 0.00001}},
        'rbfFast': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 51, 'metricLearnRate': 0.01, 'insertionTimingThresh': 5},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 40, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 50},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.28},
            'LVGB':  {}},
        'hyperplaneSlow': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 51, 'metricLearnRate': 0.01, 'insertionTimingThresh': 1000000},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 40, 'numRandomTests': 100, 'counterThreshold': 10},
            'IELM':  {'numHiddenNeurons': 50},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.006},
            'LVGB':  {}},
        'rbfSlow': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 43, 'metricLearnRate': 0.005, 'insertionTimingThresh': 5},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 40, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 50},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.015},
            'LVGB':  {'splitConfidence': 0.1}},
        'cbConst': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 3, 'metricLearnRate': 0.05, 'insertionTimingThresh': 40},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 40, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 50},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.06},
            'LVGB':  {'splitConfidence': 0.08}},
        'cbSinus': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 4.78, 'metricLearnRate': 0.02, 'insertionTimingThresh': 40},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 40, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 50},
            'LPP':   {'classifierPerChunk': 8},
            'SGD':   {'eta0': 0.04},
            'LVGB':  {'splitConfidence': 0.1}},
        'keystroke': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 11.7, 'metricLearnRate': 0.01, 'insertionTimingThresh': 10},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 30, 'numRandomTests': 100, 'counterThreshold': 10},
            'IELM':  {'numHiddenNeurons': 180},
            'LPP':   {'classifierPerChunk': 7},
            'SGD':   {'eta0': 0.03},
            'LVGB':  {'splitConfidence': 0.003}},
        'spam': {
            'ILVQ':  {'netType': 'GLVQ', 'activFct': 'logistic', 'learnRateInitial': 60000, 'metricLearnRate': 0.01, 'insertionTimingThresh': 200},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 20, 'numRandomTests': 100, 'counterThreshold': 4},
            'IELM':  {'numHiddenNeurons': 100},
            'LPP':   {'classifierPerChunk': 1},
            'SGD':   {'eta0': 0.0004},
            'LVGB':  {}},
        'covType': {
            'ILVQ':  {'netType': 'GMLVQ', 'activFct': 'logistic', 'learnRateInitial': 15, 'metricLearnRate': 0.03, 'insertionTimingThresh': 100000000},
            'ISVM':  {'C': 2**0, 'sigma': 1.7},
            'ORF':   {'numTrees': 20, 'numRandomTests': 100, 'counterThreshold': 3},
            'IELM':  {'numHiddenNeurons': 100},
            'LPP':   {'classifierPerChunk': 1},
            'SGD':   {'eta0': 0.12},
            'LVGB':  {}}
        }


