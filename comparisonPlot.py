from ILVQ.DataSetFactory import isStationary
import matplotlib.pyplot as plt
import matplotlib
from ILVQ import Serialization
import csv
import os
import seaborn
from ILVQ.classifierCommon import *
from ILVQ.DataSet import *
from sklearn.metrics import accuracy_score, cohen_kappa_score
import itertools

seaborn.set()
USE_ERROR_RATE = True
FONT_SIZE_TITLE = 24
FONT_SIZE_TICKS = 20
FONT_SIZE_AXIS_LABEL = 22
FONT_SIZE_LEGEND = 18



classifierNameMapping = {cHTJ: 'VFDT', cHTJ: 'VFDT', cHTOSM: 'OSM', 'HoeffTreeJ3': 'CGD',
                         #cSAMBG: r'$\text{SAM}^{\text{B}}_{\text{k,f,d}}$',
                         cARF: 'ARF',
                         cLVGB: 'LVGB',
                         cSAMBG: 'SAM-E',
                         #cSAMBG: r'$\text{SAM-E}_{\text{k,f,d}}$',
                         cSAMKNNJ: 'SAM',
                         cSAMBGFSK: r'$\text{SAM-E}_{\text{k,f}}$',
                         cSAMBGK: r'$\text{SAM-E}_{\text{k}}$',
                         cSAMBGNone: r'$\text{SAM-E}_{\text{None}}$',
                         cPAW: 'PAW',
                         cKNN: r'$\text{KNN}_{S}$',
                         #cKNN: 'ws 500', cKNN2: 'ws 1000', cKNN3: 'ws 5000', cKNN4: 'ws 20000',

                         cILVQ: 'COSMOS',
cILVQClosest: 'Closest',
cILVQCluster: 'Cluster',
cILVQVoronoi: 'Voronoi'
                         }



datasetMapping = {dsCovType: 'Cover type', dsHiggs: 'HIGGS', dsRbfLarge: 'RBF', dsRtgLarge: 'RTG', dsMnist8m: 'MNIST8M', dsPoker: 'Poker', dsPamap: 'PAMAP2', dsElec: 'Electricity',
                  dsMovingRBF: 'Moving RBF',dsRialto: 'Rialto', dsAirline: 'Airline', dsLedDrift: 'LED-Drift', dsLedDriftSmall : 'LED-Drift', dsOutdoor : 'Outdoor', dsWeather: 'Weather',
                  dsKdd: 'KDD99', dsBorder: 'Border', dsOverlap: 'Overlap', dsInterRBF: 'Interchanging RBF', dsTransientChessb: 'Transient Chessboard', dsMovingSquares: 'Moving Squares'}

#datasetStepSizeMapping = {dsCovType: 20000, dsPoker: 20000, dsRialto: 3000, dsWeather: 1000}
datasetStepSizeMapping = {}

def getDatasetStepsize(key):
    if key in datasetStepSizeMapping:
        return datasetStepSizeMapping[key]
    else:
        return None

def getDatasetMapping(key):
    if key in datasetMapping:
        return datasetMapping[key]
    else:
        return key

def getClassifierMapping(key):
    if key in classifierNameMapping:
        return classifierNameMapping[key]
    else:
        return key

def getColorToClassifier(classifierName):
    #return seaborn.color_palette(n_colors=12)[[cDACC, cSAMKNNJ, cPAW, cLVGB, cARF, cSAMBG, cSAMBGFS, cSAMBGK, cSAMBGNone, cSAMBGFSK, cSAMBGST, cKNN].index(classifierName)]

    #return ['red', 'green', 'blue', 'orange', 'orange', 'magenta'][[cHTJ, 'HoeffTreeJ2', 'HoeffTreeJ3'].index(classifierName)]

    #return seaborn.color_palette(n_colors=12)[[cHTJ, 'HoeffTreeJ2', 'HoeffTreeJ3'].index(classifierName)]
    return seaborn.color_palette(n_colors=12)[[cILVQ, cILVQClosest, cILVQCluster, cILVQVoronoi, cSAMKNN].index(classifierName)]
    #return seaborn.color_palette(n_colors=12)[[cILVQ, cISVM, cIELM, cLPP, cORF].index(classifierName)]

    #return seaborn.color_palette(n_colors=12)[[cKNN, cKNN2, cKNN3, cKNN4].index(classifierName)]





def getLineStyleToClassifier(classifierName):
    '''if unicodedata.normalize('NFKD', classifierName).encode('ascii','ignore') in ['iSVM', 'iLVQ']:
        return '-'
    else:
        return '--'
        '''
    return '-'

def getMarkerToClassifier(classifierName):
    '''if unicodedata.normalize('NFKD', classifierName).encode('ascii','ignore') in ['ILVQ']:
        return '^'
    elif unicodedata.normalize('NFKD', classifierName).encode('ascii','ignore') in ['ORF', 'IELM']:
        return 'o'
    elif unicodedata.normalize('NFKD', classifierName).encode('ascii','ignore') in ['LPP']:
        return 's'
    else:
        return '''''


def plotCSVData(filePath, classifierNames, xAxisName, yAxisName, title=None, invertYValues=False, logScale=False, showLegend=True):
    matplotlib.rc('xtick', labelsize=FONT_SIZE_TICKS)
    matplotlib.rc('ytick', labelsize=FONT_SIZE_TICKS)
    plt.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    data = np.loadtxt(filePath, delimiter=',')
    xValues = data[0, :]
    fig, subplot = plt.subplots(1, 1)
    for lineIdx in range(data.shape[0]-1):
        yValues = data[lineIdx+1, :]
        if invertYValues:
            yValues = 1-yValues

        classifierName = classifierNames[lineIdx]
        subplot.plot(xValues, yValues, label=getClassifierMapping(classifierName), color=getColorToClassifier(classifierName),
                         marker=getMarkerToClassifier(classifierName))

    if title is not None:
        subplot.set_title(title, fontsize=FONT_SIZE_TITLE)
    subplot.set_xlabel(xAxisName, fontsize=FONT_SIZE_AXIS_LABEL)
    subplot.set_ylabel(yAxisName, fontsize=FONT_SIZE_AXIS_LABEL)
    if logScale:
        subplot.set_yscale('log')
    if showLegend:
        subplot.legend(fancybox=True, framealpha=0.5, fontsize=FONT_SIZE_LEGEND)
    '''dirName = os.path.dirname(filePath)
    fileName = os.path.basename(filePath)'''
    savePath = os.path.splitext(filePath)[0] + '_plot.pdf'
    fig.savefig(savePath, bbox_inches='tight')

def plotSingleAccuracies(dataSetName, classifierEvaluations, dataSetIDStr, dstDirectory, classifierNames=None, stepSize=None, permutate=False):
    matplotlib.rc('xtick', labelsize=FONT_SIZE_TICKS)
    matplotlib.rc('ytick', labelsize=FONT_SIZE_TICKS)
    plt.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    fig, subplot = plt.subplots(1, 1)
    accuracyName = 'accuracies'
    iterationIdx = 0
    for classifierName in classifierEvaluations['values']:
        if classifierEvaluations['meta']['streamSetting']:
            data = classifierEvaluations['values'][classifierName][iterationIdx]
            if (not classifierNames or classifierName in classifierNames) and len(data['Y']) > 0:
                #accuracies = []
                #for iteration in range(len(classifierEvaluations['values'][classifierName][accuracyName])):
                #    accuracies.append(classifierEvaluations['values'][classifierName][accuracyName][iteration])
                #accuracies = np.mean(accuracies, axis=0)
                if stepSize is None:
                    stepSize = getDatasetStepsize(dataSetName)
                if stepSize is None:
                    stepSize = int(len(data['Y'])/20.)

                xValues = []
                yValues = []
                for x in np.arange(stepSize, len(data['Y']), stepSize):
                #for x in np.arange(stepSize, 30000, stepSize):
                    xValues.append(x)

                    accuracy = accuracy_score(data['Y'][x-stepSize:x], data['YPred'][x-stepSize:x])
                    yValues.append(accuracy)
        else:
            yValues = []
            for classifiedLabels in classifierEvaluations['values'][classifierName]['YPred']:
                classifiedLabels = np.array(classifiedLabels).astype(np.int)
                yValues.append(accuracy_score(classifierEvaluations['values'][classifierName]['Y'], classifiedLabels))
            xValues = classifierEvaluations['values'][classifierName]['times']

        if USE_ERROR_RATE:
            plt.ylabel('Error rate', fontsize=FONT_SIZE_AXIS_LABEL)
            yValues = 1 - np.array(yValues)
        else:
            plt.ylabel('accuracy', fontsize=FONT_SIZE_AXIS_LABEL)
        subplot.plot(xValues, yValues, label=getClassifierMapping(classifierName), color=getColorToClassifier(classifierName),
                         marker=getMarkerToClassifier(classifierName))

    #subplot.set_xlim([xValues[0], xValues[-1]])
    subplot.set_title('%s %s' % (getDatasetMapping(dataSetName), 'shuffled' if permutate else ''), fontsize=FONT_SIZE_TITLE)
    subplot.set_xlabel('\#Samples', fontsize=FONT_SIZE_AXIS_LABEL)
    subplot.legend(fancybox=True, framealpha=0.5, fontsize=FONT_SIZE_LEGEND)
    #subplot.set_ylim([0,0.7])
    fig.savefig(dstDirectory + dataSetIDStr + '_' + accuracyName + '.pdf', bbox_inches='tight')


def plotSingleComplexities(dataSetName, classifierEvaluations, dataSetIDStr, dstDirectory, classifierNames=None):
    fig, subplot = plt.subplots(1, 1)

    for classifierName in classifierEvaluations['values']:
        if not classifierNames or classifierName in classifierNames:
            #for iteration in range(len(classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'])):
            #    complexities.append(classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'][iteration])
            #complexities = np.mean(complexities, axis=0)
            xValues = classifierEvaluations['values'][classifierName]['idx']
            complexities = classifierEvaluations['values'][classifierName]['complexitiesNumParamMetric']
            subplot.plot(xValues, complexities, label=getClassifierMapping(classifierName), color=getColorToClassifier(classifierName))
            subplot.set_title('%s-complexity (num)' % dataSetName, fontsize=20)
            subplot.set_xlabel('#Samples', fontsize=16)
            subplot.set_ylabel('#Parameter', fontsize=16)
            subplot.legend(fontsize=14)

    fig.savefig(dstDirectory + dataSetIDStr + '_complexitiesNumParamMetric' + '.pdf', bbox_inches='tight')

def plotChunkSizeAccuracies(dataSetName, classifierEvaluations, streamSetting, dataSetIDStr, dstDirectory, accuracyName, criteria, sortedIndices, classifierNames=None):
    fig, axTestAccuraciesChunkSizes = plt.subplots(1, 1)
    for i in range(len(classifierEvaluations['values'].keys())):
        classifierName = classifierEvaluations['values'].keys()[i]
        if not classifierNames or classifierName in classifierNames:
            criterionAccuracies = []
            for criterion in classifierEvaluations['values'][classifierName].keys():
                if classifierEvaluations['values'][classifierName][criterion].has_key(accuracyName):
                    accuracies = []
                    for iteration in range(len(classifierEvaluations['values'][classifierName][criterion][accuracyName])):
                        if streamSetting:
                            accuracies.append(np.mean(classifierEvaluations['values'][classifierName][criterion][accuracyName][iteration]))
                        else:
                            accuracies.append(classifierEvaluations['values'][classifierName][criterion][accuracyName][iteration][-1])
                    criterionAccuracies.append(np.mean(accuracies))
            axTestAccuraciesChunkSizes.plot(criteria, np.array(criterionAccuracies)[sortedIndices], label=getClassifierMapping(classifierName), color=getColorToClassifier(classifierName), linestyle=getLineStyleToClassifier(classifierName), marker=getMarkerToClassifier(classifierName), lw=2, ms=7)
            matplotlib.rc('xtick', labelsize=16)
            matplotlib.rc('ytick', labelsize=16)
            axTestAccuraciesChunkSizes.set_xlabel('window-/chunk size', fontsize=16)
            axTestAccuraciesChunkSizes.set_ylabel('Accuracy', fontsize=16)
            axTestAccuraciesChunkSizes.set_xlim([criteria[0], criteria[-1]])
            axTestAccuraciesChunkSizes.legend(loc=0)
    fig.savefig(dstDirectory + dataSetIDStr + '_' + accuracyName + 'ChunkSizes.pdf', bbox_inches='tight')

def plotChunkSizeComplexities(dataSetName, classifierEvaluations, dataSetIDStr, dstDirectory, criteria, sortedIndices, classifierNames=None):
    fig, axComplexitiesChunkSizes = plt.subplots(1, 1)
    for i in range(len(classifierEvaluations['values'].keys())):
        classifierName = classifierEvaluations['values'].keys()[i]
        if not classifierNames or classifierName in classifierNames:
            criterionComplexities = []
            for criterion in classifierEvaluations['values'][classifierName].keys():
                if classifierEvaluations['values'][classifierName][criterion].has_key('complexitiesNumParamMetric'):
                    complexities = []
                    for iteration in range(len(classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'])):
                        complexities.append(classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'][iteration][-1])
                    criterionComplexities.append(np.mean(complexities))
            axComplexitiesChunkSizes.semilogy(criteria, np.array(criterionComplexities)[sortedIndices], label=getClassifierMapping(classifierName), color=getColorToClassifier(classifierName), linestyle=getLineStyleToClassifier(classifierName), marker=getMarkerToClassifier(classifierName), lw=2, ms=7)
            #axComplexitiesChunkSizes.set_title(dataSetName + ' - Chunk sizes vs. Model complexity')
            matplotlib.rc('xtick', labelsize=16)
            matplotlib.rc('ytick', labelsize=16)
            axComplexitiesChunkSizes.set_xlabel('window-/chunk size', fontsize=16)
            axComplexitiesChunkSizes.set_ylabel('#Parameter', fontsize=16)
            axComplexitiesChunkSizes.set_xlim([ criteria[0], criteria[-1]])
            axComplexitiesChunkSizes.set_ylim([50, 25000])
            #axComplexitiesChunkSizes.legend()
    fig.savefig(dstDirectory + dataSetIDStr + '_complexityNumParamMetricChunkSizes.pdf', bbox_inches='tight')

def plotComplexityAccuracies(dataSetName, classifierEvaluations, streamSetting, evalFilePrefix, dstDirectory, accuracyName, classifierNames=None):
    fig, axTestAccuraciesChunkSizes = plt.subplots(1, 1)
    for i in range(len(classifierEvaluations['values'].keys())):
        classifierName = classifierEvaluations['values'].keys()[i]
        if not classifierNames or classifierName in classifierNames:
            criterionAccuracies = []
            criterionComplexities = []
            for criterion in classifierEvaluations['values'][classifierName].keys():
                if classifierEvaluations['values'][classifierName][criterion].has_key(accuracyName):
                    accuracies = []
                    for iteration in range(len(classifierEvaluations['values'][classifierName][criterion][accuracyName])):
                        if streamSetting:
                            accuracies.append(np.mean(classifierEvaluations['values'][classifierName][criterion][accuracyName][iteration]))
                        else:
                            accuracies.append(classifierEvaluations['values'][classifierName][criterion][accuracyName][iteration][-1])
                    criterionAccuracies.append(np.mean(accuracies))

                if classifierEvaluations['values'][classifierName][criterion].has_key('complexitiesNumParamMetric'):
                    complexities = []
                    for iteration in range(len(classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'])):
                        complexities.append(classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'][iteration][-1])
                    criterionComplexities.append(np.mean(complexities))
            axTestAccuraciesChunkSizes.plot(criterionComplexities, criterionAccuracies, label=getClassifierMapping(classifierName), c=seaborn.color_palette()[i])
            axTestAccuraciesChunkSizes.set_title(dataSetName + ' - ' + 'Model complexity vs. Accuracy')
            axTestAccuraciesChunkSizes.set_xlabel('#Parameter')
            axTestAccuraciesChunkSizes.set_ylabel('Accuracy')
            axTestAccuraciesChunkSizes.legend(fontsize=14)
    fig.savefig(dstDirectory + evalFilePrefix + '_' + accuracyName + 'Complexity.pdf', bbox_inches='tight')

def plotRunTimes(dataSetName, classifierEvaluations, dataSetIDStr, dstDirectory, classifierNames, permutate):
    plt.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    fig, subplot = plt.subplots(1, 1)
    for classifierName in classifierEvaluations['values']:
        if classifierNames is None or classifierName in classifierNames:
            if len(classifierEvaluations['values'][classifierName]['nInstances']) > 0:
                xValues = classifierEvaluations['values'][classifierName]['nInstances']
                yValues = classifierEvaluations['values'][classifierName]['runTimes']
                xValues.insert(0,0)
                yValues.insert(0, 0)
                subplot.plot(xValues, yValues, label=getClassifierMapping(classifierName),
                             color=getColorToClassifier(classifierName),
                             marker=getMarkerToClassifier(classifierName))
    #subplot.legend(fontsize=14)
    subplot.set_title('%s %s' % (getDatasetMapping(dataSetName), 'shuffled' if permutate else ''), fontsize=FONT_SIZE_TITLE)
    subplot.set_ylabel("Run time (s)", fontsize=FONT_SIZE_AXIS_LABEL)
    subplot.set_xlabel("\#Samples", fontsize=FONT_SIZE_AXIS_LABEL)
    plt.xlim([0, np.max(xValues)])
    #plt.ylim([0, np.max(yValues)])
    fig.savefig(dstDirectory + dataSetIDStr + '_' + 'runTimes' + '.pdf', bbox_inches='tight')


def autolabel(rects, subplot):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        subplot.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.2f' % height,
                ha='center', va='bottom', rotation=90, fontsize=18)

def plotRuntTimesMultiThreading(dstDirectory):
    plt.rc('text', usetex=True)
    matplotlib.rc('xtick', labelsize=FONT_SIZE_TICKS)
    matplotlib.rc('ytick', labelsize=FONT_SIZE_TICKS)
    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    width=10
    spacing=0.5

    nSizes = np.array([20, 70, 120, 170])
    runTimesST = [4053.64, 8168.39, 20765.95, 42650.52]
    ramHoursST = [0.22,0.87,5.42,21.70]

    runTimesMT = [1913.42,3709.86,7512.78,13668.69]
    ramHoursMT = [0.20,0.76,3.81,13.85]

    runTimesMTBuffered = [1700.89,2081.05,3406.38, 6621.31]
    ramHoursMTBuffered = [0.18,0.43,1.74,6.63]

    fig, subplot = plt.subplots(1, 1)
    subplot.set_title('CPU-Time', fontsize=FONT_SIZE_TITLE)

    rects = subplot.bar(nSizes-width-spacing, runTimesST, label="sequential", color=seaborn.color_palette()[0], width=width)
    autolabel(rects, subplot)
    rects = subplot.bar(nSizes, runTimesMT, label="parallel", color=getColorToClassifier(cSAMBG), width=width)
    autolabel(rects, subplot)
    rects = subplot.bar(nSizes+width+spacing, runTimesMTBuffered, label="parallel-buffered", color=seaborn.color_palette()[2], width=width)
    autolabel(rects, subplot)
    subplot.set_ylabel("Avg. CPU time (s)", fontsize=FONT_SIZE_AXIS_LABEL)
    subplot.set_xlabel("Ensemble size", fontsize=FONT_SIZE_AXIS_LABEL)
    subplot.legend(loc=2, fontsize=FONT_SIZE_LEGEND)
    plt.xticks(nSizes, [10,20,50,100])
    plt.xlim([0,200])
    plt.ylim([0, np.max(runTimesST)*1.5])
    fig.savefig(dstDirectory + 'runtimesMultiThreading.pdf', bbox_inches='tight')


    fig, subplot = plt.subplots(1, 1)
    subplot.set_title('RAM-Hours', fontsize=FONT_SIZE_TITLE)
    rects = subplot.bar(nSizes-width-spacing, ramHoursST, label="Sequential", color=seaborn.color_palette()[0], width=width)
    autolabel(rects, subplot)
    rects = subplot.bar(nSizes, ramHoursMT, label="Parallel", color=getColorToClassifier(cSAMBG), width=width)
    autolabel(rects, subplot)
    rects = subplot.bar(nSizes+width+spacing, ramHoursMTBuffered, label="Parallel-buffered", color=seaborn.color_palette()[2], width=width)
    autolabel(rects, subplot)
    subplot.set_ylabel("Avg. RAM-Hours (GB-Hours)", fontsize=FONT_SIZE_AXIS_LABEL)
    subplot.set_xlabel("Ensemble size", fontsize=FONT_SIZE_AXIS_LABEL)
    #subplot.legend(loc=2)
    #subplot.spines["top"].set_visible(False)
    #subplot.spines["right"].set_visible(False)
    #subplot.spines["bottom"].set_visible(True)
    #subplot.spines["left"].set_visible(False)

    plt.xticks(nSizes, [10,20,50,100])
    plt.xlim([0,200])
    plt.ylim([0, np.max(ramHoursST)*1.3])
    fig.savefig(dstDirectory + 'ramHoursMultiThreading.pdf', bbox_inches='tight')

def plotKappaError(dataSetName, classifierEvaluations, dataSetIDStr, dstDirectory, classifierNames, permutate):
    plt.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    matplotlib.rc('xtick', labelsize=FONT_SIZE_TICKS)
    matplotlib.rc('ytick', labelsize=FONT_SIZE_TICKS)
    fig, subplot = plt.subplots(1, 1)
    iterationIdx = 0
    for classifierName in classifierEvaluations['values']:
        if classifierNames is None or classifierName in classifierNames:
            data = classifierEvaluations['values'][classifierName][iterationIdx]
            if 'YPredEnsemble' in data and len(data['YPredEnsemble']) > 0 and len(data['Y'])> 0:
                ensembleLabels = np.atleast_2d(data['YPredEnsemble'])
                Y = data['Y']
                errors = []
                for labels in ensembleLabels:
                    errors.append(1 - accuracy_score(Y, labels))
                xValues = []
                yValues = []
                pairWiseIndices = itertools.combinations(range(ensembleLabels.shape[0]), 2)
                for indices in pairWiseIndices:
                    avgError = (errors[indices[0]] + errors[indices[1]]) / 2.
                    xValues.append(cohen_kappa_score(ensembleLabels[indices[0], :], ensembleLabels[indices[1], :]))
                    yValues.append(avgError)
                subplot.scatter(xValues, yValues, label=getClassifierMapping(classifierName), color=getColorToClassifier(classifierName),
                                 marker=getMarkerToClassifier(classifierName))
    #subplot.legend(fontsize=FONT_SIZE_LEGEND)
    subplot.set_title('%s %s' % (getDatasetMapping(dataSetName), 'shuffled' if permutate else ''), fontsize=FONT_SIZE_TITLE)
    subplot.set_ylabel("Avg. error", fontsize=FONT_SIZE_AXIS_LABEL)
    subplot.set_xlabel("Pairwise kappa statistic", fontsize=FONT_SIZE_AXIS_LABEL)
    fig.savefig(dstDirectory + dataSetIDStr + '_' + 'kappaError' + '.pdf', bbox_inches='tight')


def comparisonPlots(dataSetName, classifierEvaluations, dataSetIDStr, dstDirectory, classifierNames=None, stepSize=None, permutate=False):
    plotSingleAccuracies(dataSetName, classifierEvaluations, dataSetIDStr, dstDirectory, classifierNames=classifierNames, stepSize=stepSize, permutate=permutate)
    #plotKappaError(dataSetName, classifierEvaluations, dataSetIDStr, dstDirectory, classifierNames=classifierNames, permutate=permutate)
    #plotRunTimes(dataSetName, classifierEvaluations, dataSetIDStr, dstDirectory, classifierNames=classifierNames, permutate=permutate)
    #plotSingleComplexities(dataSetName, classifierEvaluations, dataSetIDStr, dstDirectory, classifierNames=classifierNames)

    '''
    print(classifierEvaluations['values'])
    criteria = np.array(list(classifierEvaluations['values'][list(classifierEvaluations['values'].keys())[0]].keys())).astype(int)
    sortedIndices = np.argsort(criteria)

    if len(criteria) > 1:
        if criterionName == 'chunkSize':
            plotChunkSizeAccuracies(dataSetName, classifierEvaluations, streamSetting, dataSetIDStr, dstDirectory, accuracyName, criteria, sortedIndices, classifierNames=classifierNames)
            plotChunkSizeComplexities(dataSetName, classifierEvaluations, dataSetIDStr, dstDirectory, criteria, sortedIndices, classifierNames=classifierNames)
        elif criterionName == 'complexity':
            plotComplexityAccuracies(dataSetName, classifierEvaluations, streamSetting, dataSetIDStr, dstDirectory, accuracyName, classifierNames=classifierNames)'''


def doComparisonPlot(dataSetName, classifierNames, dataSetIDStr=None, stepSize=None, filePath=None, permutate=False):
    if dataSetIDStr is None:
        dataSetIDStr = DataSet.getIDStr(dataSetName, False)
    classifierEvaluations = Serialization.loadJsonFile(filePath)

    comparisonPlots(dataSetName, classifierEvaluations, dataSetIDStr, os.path.dirname(filePath), classifierNames=classifierNames, stepSize=stepSize, permutate=permutate)


def getBestChunkSize(classifierEvaluations, classifierName, streamSetting):
    maxValue = 0
    bestChunkSize = 0
    if streamSetting:
        accuracyName = 'trainPredictionAccuracies'
    else:
        accuracyName = 'testAccuracies'
    for key in classifierEvaluations['values'][classifierName]:
        criterion = key
        if accuracyName in classifierEvaluations['values'][classifierName][criterion]:
            accuracies = []
            for iteration in range(len(classifierEvaluations['values'][classifierName][criterion][accuracyName])):
                accuracies.append(classifierEvaluations['values'][classifierName][criterion][accuracyName][iteration])
            if streamSetting:
                accuracy = np.mean(accuracies)
            else:
                accuracy = np.mean(accuracies, axis=0)[-1]
            if accuracy > maxValue:
                maxValue = accuracy
                bestChunkSize = int(criterion)
    return bestChunkSize, maxValue

def getSingleRunValues(classifierEvaluations, classifierName, bestChunkSize):
    accuracies = []
    meanAccs = []
    complexitiesNumParametric = []
    for iteration in range(len(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['complexitiesNumParamMetric'])):
        complexitiesNumParametric.append(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['complexitiesNumParamMetric'][iteration])
    complexitiesNumParametric = np.array(complexitiesNumParametric)


    complexities = []
    for iteration in range(len(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['complexities'])):
        complexities.append(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['complexities'][iteration])
    complexities = np.array(complexities)
    meanEndComplexity = np.mean(complexities, axis=0)[-1]
    stdEndComplexity = np.std(complexities, axis=0)[-1]
    numTrainSamples = classifierEvaluations['meta']['numTrainSamples']
    if 'testAccuracies' in classifierEvaluations['values'][classifierName][str(bestChunkSize)]:
        for iteration in range(len(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['testAccuracies'])):
            accuracies.append(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['testAccuracies'][iteration])
            print(accuracies)
    elif classifierEvaluations['values'][classifierName][str(bestChunkSize)].has_key('trainPredictionAccuracies'):
        for iteration in range(len(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['trainPredictionAccuracies'])):
            accuracies.append(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['trainPredictionAccuracies'][iteration])
            meanAccs.append(classifierEvaluations['values'][classifierName][str(bestChunkSize)]['finalACC'][iteration])
    #print accuracies
    xValues = np.arange(bestChunkSize, numTrainSamples + 1, bestChunkSize)
    print(xValues, bestChunkSize, numTrainSamples)
    if xValues[-1] != numTrainSamples:
        xValues = np.append(xValues, numTrainSamples)
    #print xValues
    if len(xValues) > 1:
        if xValues[-1] != numTrainSamples:
            xValues = np.append(xValues, numTrainSamples)
        accuracies = np.array(accuracies)
        meanFirstTenthAcc = np.interp(0.1 * numTrainSamples, xValues, np.mean(accuracies, axis=0))
        stdFirstTenthAcc = np.interp(0.1 * numTrainSamples, xValues, np.std(accuracies, axis=0))
        meanFirstQuarterAcc = np.interp(0.25 * numTrainSamples, xValues, np.mean(accuracies, axis=0))
        stdFirstQuarterAcc = np.interp(0.25 * numTrainSamples, xValues, np.std(accuracies, axis=0))
        meanSecondQuarterAcc = np.interp(0.5 * numTrainSamples, xValues, np.mean(accuracies, axis=0))
        stdSecondQuarterAcc = np.interp(0.5 * numTrainSamples, xValues, np.std(accuracies, axis=0))
        meanThirdQuarterAcc = np.interp(0.75 * numTrainSamples, xValues, np.mean(accuracies, axis=0))
        stdThirdQuarterAcc = np.interp(0.75 * numTrainSamples, xValues, np.std(accuracies, axis=0))
        meanEndAccuracy = np.mean(accuracies, axis=0)[-1]
        stdEndAccuracy = np.std(accuracies, axis=0)[-1]
        meanEndComplexityNumParamMetric = np.mean(complexitiesNumParametric, axis=0)[-1]
        stdEndComplexityNumParamMetric = np.std(complexitiesNumParametric, axis=0)[-1]
    else:
        meanFirstTenthAcc = 0
        stdFirstTenthAcc = 0
        meanFirstQuarterAcc = 0
        stdFirstQuarterAcc = 0
        meanSecondQuarterAcc = 0
        stdSecondQuarterAcc = 0
        meanThirdQuarterAcc = 0
        stdThirdQuarterAcc = 0
        meanEndAccuracy = np.mean(accuracies, axis=0)

        stdEndAccuracy = np.std(accuracies, axis=0)
        meanEndComplexityNumParamMetric = np.mean(complexitiesNumParametric, axis=0)
        stdEndComplexityNumParamMetric = np.std(complexitiesNumParametric, axis=0)
    if classifierEvaluations['values'][classifierName][str(bestChunkSize)].has_key('trainPredictionAccuracies'):
        meanAccuracy = np.mean(meanAccs)
        stdAccuracy = np.std(meanAccs)
    else:
        meanAccuracy = np.mean(accuracies)
        stdAccuracy = np.std(accuracies)

    return [meanFirstTenthAcc, stdFirstTenthAcc, meanFirstQuarterAcc, stdFirstQuarterAcc, meanSecondQuarterAcc, stdSecondQuarterAcc, meanThirdQuarterAcc, stdThirdQuarterAcc,
            meanEndAccuracy, stdEndAccuracy, meanAccuracy, stdAccuracy, meanEndComplexity, stdEndComplexity, meanEndComplexityNumParamMetric, stdEndComplexityNumParamMetric]


def getChunkValues(classifierEvaluations, classifierName):
    endAccuracies = []
    allAccuracies = []
    complexities = []
    complexitiesNumParamMetric = []
    for criterion in classifierEvaluations['values'][classifierName].keys():
        for iteration in range(len(classifierEvaluations['values'][classifierName][criterion]['complexities'])):
            if classifierEvaluations['values'][classifierName][criterion].has_key('testAccuracies'):
                allAccuracies = allAccuracies + classifierEvaluations['values'][classifierName][criterion]['testAccuracies'][iteration]
                endAccuracies.append(classifierEvaluations['values'][classifierName][criterion]['testAccuracies'][iteration][-1])
            else:
                allAccuracies = allAccuracies + classifierEvaluations['values'][classifierName][criterion]['trainPredictionAccuracies'][iteration]
                endAccuracies.append(classifierEvaluations['values'][classifierName][criterion]['trainPredictionAccuracies'][iteration][-1])
            complexities.append(classifierEvaluations['values'][classifierName][criterion]['complexities'][iteration][-1])
            complexitiesNumParamMetric.append(classifierEvaluations['values'][classifierName][criterion]['complexitiesNumParamMetric'][iteration][-1])
    meanAllAcuracy = np.mean(allAccuracies)
    stdAllAcuracy = np.std(allAccuracies)
    meanChunkAccuracy = np.mean(endAccuracies)
    stdChunkAccuracy =  np.std(endAccuracies)
    meanChunkComplexity = np.mean(complexities)
    stdChunkComplexity = np.std(complexities)
    meanChunkComplexityNumParamMetric = np.mean(complexitiesNumParamMetric)
    stdChunkComplexityNumParamMetric = np.std(complexitiesNumParamMetric)
    return [meanChunkAccuracy, stdChunkAccuracy, meanChunkComplexity, stdChunkComplexity, meanChunkComplexityNumParamMetric, stdChunkComplexityNumParamMetric, meanAllAcuracy, stdAllAcuracy]




