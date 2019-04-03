from . import LVQFactory
from .classifierCommon import *

def getBaseClassifier(classifierName, classifierParams, listener=[]):
     if classifierName in [cILVQ, cILVQVoronoi, cILVQCluster, cILVQClosest]:
         return LVQFactory.getLVQClassifierByCfg(classifierParams, listener)
     elif classifierName == 'SGD':
         return SGD(eta0=classifierParams['eta0'], learningRate=classifierParams['learningRate'])
     elif classifierName == 'GNB':
         return GaussianNaiveBayes()
     elif classifierName == cMNB:
         return MultinomialNaiveBayes()
     elif classifierName == 'IELM':
         return IELM(numHiddenNeurons=classifierParams['numHiddenNeurons'])
     elif classifierName == 'LPP':
         return LPP(classifierPerChunk=classifierParams['classifierPerChunk'])
     elif classifierName == 'LPPNSE':
         return LPPNSE()
     elif classifierName == 'ISVM':
         return ISVM(kernel=classifierParams['kernel'], C=classifierParams['C'], sigma=classifierParams['sigma'], maxReserveVectors=classifierParams['maxSize'])
     elif classifierName == 'LASVM':
         return LASVM(C=classifierParams['C'], kGamma=classifierParams['kGamma'])
     elif classifierName in ['ORF', 'ORF2', 'ORF3', 'ORF4']:
         return ORF(numTrees=classifierParams['numTrees'], numRandomTests=classifierParams['numRandomTests'], maxDepth=classifierParams['maxDepth'], counterThreshold=classifierParams['counterThreshold'])
     elif classifierName in [cKNN, cKNN2, cKNN3, cKNN4]:
         #return KNNWindow(n_neighbors=classifierParams['nNeighbours'], maxSize=classifierParams['maxSize'], knnWeights=classifierParams['weights'], driftStrategy=classifierParams['driftStrategy'], listener=listener)
         return SAMKNN(n_neighbors=classifierParams['nNeighbours'], maxSize=classifierParams['maxSize'], knnWeights=classifierParams['knnWeights'], recalculateSTMError=None, useLTM=False, listener=listener)
     elif classifierName == 'SAMKNN':
         return SAMKNN(n_neighbors=classifierParams['nNeighbours'], maxSize=classifierParams['maxSize'], knnWeights=classifierParams['knnWeights'], recalculateSTMError=classifierParams['recalculateSTMError'], useLTM=classifierParams['useLTM'], listener=listener)
     elif classifierName in [cHT, cHTCGD, cHTOSM]:
         return HoeffdingTree(gracePeriod=classifierParams['gracePeriod'],
                              splitConfidence=classifierParams['splitConfidence'],
                              tieThreshold=classifierParams['tieThreshold'],
                              removePoorAtts=classifierParams['removePoorAtts'],
                              mode=classifierParams['mode'],
                              gracePeriodFunction=classifierParams['gracePeriodFunction'],
                              nBins=classifierParams['nBins'],
                              listener=listener)
     elif classifierName == cSAMNB:
         #return MMMNaiveBayes(windowSize=classifierParams['windowSize'], driftStrategy=classifierParams['driftStrategy'], listener=listener)
         return SAMNB(n_neighbors=classifierParams['nNeighbours'], maxSize=classifierParams['maxSize'], recalculateSTMError=classifierParams['recalculateSTMError'], useLTM=classifierParams['useLTM'], listener=listener)
     elif classifierName == 'MMMGNB':
         return MMMGNaiveBayes(windowSize=classifierParams['windowSize'], driftStrategy=classifierParams['driftStrategy'], listener=listener)
     elif classifierName == 'WAVG':
         return WeightedAvg(length=1)