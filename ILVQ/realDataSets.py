import json

import numpy as np

from . import Paths



def getOutdoor():
    trainSamples = np.loadtxt(Paths.outdoorTrainSamplesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.outdoorTrainLabelsPath(), skiprows=1, dtype=np.uint8)
    testSamples = np.loadtxt(Paths.outdoorTestSamplesPath(), skiprows=1)
    testLabels = np.loadtxt(Paths.outdoorTestLabelsPath(), skiprows=1, dtype=np.uint8)
    return trainSamples, trainLabels, testSamples, testLabels

def getOutdoorStream():
    samples = np.loadtxt(Paths.outdoorStreamSamplesPath(), skiprows=0)
    labels = np.loadtxt(Paths.outdoorStreamLabelsPath(), skiprows=0, dtype=np.uint8)
    return samples, labels

def getBorder():
    trainSamples = np.loadtxt(Paths.borderTrainSamplesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.borderTrainLabelsPath(), skiprows=1, dtype=np.uint8)
    testSamples = np.loadtxt(Paths.borderTestSamplesPath(), skiprows=1)
    testLabels = np.loadtxt(Paths.borderTestLabelsPath(), skiprows=1, dtype=np.uint8)
    return trainSamples, trainLabels, testSamples, testLabels


def getOverlap():
    trainSamples = np.loadtxt(Paths.overlapTrainSamplesPath(), skiprows=1)
    trainLabels = np.loadtxt(Paths.overlapTrainLabelsPath(), skiprows=1, dtype=np.uint8)
    testSamples = np.loadtxt(Paths.overlapTestSamplesPath(), skiprows=1)
    testLabels = np.loadtxt(Paths.overlapTestLabelsPath(), skiprows=1, dtype=np.uint8)
    return trainSamples, trainLabels, testSamples, testLabels

def getCOIL():
    trainSamples = np.loadtxt(Paths.coilTrainSamplesPath(), skiprows=0)
    trainLabels = np.loadtxt(Paths.coilTrainLabelsPath(), skiprows=0, dtype=np.uint8)
    testSamples = np.loadtxt(Paths.coilTestSamplesPath(), skiprows=0)
    testLabels = np.loadtxt(Paths.coilTestLabelsPath(), skiprows=0, dtype=np.uint8)
    return trainSamples, trainLabels, testSamples, testLabels