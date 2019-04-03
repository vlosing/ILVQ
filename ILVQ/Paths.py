import os

stationaryPrefix = './datasets/'

def outdoorTrainSamplesPath():
    return os.path.join(stationaryPrefix, 'Outdoor/Outdoor-train.data')

def outdoorTrainLabelsPath():
    return os.path.join(stationaryPrefix, 'Outdoor/Outdoor-train.labels')

def outdoorTestSamplesPath():
    return os.path.join(stationaryPrefix, 'Outdoor/Outdoor-test.data')

def outdoorTestLabelsPath():
    return os.path.join(stationaryPrefix, 'Outdoor/Outdoor-test.labels')

def coilTrainSamplesPath():
    return os.path.join(stationaryPrefix, 'COIL/COIL-train.data')

def coilTrainLabelsPath():
    return os.path.join(stationaryPrefix, 'COIL/COIL-train.labels')

def coilTestSamplesPath():
    return os.path.join(stationaryPrefix, 'COIL/COIL-test.data')

def coilTestLabelsPath():
    return os.path.join(stationaryPrefix, 'COIL/COIL-test.labels')

def borderTrainSamplesPath():
    return os.path.join(stationaryPrefix, 'Border/border-train.data')

def borderTrainLabelsPath():
    return os.path.join(stationaryPrefix, 'Border/border-train.labels')

def borderTestSamplesPath():
    return os.path.join(stationaryPrefix, 'Border/border-test.data')

def borderTestLabelsPath():
    return os.path.join(stationaryPrefix, 'Border/border-test.labels')

def overlapTrainSamplesPath():
    return os.path.join(stationaryPrefix, 'Overlap/overlap-train.data')

def overlapTrainLabelsPath():
    return os.path.join(stationaryPrefix, 'Overlap/overlap-train.labels')

def overlapTestSamplesPath():
    return os.path.join(stationaryPrefix, 'Overlap/overlap-test.data')

def overlapTestLabelsPath():
    return os.path.join(stationaryPrefix, 'Overlap/overlap-test.labels')

