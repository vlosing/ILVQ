import os
import copy
import shutil
import json
import pickle

from PIL import Image
import scipy.ndimage
import numpy

from natsort import natsort

# # read label classes and action definitions from JSON file
# #
## Label classes and action definitions are stored in global variable setupClasses.
## @param path class definition file path (typically data/setups/DefaultClass.json
def fetchSetupClasses(path):
    fp = open(path, "r")
    setupClasses = json.load(fp)
    fp.close()
    return setupClasses

def loadImage(width, height, filename):
    m = Image.open(filename)
    m = m.resize([width, height])
    m = scipy.misc.fromimage(m)
    return m

def loadImage2(filename):
    m = Image.open(filename)
    m = scipy.misc.fromimage(m)
    return m


def saveImage(image, filename, reshape=None, reshapeHeight=None, reshapeWidth=None):
    tmp = image.copy();
    if reshape == True:
        tmp.shape = [reshapeHeight, reshapeWidth, 3]
    pic = scipy.misc.toimage(tmp);
    pic.save(filename);


## load masks, resize them appropriately, covert to 2d numpy array
##
## these binary masks define areas in pixel space (which areas are left, right, close of robot)
## ignore: if robot is in the (lower part of the) image, we can set this to ignore this area
## all masks are expected in ./masks/ directory
def loadMasks(width, height, load=('left', 'right', 'ignore')):
    masks = dict()
    for mname in load:
        m = Image.open(('../masks/%s' + '.png') % mname)
        m = m.resize([width, height])
        m = scipy.misc.fromimage(m)
        if len(m.shape) > 2: m = m[:, :, 0]
        masks[mname] = m

    return masks


class RingBuffer:
    """ Buffer lists of objects and try to recycle buffers"""

    def __init__(self, lenHistory):
        self.lenHistory = lenHistory
        self.buf = lenHistory * [None]
        self.validEntries = 0
        self.nextNew = 0

    def clear(self):
        self.validEntries = 0
        self.nextNew = 0

    def add(self, entryList):
        if self.buf[self.nextNew] is None:
            self.buf[self.nextNew] = copy.deepcopy(entryList)
        else:
            assert len(entryList) == len(self.buf[self.nextNew])
            for i in range(len(entryList)):
                try:
                    self.buf[self.nextNew][i][:] = entryList[i]  # reuse existing buffer (for numpy arrays)
                except:
                    self.buf[self.nextNew][i] = entryList[i]  # just copy
        self.validEntries = min(self.validEntries + 1, self.lenHistory)
        self.nextNew = (self.nextNew + 1) % self.lenHistory

    def getHistory(self):
        out = list()
        for i in range(self.validEntries):
            pos = (self.nextNew - self.validEntries + i) % self.lenHistory
            out.append(self.buf[pos])
        return out


def saveImages(srcDir, fileNames, dstDir):
    if not os.path.exists(dstDir):
        os.makedirs(dstDir)
    for i in range(len(fileNames)):
        srcPath = srcDir + fileNames[i]
        if os.path.exists(srcPath):
            dstPath = dstDir + fileNames[i]
            shutil.copy(srcPath, dstPath)


class SimpleFileIterator:
    directoryName = ''
    fileSuffix = ''
    fileNames = []
    curIdx = -1

    def __init__(self, directoryName, fileSuffix, revertOrder=False):
        self.directoryName = directoryName
        self.fileSuffix = fileSuffix
        self.fileNames = os.listdir(directoryName)
        self.fileNames = natsort(self.fileNames)
        if revertOrder:
            self.fileNames = self.fileNames[::-1]

    def getCurrentFileName(self):
        return self.fileNames[self.curIdx]

    def getNextFileName(self):
        while True:
            self.curIdx += 1
            if (self.curIdx >= len(self.fileNames)):
                return None
            else:
                fileName = self.fileNames[self.curIdx]
                if fileName.endswith(self.fileSuffix):
                    return fileName

    def skipFiles(self, number):
        self.curIdx += number

    def reset(self):
        self.curIdx = -1

    def getFileNamesInRange(self, startName, endName):
        result = []
        for fileName in self.fileNames:
            if fileName > startName:
                if fileName > endName:
                    break
                result.append(fileName)

        return result

    def getFileNames(self):
        return self.fileNames






def loadPickleFile(fileNamePath):
    f = open(fileNamePath, 'r')
    try:
        data = pickle.load(f)
    finally:
        f.close()
    return data

def loadJsonFile(fileNamePath):
    try:
        f = open(fileNamePath, 'r')
        try:
            data = json.load(f)
        finally:
            f.close()
        return data
    except ValueError:
        print('valError ' + fileNamePath)
        return None