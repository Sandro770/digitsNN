import numpy as np
import math 
from os import system
from keras.datasets import mnist
import copy
class Errors:
    def __init__(self):
        self.a = []
        self.b = []
        self.z = []
        self.w = []
        self.cntAdd = 0
    def add(self, toAdd):
        if(self.cntAdd == 0):
            self.b = toAdd.b
            self.w = toAdd.w
        else:
            self.b = np.add(self.b, toAdd.b)    
            self.w = np.add(self.w, toAdd.w)  
        self.cntAdd += 1
    def takeAverage(self):
        for i in range(0,len(self.b)):
            self.b /= self.cntAdd
        for i in range(0,len(self.w)):
            for j in range(0,len(self.w[i])):
                self.w[i][j] /= self.cntAdd
        self.cntAdd = 0
class Layer:
    def __init__(self):
        self.activations = []
        self.weights = []
        self.biases = []
        self.z = []
        self.expectedValues = []
        self.costs = []
        self.szNxt = 0
        self.sz = 0
        self.correctDigit = -1
        self.errors = Errors()
        self.sumErrors = Errors()
    def setBiases(self):
        B = []
        for i in range(0, self.sz):
            B.append(0)
        self.biases = B
    def setWeights(self):
        weights = []
        weights = np.random.randn(self.sz,self.szNxt) * np.sqrt(1/self.sz)
        self.weights = weights
    def buildLayer(self, szAct, szNxt, szPrev):
        self.sz = szAct
        self.szNxt = szNxt
        self.szPrev = szPrev
        self.setBiases()
        self.setWeights()
    def sigmoid(self, x):
        try:
            sig = 1 / (1 + math.exp(-x))
        except:
            print(-x)
            exit(0)
        return sig
    def sigDer(self, x):
        sig = self.sigmoid(x)
        res = sig * (1 - sig)
        return res
    def squish(self):
        self.activations = [] 
        for z in self.z:
            sig = self.sigmoid(z)
            self.activations.append(sig)
    def setInput(self, input):
        input = copy.deepcopy(input)
        self.activations = input
        self.z = input
    def getCost(self, actual, expected):
        cost = (actual - expected) * (actual - expected)
        return cost
    def setCorrectDigit(self, digit):
        self.expectedValues.clear()
        self.correctDigit = digit
        for i in range(0,11):
            if(i == digit):
                self.expectedValues.append(1)
            else:
                self.expectedValues.append(0)
    def countCosts(self):
        self.costs.clear()
        costs = []
        for i in range(0,10):
            actual = self.activations[i]
            expected = self.expectedValues[i]
            costs.append(self.getCost(actual,expected))
        self.costs = costs
class NeuralNetwork:
    def __init__(self):
        self.epochs = 100
        self.layers = []
        self.nbOfLayers = 0
    def setLayers(self, sizes):
        nbOfLayers = len(sizes)
        self.nbOfLayers = nbOfLayers
        if(nbOfLayers < 3):
            print("Error, nb Of Layers shouldn't be less than 3")
            return
        for layerId in range(0,nbOfLayers):
            sz = sizes[layerId]
            szNxt = 0
            if(layerId != nbOfLayers - 1):
                szNxt = sizes[layerId + 1]
            layer = Layer()
            szPrev = 0
            if(layerId != 0):
                szPrev = self.layers[layerId-1].sz
            layer.buildLayer(sz,szNxt,szPrev)
            self.layers.append(layer)
    def setInputLayer(self, input):
        self.layers[0].setInput(input)
    def setInput(self, input, lbl):
        if(len(input) != self.layers[0].sz):
            print("input size is not equal to size of first layer")
            return
        self.setInputLayer(input)
        self.layers[-1].setCorrectDigit(lbl)
    def propagateForward(self, prevLayer, actLayer):
        newZ = np.add(np.dot(prevLayer.activations, prevLayer.weights),actLayer.biases)
        actLayer.z = newZ
        actLayer.squish()
        return actLayer
    def forwardPropagation(self):
        for i in range(0,self.nbOfLayers - 1):
            actLayer = self.layers[i]
            nxtLayer = self.layers[i + 1]
            nxtLayer = self.propagateForward(actLayer,nxtLayer)
            self.layers[i + 1] = nxtLayer
        self.layers[-1].countCosts()
    def cntAErrorsForOutputLayer(self):
        aErrors = []
        layer = self.layers[-1]
        for i in range(0,layer.sz):
            act = layer.activations[i]
            expected = layer.expectedValues[i]
            aErrors.append(2 * (act - expected))
        self.layers[-1].errors.a = aErrors
    def cntAErrors(self, layerId):
        layer = self.layers[layerId]
        nxtLayer = self.layers[layerId + 1]
        aErrors = []
        for i in range(0,layer.sz):
            sum = 0
            for j in range(0,nxtLayer.sz):
                sum += layer.weights[i][j] * nxtLayer.errors.z[j]
            aErrors.append(sum)
        self.layers[layerId].errors.a = aErrors
    def cntZErrors(self, layerId):
        layer = self.layers[layerId]
        zErrors = []
        for i in range(0,layer.sz):
            z = layer.z[i]
            sigDer = layer.sigDer(z)
            zErr = sigDer * layer.errors.a[i]
            zErrors.append(zErr)
        self.layers[layerId].errors.z = zErrors
    def cntBErrors(self, layerId):
        self.layers[layerId].errors.b = self.layers[layerId].errors.z
    def cntWErrors(self, layerId):
        actLayer = self.layers[layerId]
        nxtLayer = self.layers[layerId + 1]
        wErrors = []
        for i in range(0,actLayer.sz):
            row = []
            for j in range(0, nxtLayer.sz):
                error = actLayer.activations[i] * nxtLayer.errors.z[j]
                row.append(error)
            wErrors.append(row)
        self.layers[layerId].errors.w = wErrors
    def backPropagation(self):
        for i in range(1,self.nbOfLayers + 1):
            if(i == 1): 
                self.cntAErrorsForOutputLayer()
            else:
                self.cntAErrors(-i)
            if(i != self.nbOfLayers):
                self.cntZErrors(-i)
                self.cntBErrors(-i)
            if(i != 1): 
                self.cntWErrors(-i)
    def updateBiasesAndWeights(self):
        for i in range(0,len(self.layers)):
            layer = self.layers[i]
            if(i != 0):
                layer.biases = np.subtract(layer.biases, np.array(layer.errors.b))
            if(i != len(self.layers) - 1):
                layer.weights = np.subtract(layer.weights, np.array(layer.errors.w))
            self.layers[i] = layer
    def runSingleTest(self):
        self.forwardPropagation()
        self.backPropagation()
    def addToSumGradient(self):
        for i in range(0,len(self.layers)):
            layer = self.layers[i]
            toAdd = layer.errors
            self.layers[i].sumErrors.add(toAdd)
    def takeAverageGradient(self):
        for i in range(0,len(self.layers)):
            self.layers[i].sumErrors.takeAverage()
            self.layers[i].errors = self.layers[i].sumErrors
    def isCorrect(self):
        lastLayer = self.layers[-1]
        correctDigit = lastLayer.correctDigit
        actForCorr = lastLayer.activations[correctDigit]
        for i in range(0,10):
            if(i != correctDigit and lastLayer.activations[i] >= actForCorr):
                return 0
        return 1
    def train(self, train_imgs, train_lbls, tst_imgs, tst_lbls):
        for i in range(0,self.epochs):
            cnt = 0
            for input, lbl in zip(train_imgs, train_lbls):
                cnt += 1
                self.setInput(input, lbl)
                self.runSingleTest()
                self.addToSumGradient()
                if(cnt % 50 == 0):
                    self.takeAverageGradient()
                    self.updateBiasesAndWeights()
        passed = 0
        cnt = 0
        for input, lbl in zip(tst_imgs, tst_lbls):
            cnt += 1
            self.setInput(input, lbl)
            self.runSingleTest()
            if(self.isCorrect()):
                passed += 1
        print("ratio[" + str(cnt) + "]: " + str(passed/cnt))
    
network = NeuralNetwork()
network.setLayers([784,16,16,10])

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X / 255
test_X = test_X / 255

def split(setOfImgs):
    newSet = []
    for img in setOfImgs:
        vect = []
        for row in img:
            for pixel in row:
                vect.append(pixel)
        newSet.append(vect)
    return newSet
print(train_X.shape)
train_X = split(train_X)
test_X = split(test_X)
network.train(train_X,train_y,test_X,test_y)
