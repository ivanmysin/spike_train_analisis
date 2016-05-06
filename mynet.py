# -*- coding: utf-8 -*-
"""
my artifitional network

"""

import numpy as np
import time

class FullConnectionLayer:
    def __init__ (self, inputSize, outputSize, bias = True):
        self.bias = bias
        if (self.bias):
            self.inputSize = inputSize + 1
        else:
            self.inputSize = inputSize
        self.outputSize = outputSize
        
        self.w = np.random.normal(0, 0.1, [self.outputSize, self.inputSize])
        self.dw = np.zeros([outputSize, inputSize])
        self.layer_state = np.random.rand(outputSize, 1)

        
        self.grad = np.zeros(self.outputSize)
        self.deep_grad = np.zeros(self.inputSize)
    
    def forward (self, inp):
        # print (inp)
        if (self.bias):
            self.last_input = np.append( inp, np.ones( [inp.shape[0], 1]), axis=1 ).T
        else: 
            self.last_input = inp.T
        #print (self.last_input.shape, self.w.shape)
        
        self.layer_state = self.activation( np.dot(self.w, self.last_input) ).T
        return self.layer_state
    
    def activation (self, x):
        return  1 / (1 + np.exp(-x))
    
    def activation_derivative (self, x):
        return x*(1-x)
   
    def backward (self, error):
        derivative = self.activation_derivative(self.layer_state)
        self.grad = error * derivative
        self.deep_grad = np.dot(self.w.T, self.grad.T).T
        
        self.dw = self.last_input.dot( self.grad )
        if (self.bias):        
            self.deep_grad = self.deep_grad[:, :-1]
        return self.deep_grad
    
    def update_weights (self, lr=0.01):
        
        self.w += lr*self.dw.T
        
    def getW (self):
        return self.w
        
    def setW (self, w):
        self.w = w
        
    def getState (self):
        return self.layer_state
###############################################################################
class Network:
    def __init__ (self, layers, networkInputSize):
        self.n_layers = len (layers)
        self.last_input = np.zeros( [1, networkInputSize] )
        
        self.layers = []
        for idx, l in enumerate(layers):
            if (idx==0): 
                layer = FullConnectionLayer(networkInputSize, l["n"], l["bias"])
            else:
                layer = FullConnectionLayer(layers[idx-1]["n"], l["n"], l["bias"])
            self.layers.append(layer)
        
        self.last_output = np.zeros(  [1, l["n"]] )
        
    def activate (self, networkInput):
        layerOutput = networkInput
        for layer in self.layers:
            layerOutput = layer.forward( layerOutput )
        self.last_output = layerOutput
        return self.last_output  
    
        
        
    def trainOnSample(self, networkInput, target, learningRate):
        
        networkOutput = self.activate(networkInput)
        gradient = target - networkOutput 
        error = 0.5 * np.sum ( (target - networkOutput)**2 )
        # backward propagation
        for lind in range(self.n_layers):
            back_lind = self.n_layers - lind - 1
            gradient = self.layers[back_lind].backward( gradient )


        # update weights
        for layer in self.layers:
            layer.update_weights(learningRate)
        
        return error
        
    def trainOnDataset(self, dataset, minimalError, 
                           maximalIterationNumber, learningRate=0.05,
                           minibatchSize=15, returnError=False):
        error = 1
        epochIterator = 0
        if (returnError):
            saved_error = np.array([])
        datasetSize = dataset["input"].shape[0]
        while (error > minimalError and epochIterator < maximalIterationNumber):
            el_ind = np.random.choice(datasetSize, minibatchSize)
            error = self.trainOnSample( dataset["input"][el_ind], dataset["target"][el_ind], learningRate)
            error /= minibatchSize            
            if (returnError):            
                saved_error  = np.append(saved_error , error)
            epochIterator += 1
        if (returnError):
            return saved_error
        else:
            return error
            
    def getWeights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.getW())
        return weights
    
    def setWeights(self, weights):
        for idx, layer in  enumerate(self.layers):
            layer.setW( weights[idx] )

        
###############################################################################
"""
dataset = {
    "input": [ np.array([[0, 0]]), np.array([[1, 0]]), np.array([[0, 1]]), np.array([[1, 1]])],
    "target": [np.array([[0, 0, 0]]), np.array([[1, 0, 0]]), np.array([[1, 0, 0]]), np.array([[0, 0, 0]])],
}

"""
if __name__ == "__main__":

    datasetSize = 2000
    lr = 0.1
    
    data = np.random.rand(datasetSize, 70)
    dataset = {
        "input": data,
        "target": data,
    }



    layers = [
        {
            "n": 70,
            "bias": True,
        },
        {
            "n": 70,
            "bias": True,
        },
        {
            "n": 10,
            "bias": True,
        },
        {
            "n": 70,
            "bias": True,
        },
        {
            "n": 70,
            "bias": True,
        }
    
    ]


    ct = time.time()
    net = Network(layers, dataset["input"].shape[1])
    save_err = net.trainOnDataset(dataset, 0.1, 1000, lr)
    
    print (time.time() - ct)    
    import matplotlib.pyplot as plt
    plt.plot(save_err)