# -*- coding: utf-8 -*-
import numpy as np
import os
from keras.layers import containers, AutoEncoder, Dense, Dropout
from keras import models

from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape
from autoencoder_layers import DependentDense, Deconvolution2D, DePool2D

def compress_data_by_autoencoder(X, target_dim, hidden_dim, weightsFile=None):
    input_dim = X.shape[1]
    
    # input shape: (nb_samples, 32)
    encoder = containers.Sequential([
        Dense(hidden_dim, input_dim=input_dim, activation='relu'), 
        Dropout(0.2),
        Dense(hidden_dim, activation='relu'),
        Dropout(0.2),
        Dense(hidden_dim, activation='relu'),
        Dropout(0.2),
        Dense(target_dim, activation='relu'), 
    ])
    
    decoder = containers.Sequential([
        Dense(hidden_dim, input_dim=target_dim, activation='relu'),
        Dropout(0.2),
        Dense(hidden_dim, activation='relu'),
        Dropout(0.2),
        Dense(hidden_dim, activation='relu'),
        Dropout(0.2),
        Dense(input_dim, activation='relu')
    ])
    
    autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True)
    model = models.Sequential()
    model.add(autoencoder)
    
   
    # training the autoencoder:
    model.compile(optimizer='sgd', loss='mse')
    
    if ( type(weightsFile) is str and os.path.isfile(weightsFile) ):
        model.load_weights(weightsFile)
    
    error = 1
    max_nb_epoch = 1000
    epoch = 0
    while (error > 0.005 and epoch < max_nb_epoch):
        fited = model.fit(X, X, nb_epoch=1, batch_size=64) # 
        error = fited.totals['loss']/64
        epoch += 1
    
    # predicting compressed representations of inputs:
    autoencoder.output_reconstruction = False  # the model has to be recompiled after modifying this property
    model.compile(optimizer='sgd', loss='mse')
    representations = model.predict(X)
    
    
    if ( type(weightsFile) is str ):
        model.save_weights(weightsFile, overwrite=True)
        
    return representations
    
##########################################################################
def compress_data_by_conv_autoencoder(X, target_dim, hidden_dim, weightsFile=None):
    input_dim = X.shape[1]
    dataset_size = X.shape[0]
    nb_filters = 5    
    nb_conv = 5
    nb_pool = 2
    
    X = X.reshape(dataset_size, 1, input_dim, 1)
    
    hidden_dense_layer1 = Dense(hidden_dim)
    hidden_dense_layer2 = Dense(hidden_dim)
    
    conv_layer1 = Convolution2D(nb_filters, nb_conv, 1, 
                       border_mode='same', 
                       input_shape=(1, input_dim, 1))
    mp_layer1 = MaxPooling2D(pool_size=(nb_pool, 1))
    
#    conv_layer2 = Convolution2D(nb_filters, nb_conv, 1,
#                       border_mode='same', 
#                       input_shape=(1, input_dim, 1))
                       
#    conv_layer3 = Convolution2D(nb_filters, nb_conv, 1,
#                       border_mode='same', 
#                       input_shape=(1, input_dim, 1))
#                       
#    conv_layer4 = Convolution2D(nb_filters, nb_conv, 1,
#                       border_mode='same', 
#                       input_shape=(1, input_dim, 1))
    #mp_layer2 = MaxPooling2D(pool_size=(nb_pool, 1))
 
    encoder = models.Sequential()
    encoder.add(conv_layer1)
    encoder.add(Activation("relu"))

#    encoder.add(Dropout(0.2))
#    
#    encoder.add(conv_layer2)
#    encoder.add(Activation("relu"))
    encoder.add(mp_layer1)
    #encoder.add(mp_layer2)
    encoder.add(Dropout(0.2))
    
    encoder.add(Flatten())
    encoder.add(hidden_dense_layer1)
    encoder.add(Activation("relu"))
    encoder.add(Dropout(0.2))
    encoder.add(hidden_dense_layer2)
    encoder.add(Activation("relu"))
    encoder.add(Dropout(0.2))
    encoder.add(Dense(target_dim))
    encoder.add(Activation("relu"))
    ######### decoder ###############
    decoder = models.Sequential()
    decoder.add(Dense(hidden_dim, input_dim=target_dim, activation='relu'))
    decoder.add(Dropout(0.2))    
    decoder.add(DependentDense( (nb_filters * input_dim//nb_pool),  hidden_dense_layer1))
    decoder.add(Activation('relu'))
    decoder.add(Dropout(0.2))    
    decoder.add(Reshape((nb_filters, input_dim//nb_pool, 1)))
    decoder.add(DePool2D(mp_layer1, size=(nb_pool, 1)))
    
    
    decoder.add(Deconvolution2D(conv_layer1, border_mode='same'))
    decoder.add(Activation('relu'))    
    
#    decoder.add(Deconvolution2D(conv_layer4, border_mode='same'))
#    decoder.add(Activation('relu'))
    



    
    autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True)
    model = models.Sequential()
    model.add(autoencoder)
    
   
    # training the autoencoder:
    model.compile(optimizer='sgd', loss='mse')
    
    if ( type(weightsFile) is str and os.path.isfile(weightsFile) ):
        model.load_weights(weightsFile)
    
    model.fit(X, X, nb_epoch=50, batch_size=5) # 
        
    # predicting compressed representations of inputs:
    autoencoder.output_reconstruction = False  # the model has to be recompiled after modifying this property
    model.compile(optimizer='sgd', loss='mse')
    representations = model.predict(X)
    
    
    if ( type(weightsFile) is str ):
        model.save_weights(weightsFile, overwrite=True)
    
    
    return representations
    


"""
        dataset = { "input": datasetOfSpikes, "target": datasetOfSpikes }
        layers = [
            {"n": self.window_size*resamplingFactor, "bias": True, },
            {"n": 40, "bias": True, },
            {"n": 40, "bias": True, },
            {"n": 10, "bias": True, },
            {"n": 40, "bias": True,},
            {"n": 40, "bias": True, },
            {"n": self.window_size*resamplingFactor, "bias": True,}
        ]

        autoencoder = mynet.Network(layers, dataset["input"].shape[1])
        if ( loadWeights and os.path.exists(weightsFile) ):
            weightsData = io.loadmat(weightsFile)
            weights = []
            for idx in range(len(layers)):
                weights.append(weightsData[str(idx)])
            autoencoder.setWeights(weights)

        save_err = autoencoder.trainOnDataset(dataset, 0.001, 100000, lr, 100)
        print (save_err)
        weights = autoencoder.getWeights()
        if (saveWeights):
            weightsData = {"data":{}}
            for idx, l in enumerate(weights):
                weightsData["data"][str(idx)] = l
            io.savemat(weightsFile, weightsData["data"])

        coder = mynet.Network(layers[0:4], dataset["input"].shape[1])
        coder.setWeights(weights[0:4])
        compress_data = coder.activate(dataset["input"])
"""        