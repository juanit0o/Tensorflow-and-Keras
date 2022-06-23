from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Dense
from tensorflow.keras.layers import MaxPooling2D, Activation, Flatten, Dropout
import tensorflow as tf
import keras_Sequencial as ks



def execKerasFuncional(X, Y, valid_X, valid_Y):
    print("########## \n KERAS SEQUENCIAL \n##########")
    inputs = Input(shape=(8, ),name='inputs')
    layer = Dense(4)(inputs)
    layer = Activation(LeakyReLU())(layer)
    layer = BatchNormalization(axis=-1)(layer)
    ####
    layer = Dense(4)(layer)
    layer = Activation(LeakyReLU())(layer)
    layer = BatchNormalization(axis=-1)(layer)
    #####
    layer = Dense(1)(layer)


    functionalKerasModel = Model(inputs = inputs, outputs = layer)   
    functionalKerasModel.summary()

    NUM_EPOCHS = 5000
    BS= 128
    opt = SGD(lr=0.00001, momentum=0.9)

    functionalKerasModel.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=opt, metrics=["accuracy"])
    functionalKerasModel.summary()
    fittedModel = functionalKerasModel.fit(X, Y,validation_data=(valid_X, valid_Y),batch_size=BS, epochs=NUM_EPOCHS)

    ks.plotLosses(fittedModel)