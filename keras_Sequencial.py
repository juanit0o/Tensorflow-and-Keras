from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_Funcional as kf



def execKerasSequencial(X, Y, valid_X, valid_Y):
    print("########## \n KERAS SEQUENCIAL \n##########")
    
    INIT_LR = 0.001
    NUM_EPOCHS = 5000
    BS = 128

    opt = SGD(lr = INIT_LR, momentum=0.9, decay=INIT_LR / NUM_EPOCHS)
    model = create_model()
    model.compile(loss= tf.keras.losses.MeanSquaredError(), optimizer=opt,metrics=["accuracy"])
    model.summary()

    history = model.fit(X, Y, validation_data=(valid_X, valid_Y),
                  batch_size=BS, epochs=NUM_EPOCHS)

    plotLosses(history)

    loss, acc = model.evaluate(valid_X, valid_Y)
    print('Validation Loss: ' +str(loss))
    kf.execKerasFuncional(X, Y, valid_X, valid_Y)



def create_model():
    #se aplicarmos a batch normalization antes do relu, o relu vai po los nao standardizado
    model = Sequential()
    model.add(Dense(4, input_shape = (8,)))
    model.add(Activation("leaky_relu"))
    model.add(BatchNormalization())

    model.add(Dense(4, input_shape = (8,)))
    model.add(Activation("leaky_relu"))
    model.add(BatchNormalization())
    
    model.add(Dense(1))
    return model


def plotLosses(history):
    x = range(1, len(history.history['loss']) + 1)
    trainingLoss = history.history['loss']
    validationLoss = history.history['val_loss']
    plt.plot(x, trainingLoss, 'b', label='Training loss')
    plt.plot(x, validationLoss, 'r', label='Validation loss')
    plt.title('Losses (Training and validation)')
    plt.xlabel('epoch')
    plt.legend()
    
