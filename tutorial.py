import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from asyncore import write
import tensorflow as tf
import numpy as np
from datetime import datetime
import keras_Sequencial as ks


#Standardização
data = pd.read_csv('concrete.csv')  
data = shuffle(data)
data = data.to_numpy()
means = np.mean(data,axis=0)
stds = np.std(data,axis=0)
data = (data-means)/stds

#Conjunto de validação
valid_Y = data[700:,[-1]]
valid_X = data[700:,0:8]

#Conjunto de treino
Y = data[:700,[-1]]
X = data[:700,0:8]


#######################################################################################################################
#Tensorboard#
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "logsConcrete"
log_dir = "{}/model-{}/".format(root_logdir, now)
writer = tf.summary.create_file_writer(log_dir)

@tf.function
def create_graph(X):
    _ = predict(X)

def write_graph(X):
    tf.summary.trace_on(graph=True)
    create_graph(tf.constant(X.astype(np.float32)))
    with writer.as_default():
        tf.summary.trace_export(name="trace",step=0)
#######################################################################################################################

def layer(inputs,neurons,layer_name):
  weights = tf.Variable(tf.random.normal((inputs.shape[1],neurons), stddev = 1/neurons ))
  bias = tf.Variable(tf.zeros([neurons]))
  return weights,bias

def create_network(X,layers):
    network = []
    variables = []
    previous = X
    for ix, neurons in enumerate(layers):
        weights,bias = layer(previous,neurons,f'layer_{ix}')
        network.append( (weights,bias) )
        variables.extend( (weights,bias) )
        previous = weights
    return network, variables

    
layers = [4,4,1]
network ,variables = create_network(X,layers)

def predict(X):
    net = X
    layer = 1
    for weights,bias in network[:-1]:
        with tf.name_scope(f'Layer_{layer}'):
            net = tf.add(tf.matmul(net, weights), bias,name='net')
            net = tf.nn.leaky_relu(net, name="relu")
        layer += 1
    weights,bias = network[-1]
    with tf.name_scope('Output'):
        net = tf.add(tf.matmul(net, weights), bias)
    return tf.reshape(net, [-1])

write_graph(X)

def loss(X,Y):
    yLabelsTensorflow = tf.constant(Y)
    # individual
    squaredErrorDifs = tf.math.square(yLabelsTensorflow - X)
    # for tensor
    meanSqLoss = tf.reduce_mean(squaredErrorDifs)
    return meanSqLoss

def grad(X, y, layerVars):
    with tf.GradientTape() as tape:
        loss_val = predict(X)
        loss_cost_value = loss(loss_val, y)
    return tape.gradient(loss_cost_value, layerVars), layerVars


optimizer = tf.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
batch_size = 128
batches_per_epoch = X.shape[0]//batch_size
epochs = 5000


trainingErrorSum = 0
validationErrorSum = 0
allLossesTrain=[]
allLossesValid=[]

print("########## \n TENSORFLOW VANILLA \n##########")

for epoch in range(epochs):
    shuffled = np.arange(len(Y))
    np.random.shuffle(shuffled)
    for batch_num in range(batches_per_epoch):
        start = batch_num*batch_size
        batch_xs = tf.constant(X[shuffled[start:start+batch_size],:].astype(np.float32))
        batch_ys = tf.constant(Y[shuffled[start:start+batch_size]].astype(np.float32))
        gradients,variables = grad(batch_xs, batch_ys, variables)
        optimizer.apply_gradients(zip(gradients, variables))

    ys_Pred_Train = predict(tf.constant(X.astype(np.float32)))
    ys_Pred_Val = predict(tf.constant(valid_X.astype(np.float32)))

    lossTrain = loss(tf.constant(ys_Pred_Train),tf.constant(Y.astype(np.float32)))**0.5*stds[0]
    lossVal = loss(tf.constant(ys_Pred_Val),tf.constant(Y.astype(np.float32)))**0.5*stds[0]

    print("Epoch: " +str(epoch) +" Training Error: " + str(lossTrain.numpy()) + " Validation Error: " + str(lossVal.numpy()))

    with writer.as_default():
        tf.summary.scalar("Train loss", lossTrain, step=epoch)
        tf.summary.scalar("Val loss", lossVal, step=epoch)

    allLossesTrain.append(lossTrain)
    allLossesValid.append(lossVal)
    trainingErrorSum = (trainingErrorSum + lossTrain)
    validationErrorSum = (validationErrorSum + lossVal)
writer.close()

trainingErrorAvg = (trainingErrorSum / epoch)
validationErrorAvg = (validationErrorSum / epoch)


print("Average MinSqLoss w Training Set: " + str(trainingErrorAvg))
print("Average MinSqLoss w Validation Set: " + str(validationErrorAvg))


############################# Gráficos
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(range(1,epochs+1), allLossesTrain, 'b', label='Training set')
plt.plot(range(1,epochs+1), allLossesValid, 'r', label='Validation set')
plt.title('Losses (Training and Validation)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()


ks.execKerasSequencial(X, Y, valid_X, valid_Y)