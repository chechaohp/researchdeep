import tensorflow as tf 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib
import time
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

np.random.seed(1000)

# Load data
mnist = input_data.read_data_sets('MNIST_data',one_hot = False)
temp = mnist.train
images, labels = temp.images, temp.labels
images, labels = shuffle(np.asarray(images), np.asarray(labels))
epoch_num = 100
learning_rate = 0.001

# Hyper parameters

G_input = 100
hidden_input1, hidden_input2, hidden_input3 = 128, 256, 346
hidden_input4, hidden_input5, hidden_input6 = 480, 560, 686

# Model

X = tf.placeholder(dtype = tf.float32, name = 'X', shape = images.shape)
Y = tf.placeholder(dtype = tf.float32, name = 'Y', shape = labels.shape)
# parameters to Discriminator Net
D_W1 = tf.Variable(tf.zeros(shape = [images.shape[1],hidden_input1], dtype = tf.float32),name = 'D_W1')
D_b1 = tf.Variable(0.0, dtype = tf.float32, name = 'D_b1')
D_W2 = tf.Variable(tf.zeros(shape = [hidden_input1,1], dtype = tf.float32), name = 'D_W2')
D_b2 = tf.Variable(0.0, dtype = tf.float32, name = 'D_b2')

# parameters for Generator Net
G_W1 = tf.Variable(tf.zeros(shape = (G_input, hidden_input1),dtype = tf.float32),name = 'G_W1')
G_b1 = tf.Variable(0.0, dtype = tf.float32, name = 'G_b1') 
G_W2 = tf.Variable(tf.zeros(shape = (hidden_input1, hidden_input2), dtype = tf.float32), name = 'G_W2')
G_b2 = tf.Variable(0.0, dtype = tf.float32, name = 'G_b2')
G_W3 = tf.Variable(tf.zeros(shape = (hidden_input2, hidden_input3), dtype = tf.float32), name = 'G_W3')
G_b3 = tf.Variable(0.0, dtype = tf.float32, name = 'G_b3')
G_W4 = tf.Variable(tf.zeros(shape = (hidden_input3, hidden_input4), dtype = tf.float32), name = 'G_W4')
G_b4 = tf.Variable(0.0, dtype = tf.float32, name = 'G_b4')
G_W5 = tf.Variable(tf.zeros(shape = (hidden_input4, hidden_input5), dtype = tf.float32), name = 'G_W5')
G_b5 = tf.Variable(0.0, dtype = tf.float32, name = 'G_b5')
G_W6 = tf.Variable(tf.zeros(shape = (hidden_input5, hidden_input6), dtype = tf.float32), name = 'G_W6')
G_b6 = tf.Variable(0.0, dtype = tf.float32, name = 'G_b6')
G_W7 = tf.Variable(tf.zeros(shape = (hidden_input6, 784), dtype = tf.float32), name = 'G_W7')
G_b7 = tf.Variable(0.0, dtype = tf.float32, name = 'G_b7')

# Z = tf.random_uniform(minval = -1,maxval = 1, dtype = tf.float32, name = 'Z',shape = [1,G_input])

# Generator Net
# G11 = tf.matmul(Z,G_W1) + G_b1
# G11A = tf.atan(G11)
# G12 = tf.matmul(G11A,G_W2) + G_b2
# G12A = tf.nn.relu(G12)
# G13 = tf.matmul(G12A, G_W3) + G_b3
# G13A = tf.atan(G13)

# G14 = tf.matmul(G13A, G_W4) + G_b3
# G14A = tf.nn.relu(G14)
# G15 = tf.matmul(G14A, G_W5) + G_b3
# G15A = tf.tanh(G15)
# G16 = tf.matmul(G15A, G_W6) + G_b3
# G16A = tf.nn.relu(G16)
# G17 = tf.matmul(G16A, G_W7) + G_b3
# G17A = tf.sigmoid(G17, name = 'output G')

# Discriminator
def discrimininator(x):
    D11 = tf.matmul(x, D_W1) + D_b1
    D11A = tf.nn.relu(D11)
    D12 = tf.matmul(D11A,D_W2) + D_b2
    D12A = tf.log(D12)
    return D12A

# Generator
def generator(z):
    G11 = tf.matmul(z,G_W1) + G_b1
    G11A = tf.atan(G11)
    G12 = tf.matmul(G11A,G_W2) + G_b2
    G12A = tf.nn.relu(G12)
    G13 = tf.matmul(G12A, G_W3) + G_b3
    G13A = tf.atan(G13)
    G14 = tf.matmul(G13A, G_W4) + G_b3
    G14A = tf.nn.relu(G14)
    G15 = tf.matmul(G14A, G_W5) + G_b3
    G15A = tf.tanh(G15)
    G16 = tf.matmul(G15A, G_W6) + G_b3
    G16A = tf.nn.relu(G16)
    G17 = tf.matmul(G16A, G_W7) + G_b3
    G17A = tf.sigmoid(G17)
    return G17A

Z = tf.placeholder( dtype = tf.float32, name = 'Z',shape = [1,G_input])

# Cost function
G_sample = generator(Z)

D_fake = discriminator(G_sample)
D_real = discriminator(X)

D_loss = -tf.log(D_real) + tf.log(1.0-D_fake)
G_loss = -tf.log(D_fake)

D_cost = tf.reduce_mean(D_loss)
G_cost = tf.reduce_mean(G_loss)

D_optimizer = tf.train.AdamOptimizer(learning_rate= 0.00009, beta1 = 0.9, beta2 = 0.999, epsilon)

#
with tf.Session() as sess:
    grad_f_w2 = 