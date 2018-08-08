import numpy as np 
import time
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle

# Read data
print('#LOAD DATA')
mnist = input_data.read_data_sets('MNIST_data',one_hot = False)
temp = mnist.train
images, labels = temp.images, temp.labels
images, labels = shuffle(np.asarray(images), np.asarray(labels))
