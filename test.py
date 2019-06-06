import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#for IPython, can not be used in PYTHON IDE 
import matplotlib # 注意也要import一次
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(25)

# Load MNIST dataset is provided by Keras.
(train_data, train_label), (test_data, test_label) = mnist.load_data()

print(" train_data original shape", train_data.shape)
print(" train_label original shape", train_label.shape)
print(" test_data original shape", test_data.shape)
print(" test_label original shape", test_label.shape)

#Show example image and its label
plt.imshow(train_data[0], cmap='gray')
plt.title('Class '+ str(train_label[0]))

#Reshape the data and normalize the data
train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

train_data /=255
test_data /=255


# Test the shape
print(train_data.shape)

# one-hot encode the labels
number_of_classes = 10

train_onehot_label = np_utils.to_categorical(train_label, number_of_classes)
test_onehot_label = np_utils.to_categorical(test_label, number_of_classes)