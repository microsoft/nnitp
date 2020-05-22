import tensorflow as tf
from keras.models import load_model
from nnitp.keras import Wrapper

# Fetch the MNIST dataset, normalize to range [0.0,1.0)

def get_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train), (x_test, y_test)

# Fetch the trained model

def get_model():
    return Wrapper(load_model('mnist_model.h5'))

params = {'size':100,'alpha':0.98,'gamma':0.6,'mu':0.9,'layers':[2]}
    
