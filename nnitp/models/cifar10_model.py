import numpy
from tensorflow.keras.models import load_model
from nnitp.keras import Wrapper

# Fetch the CIFAR10 dataset, normalized with mean=0, std = 1

def get_data():
    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = numpy.mean(x_train,axis=(0,1,2,3))
    std = numpy.std(x_train,axis=(0,1,2,3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)
    return (x_train, y_train), (x_test, y_test)

# Fetch the trained model

def get_model():
    model = load_model('cifar10_model.h5')
#    model.summary()
    return Wrapper(model)

params = {'size':20000,'alpha':0.95,'gamma':0.6,'mu':0.9,'layers':[6,14]}
    
