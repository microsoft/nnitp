#
# Copyright (c) Microsoft Corporation.
#

from keras import backend as K
import os
    
# Code for fetching models and datasets.
#
# TODO: this is dependent on Keras framework.
#
# The models and datasets are defined by files in sub-directory
# `models` with names of the form `<name>_model.py` where `<name>` is
# the name to associated with the model/dataset. Each file is a python module
# containing two functions:
#
# - `get_data()` returns the datasets in the form `(x_train,y_train),(x_test,y_test)`
# - `get_model()` returns the trained model (usually stored in a file in the `models` directory)
#
# These functions are executed with `models` as working directory.
#

# This code scans the `models` directory and reads all of the modules
# into a dictionary `datasets`.

datasets = {}
suffix = '_model.py'
model_dir = os.path.join(os.path.dirname(__file__),'models')
for fname in os.listdir(model_dir):
    if fname.endswith(suffix):
        modname = fname[0:-3]
        module = __import__('models.'+modname)
        name = fname[0:-len(suffix)]
        datasets[name] = module.__dict__[modname]
        
# Class `DataModel` is a combination of a dataset (training and test)
# and a trained model. 

class DataModel(object):

    # Intial, the DataModel is unloaded, unless a name is given.

    def __init__(self, name = None):
        self.loaded = False
        self.load(name)
    
    # Load a `DataModel` by name. 

    def load(self,name):
        self.name = name
        if name is not None:
            module = datasets[name]
            cwd = os.getcwd()
            os.chdir(model_dir)
            print (module.__dict__)
            self.model = module.get_model()
            (self.x_train, self.y_train), (self.x_test, self.y_test) = module.get_data()
            os.chdir(cwd)
            self._backend_session = K.get_session()
            self.loaded = True

    # TRICKY: to use tensorflow on a given model in a thread, we have
    # to set it up as the default session and also set up the tensorflow
    # default graph. This method returns a context object suitable for this purpose.
    # If you want to run inference using a DataModel `foo`, you have to
    # use `with foo.session(): ...`.  

    def session(self):
        K.set_session(self._backend_session)
        return self._backend_session.graph.as_default()
    
# Computes the activation of layer `lidx` in model `model` over input
# data `test`. Layer index `-1` stands for the input data.
#

def compute_activation(model,lidx,test):
    if lidx < 0:
        return test
    inp = model.input
    functor = K.function([inp, K.learning_phase()], [model.layers[lidx].output] )
    return functor([test])[0]
