#
# Copyright (c) Microsoft Corporation.
#

import sys
import os
import numpy as np
from importlib import import_module
from .error import Error

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

# TODO: maybe this belongs in a 'utilities' file.

def unflatten_unit(input_shape,unit):
    unit = unit[0] if isinstance(unit,tuple) else unit
    res = tuple()
    while len(input_shape) > 0:
        input_shape,dim = input_shape[:-1],input_shape[-1]
        res = (unit%dim,) + res
        unit = unit//dim
    return res

# This code scans the `models` directory and reads all of the modules
# into a dictionary `datasets`.

datasets = {}
suffix = '_model.py'
model_path = [os.path.join(os.path.dirname(__file__),'models'),'.']
orig_sys_path = sys.path
sys.path.extend(model_path)
for dir in model_path:
    for fname in os.listdir(dir):
        if fname.endswith(suffix):
            modname = fname[0:-3]
            module = import_module(modname)
            name = fname[0:-len(suffix)]
            datasets[name] = module
sys.path = orig_sys_path
        
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
            model_dir = os.path.dirname(module.__file__)
            os.chdir(model_dir)
            self.model = module.get_model()
            (self.x_train, self.y_train), (self.x_test, self.y_test), self.datatype = module.get_data()
            self.params = module.params if hasattr(module,'params') else {}
            os.chdir(cwd)
            self.loaded = True

    # TRICKY: to use the model in a thread other than the one in which
    # it was created, we have to set it up as the default
    # session. This method returns a context object suitable for this
    # purpose.  If you want to run inference using a DataModel `foo`,
    # you have to use `with foo.session(): ...`.

    def session(self):
        return self.model.session()
    
    def set_sample_size(self,size:int):
        self._train_eval = ModelEval(self.model,self.x_train[:size])
        self._test_eval = ModelEval(self.model,self.x_test[:size])

    def output_layer(self) -> int:
        return len(self.model.layers) - 1
        
# Computes the activation of layer `lidx` in model `model` over input
# data `test`. Layer index `-1` stands for the input data.
#

def compute_activation(model,lidx,test):
    return model.compute_activation(lidx,test)

# Given a 'flat' index into a tensor, return the element index.
# Here, `input_shape` is the shape of the tensor, and `unit` is the
# index to an element of the tensor flattened into a vector.

    
# Object for evaluating a model on an input set and caching the
# results. The constructor takes a model and some input data.  The
# `eval` method returns the activation value of layer `idx`. The method
# `set_pred` records a predicate `p` over layer `idx`. The method
# `split` returns a pair consisting of the the activations at layer `idx`
# when the predicate is true/false. Method `indices` returns a vector
# of the indices satisfying `p`.

class ModelEval(object):
    def __init__(self,model,data):
        self.model,self.data = model,data
        self.eval_cache = dict()
    def eval(self,idx):
        if idx in self.eval_cache:
            return self.eval_cache[idx]
        print("evaluating layer {}".format(idx))
        # Evaluate in batches of 10000 to avoid memout
        res = np.concatenate([compute_activation(self.model,idx,self.data[base:base+10000])
                              for base in range(0,len(self.data),10000)])
        print("done")
        self.eval_cache[idx] = res
        return res
    def set_pred(self,idx,p):
        self.split_cache = dict()
        self.cond = vect_eval(p,self.eval(idx))
    def set_layer_pred(self,lp):
        self.split_cache = dict()
        self.cond = lp.eval(self)
    def split(self,idx):
        if idx in self.split_cache:
            return self.split_cache[idx]
        def select(c):
            return np.compress(c,self.eval(idx),axis=0)
        res = (select(self.cond),select(np.logical_not(self.cond)))
        self.split_cache[idx] = res
        return res
    def indices(self):
        return np.compress(self.cond,np.arange(len(self.cond)))
    def eval_one(self,idx,input):
        data = input.reshape(1,*input.shape)
        return compute_activation(self.model,idx,data)[0]
    def eval_all(self,idx,data):
        return compute_activation(self.model,idx,data)

#
# Evaluate a predicate on a vector. 
#
# TODO: replace this with Predicate.map

def vect_eval(p,data):
    return np.array(list(map(p,data)))
        
