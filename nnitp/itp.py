#
# Copyright (c) Microsoft Corporation.
#

#
# Data structures related to interpolants
#

import numpy as np # type: ignore
import os
from typing import Tuple,List
from abc import ABCMeta, abstractmethod


# Evaluate interpolant predicate. This is a conjunction of predicates of the
# form `x[idx] <> v`. Each predicate is represented by a tuple
# `(idx,v,pos)` such that `<>` is `<=` if pos is true and `<=` if pos is
# false.

def itp_eval(conj,val):
    for atom in conj:
        if bound(*atom)(val):
            return False
    return True

def itp_pred(conj):
    return lambda val: itp_eval(conj,val)

# Form the conjunction of interpolant predicates.

def itp_conj(*args):
    return sum(args,[]) # Just concatenate the lists of conjuncts

# Returns the predicate `x[idx] <> val`, where `idx` is a tensor
# index, `val` is a numeric value and `<>` is `<` is pos is true and
# `>` if false.

def bound(idx,val,pos):
    return (lambda x: x[idx] < val) if pos else (lambda x: x[idx] > val)

# Weaken a predicate slightly to account for rounding error

def relax_itp_pred(pred):
    def weaken(bp):
        idx,v,pos = bp
        return (idx,v-0.001,pos) if pos else (idx,v+0.001,pos)
    return map(weaken,pred)

# Class of predicates. A predicate can be applied to a valuation to
# yield a truth value, or mapped over a vector of valuations to yield
# a vector of truth values.

class Predicate(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self,data:np.ndarray) -> bool: pass
    @abstractmethod
    def map(self,data:np.ndarray) -> np.ndarray: pass
    def conjs(self) -> List['Predicate']:
        return [self]
    @abstractmethod
    def cone(self,shape:Tuple[int,...]): pass
    @abstractmethod
    def __eq__(self,other) -> bool: pass
    
# A bound predicate states that `var >= val`, where `var` is a
# variable `val` is a numeric value, when `pos` is true. When `pos` is
# false, it represents `var <= val`. Here, `var` is an index into a
# tensor of variables.


class BoundPredicate(Predicate):
    var : Tuple[int,...]
    val : float
    pos : bool
    def __eq__(self,other) -> bool:
        return (type(other) == type(self) and
                other.var == self.var and
                other.val == self.val and
                other.pos == self.pos)
    def __call__(self,data:np.ndarray):
        return data[self.var] >= self.val if self.pos else data[self.var] <= self.val
    def map(self,data):
        vals = data[(slice(None),)+self.var]
        return vals >= self.val if self.pos else vals <= self.val
    def __init__(self,var:Tuple[int,...],val:float,pos:bool):
        self.var,self.val,self.pos = var,val,pos
    def __str__(self):
        return (self.var_name() +
                ('>=' if self.pos else '<=') + str(self.val))
    def cone(self,shape:Tuple[int,...]):
        idx:Tuple = self.var if isinstance(self.var,tuple) else tuple([self.var])
        return tuple(slice(n,n+1) for n in idx)
    def var_val(self,data:np.ndarray):
        return data[self.var]
    def var_name(self):
        idx = self.var if isinstance(self.var,tuple) else (self.var,)
        return 'v({})'.format(','.join(str(x) for x in idx))
        
    
def cone_join(slices):
    print (slices)
    lbs = tuple(tuple(x.start for x in y) for y in slices)
    print ('lbs = {}'.format(lbs))
    lbs = tuple(zip(*lbs))
    lb = tuple(min(x) for x in lbs)
    ubs = zip(*tuple(tuple(x.stop for x in y) for y in slices))
    ub = tuple(max(x) for x in ubs)
    print ('lbs,lb,ubs,ub = {},{},{},{}'.format(lbs,lb,ubs,ub))
    return tuple(slice(x,y) for x,y in zip(lb,ub))

# Conjunction of predicates.

class And(Predicate):
    args : List[Predicate]
    def __eq__(self,other) -> bool:
        return (type(other) == type(self) and
                all(x==y for x,y in zip(self.args,other.args)))
    def __call__(self,data:np.ndarray):
        for pred in self.args:
            if not pred(data):
                return False
        return True
    def map(self,data):
        res = np.ones(len(data), dtype=bool)
        for pred in self.args:
            res = np.logical_and(res,pred.map(data))
        return res
    def __init__(self,*args : Predicate):
        self.args = list(args)
    def __str__(self):
        return ('(' + ' & '.join(str(x) for x in self.args) + ')') if self.args else 'true'
    def conjs(self):
        return self.args
    def cone(self,shape:Tuple[int,...]):
        res = cone_join(list(x.cone(shape) for x in self.args))
        print ('cone res = {}'.format(res))
        return res
    
# Negation of predicate.

class Not(Predicate):
    arg : Predicate
    def __eq__(self,other) -> bool:
        return type(other) == type(self) and self.arg == other.arg
    def __call__(self,data:np.ndarray):
        return not self.arg(data)
    def map(self,data):
        return np.logical_not(self.arg.map(data))
    def __init__(self,arg : Predicate):
        self.arg = arg
    def __str__(self):
        return '~' + str(self.arg)
    def conjs(self):
        return [self]
    def cone(self,shape:Tuple[int,...]):
        return self.arg.cone(shape)

# Predicate stating that variable `var` is over all variables. Here,
# `var` is an index into a tensor of variables.

class is_max(Predicate):
    def __init__(self,unit):
        self.unit = unit
    def __eq__(self,other) -> bool:
        return type(other) == type(self) and self.unit == other.unit
    def __call__(self,x):
        if self.unit < 0 or self.unit >= len(x):
            return False
        return np.all(x[self.unit] >= x) 
    def map(self,data):
        return np.array([self(y) for y in data])
    def __repr__(self):
        return 'is_max({})'.format(self.unit)
    def cone(self,shape:Tuple[int,...]):
        res = tuple(slice(0,x) for x in  shape)
        return res
            
class LayerPredicate(object):
    layer : int
    pred : Predicate
    def __eq__(self,other) -> bool:
        return (type(other) == type(self) and
                self.layer == other.layer and
                self.pred == other.pred)
    def __init__(self,layer:int,pred:Predicate):
        self.layer,self.pred = layer,pred
    def __str__(self):
        return 'layer {}: {}'.format(self.layer,self.pred)
    def conjs(self):
        return [LayerPredicate(self.layer,p) for p in self.pred.conjs()]
    def cone(self,shape:Tuple[int,...]):
        return self.pred.cone(shape)
    def negate(self):
        return LayerPredicate(self.layer,Not(self.pred))
    def eval(self,model):
        return self.pred.map(model.eval(self.layer))
    def eval_one(self,model,x):
        return self.pred(model.eval_one(self.layer,x))
    def eval_all(self,model,data):
        return self.pred.map(model.eval_all(self.layer.data))
    def sat(self,model):
        model.set_pred(self.layer,self.pred)
        res,_ = model.split(-1)
        return res

class AndLayerPredicate(object):
    args : Tuple[LayerPredicate,...]
    def __eq__(self,other) -> bool:
        return (type(other) == type(self) and
                all(x==y for x,y in zip(self.args,other.args)))
    def __init__(self,*args: LayerPredicate):
        self.args = args
    def eval(self,model):
        res = np.ones(len(model.data), dtype=bool)
        for lpred in self.args:
            res = np.logical_and(res,lpred.eval(model))
        return res
    
def output_category_predicate(data_model,category) -> LayerPredicate:
    return LayerPredicate(data_model.output_layer(),is_max(category))

def output_range_predicate(data_model,lower:float,upper:float) -> LayerPredicate:
    return LayerPredicate(data_model.output_layer(),
                          And(BoundPredicate((0,),lower,True),
                              BoundPredicate((0,),upper,False)))
    
# Interpolation log files
#
# The log format:
#
# The first line is a dictionary containing parameter values in python format.
#
# Each subsequent line is a triple `(idx,out_pred,inp_itp,l1_itp)` where:
# - `idx` is the index of the input,
# - `out_pred` is the output predicate (the conclusion of the interpolant)
# - `inp_itp` is a list of interpolants at the input layer,
# - `l1_itp` is the interpolant at layer `l1`, where `l1` is a parameter
#
# Each interpolant is of the form `(pred,train_error,test_error)`
# where `pred` is a predicate, and `train_error`, `test_error` are,
# respectively, the training and test error of the interpolant. The
# errors are represented as triples, `(F,N,P)` where `F` is the number
# of false positive predications, `N` is the number of predications of
# the interpolant, `P` is the number of predictions of the model.
#

# Open a log file. Appends a number to the name to make it unique.

def log_file(name):
    idx = 0
    while True:
        fname = name + '_' + str(idx) + '.log'
        if not os.path.exists(fname):
            print ("log file name: {}".format(fname))
            return open(fname,"w")
        idx += 1

def write_summary(log,name,outputs,kwargs):
    params = kwargs.copy()
    params.update({"name":name,"outputs":outputs,"version":1})
    log.write(str(params) + '\n')
    return params

# read a log file with one entry per line, in Python format. If `output`
# is given returns only results for predicate `is_max(output)`.

def read_log(fname,output=None,header_only=False):
    log = []
    with open(fname) as f:
        for line in f:
            try:
                log.append(eval(line))
                if header_only:
                    break
            except:
                print("{}: cannot parse log line {}:".format(fname,len(log)+1))
                print (line)
                exit(1)
    if len(log) == 0:
        print("{}: log is empty".format(fname))
        exit(1)
        
    params = log[0]
    results = log[1:]
    if log[0]['version'] == 0:  # version 0 did not record input index and conclusion
        results = [(None,None)+res for res in results]
    elif output is not None:
        results = [res for res in results if isinstance(res[1],is_max) and res[1].unit == output]
    return params,results

def write_log(summary,results):
    name = summary['name']
    with log_file(name) as log:
        write_summary(log,name,summary['outputs'],summary)
        for res in results:
            log.write(repr(res) + '\n')


