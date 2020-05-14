#
# Copyright (c) Microsoft Corporation.
#

import model_mgr
import numpy as np
import math
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from itp import is_max,log_file,write_summary,itp_pred, bound, Predicate, LayerPredicate
from itp import BoundPredicate, And
import time
import random
from typing import Tuple, List

#
# Code for computing Bayesian interpolants.
#

# Acc is a triple `(F,N,P)` where `F` is the number of false
# positives, `N` is the number of positives, and `P` is the number of
# relevant elements (i.e., `P` is true positives plus false
# negatives). Thus precision is `(N-F)/N` and recall is `(N-F)/P`.

Acc = Tuple[int,int,int]

# Stats contains information about an interpolant

class Stats(object):
    train_acc : Acc  # Accuracy over training set
    test_acc : Acc   # Accuracy over test set
    time : float     # time to compute

    def _describe_acc(self,s,acc):
        F,N,P = acc
        prec = (N - F)/N if N != 0 else None
        recall = (N - F)/P if P != 0 else None
        return ("On {} set: F = {}, N = {}, P = {}, precision={}, recall={}\n"
                .format(s,F,N,P,prec,recall))
        
    def __str__(self):
        return (self._describe_acc('training',self.train_acc) +
                self._describe_acc('test',self.test_acc) +
                'Compute time: {}\n'.format(self.time))
        
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
        res = model_mgr.compute_activation(self.model,idx,self.data)
        print("done")
        self.eval_cache[idx] = res
        print (res.shape)
        return res
    def set_pred(self,idx,p):
        self.split_cache = dict()
        self.cond = vect_eval(p,self.eval(idx))
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
        return model_mgr.compute_activation(self.model,idx,data)[0]

        
#
# Evaluate a predicate on a vector. 
#
# TODO: replace this with Predicate.map

def vect_eval(p,data):
    return np.array(list(map(p,data)))

# Returns the predicate `x[idx] <> val`, where `idx` is a tensor
# index, `val` is a numeric value and `<>` is `<` is pos is true and
# `>` if false.

def bound_val(val,pos):
    return (lambda x: x < val) if pos else (lambda x: x > val)

def not_bound_val(val,pos):
    return (lambda x: x >= val) if pos else (lambda x: x <= val)

# Computes the discrimination of all the separators of the form
# `x[idx] <> y[idx]` where `<>` is `<` is pos is true and `>` if
# false. Here, `x` is a given tensor, `idx` is any tensor index and
# `onset,offset` is are sets of tensors. The discrimination `d` is a
# pair `(t,f)` where t is the number of tensors `y` in `onset` such that
# the predicate is true, and `f` is the number of tensors `y` in `offset` such that
# the predicate is false. The return value is a list
# of pairs `(idx,d)` where `d` is the discrimination, sorted by
# descending `d`. Notice that because of the standard sorting order on pairs,
# the `t` score is more important than the `f` score. 
# The parameter `gamma` is the minimum fraction of points in the
# offset that must be correctly classified.

def discrimination(x,onset,offset,pos,gamma):
    minoff = int(math.ceil(gamma * len(offset))) - 1
    if minoff >= 0:
        offvals = offset.copy()
        offvals.sort(axis=0)
        y = (np.flip(offvals,axis=0) if pos else offvals)[minoff]
        x = np.where(bound_val(x,pos)(y),y,x)
    t = np.count_nonzero(np.apply_along_axis(not_bound_val(x,pos),1,onset),axis=0)
    f = np.count_nonzero(np.apply_along_axis(not_bound_val(x,pos),1,offset),axis=0)
#    t.flatten(), f.flatten()
#    res = [(i,v,q) for (i,v),q in zip(np.ndenumerate(x),zip(t,f))]
    res = zip(range(len(x)),x,zip(t,f))
    foo = sorted(res,key=cost)
    return foo

def cost(y):
    t,f = y[2]
    return float(t)/(t+f)

# Get the subset of a set of tensors `e` satisfying a predicate of the form
# `x[idx] <> v`. The predicate is represented by a tuple `(idx,v,pos)`
# such that `<>` is `<` if pos is true and `>` if pos is false.

def filter(e,pred):
    idx,val,pos = pred
    cond = np.array(list(map(bound(idx,val,pos),e)))
    return np.compress(cond,e,axis=0)

# Remove from a set of tensors `e` satisfying a predicate of the form
# `x[idx] <> v`. The predicate is represented by a tuple `(idx,v,pos)`
# such that `<>` is `<` if pos is true and `>` if pos is false.

def remove(e,pred):
    idx,val,pos = pred
    cond = np.array(list(map(lambda x: not bound(idx,val,pos)(x),e)))
    return np.compress(cond,e,axis=0)

# Compute a separator between a tensor `x` and a set of tensors `onset`.
# The returned predicate should be true of `x` and false of all but a
# fraction epsilon of thr elements of `e`. In addition, we should maximize the
# number of elements of `offset` for which the separator is true. The predictate is a
# conjunction of bound predicates represented as a list.

def separator(x,onset,offset,epsilon,gamma,mu):
    res = []
    orig_gamma = gamma
#    print ("separator start")
    while len(onset) > epsilon * (len(onset)+len(offset)) and gamma > 0.001:
        pda = discrimination(x,onset,offset,True,gamma)
        nda = discrimination(x,onset,offset,False,gamma)
        pos = cost(pda[0]) <= cost(nda[0])
        idx,val,d = pda[0] if pos else nda[0]
        if d[0] < len(onset):
            pred = (idx,val,pos)
            res.append(pred)
            onset = remove(onset,pred)
            offset = remove(offset,pred)
            gamma = gamma * mu
        else:
#            print ("*** no improvement ***")
            gamma = orig_gamma * gamma
#        print('  F = {}, N = {}, P = {}'.format(len(onset),len(onset)+len(offset),N))
#        print('  precision = {}, recall = {}, gamma = {}'.format(
#            float(len(offset))/(len(onset)+len(offset)),
#            float(len(offset))/N,
#            gamma))
#    print ("separator end")
    return res

#
# Same as the above, but computes the result over a random subset of
# the features of size `samp_size`. Default sample size is `sqrt(len(x))`.

use_random_subspace = False
random_subspace_size = None

def randseparator(x,onset,offset,epsilon,gamma,mu):
    if use_random_subspace:
        samp_size = random_subspace_size if random_subspace_size is not None else int(math.sqrt(len(x)))
        subset = np.array(random.sample(list(range(len(x))),samp_size))
        sx = x[subset]
        sonset = onset[:,subset]
        soffset = offset[:,subset]
        res = separator(sx,sonset,soffset,epsilon,gamma,mu)
        res = [(subset[idx],val,pos) for idx,val,pos in res]
    else:
        res = separator(x,onset,offset,epsilon,gamma,mu)
    return res


# Same as above, but creates an ensemble of predictors of size `ensemble_size`

ensemble_size = 1

def ensembleseparator(x,onset,offset,epsilon,gamma,mu):
    res = []
    for idx in range(ensemble_size):
        res.extend(randseparator(x,onset,offset,epsilon,gamma,mu))
    return res

#
# Same as above, but allows the values to be N-dimentional. Unlike the above
# functions, the coordinates in the result are tuples, even if the input is
# one-dimensional. 
#

def ndseparator(x,onset,offset,epsilon,gamma,mu) -> List[Tuple[Tuple,float,bool]]:
    shape = x.shape
    if len(shape) > 1:
        x = np.ravel(x)
        onset = onset.reshape(len(onset),-1)
        offset = offset.reshape(len(offset),-1)
    res = ensembleseparator(x,onset,offset,epsilon,gamma,mu)
    if len(shape) > 1:
        res = [(unflatten_unit(shape,idx),val,pos) for idx,val,pos in res]
    else:
        res = [((idx,),val,pos) for idx,val,pos in res]
    return res
    
#
# Same as above, but separator is expressed only over a slice `cone`
# of the input tensors. If `cone` is None, no slicing is done.
#

def slice_ndseparator(ndx,ndonset,ndoffset,epsilon,gamma,mu,cone=None):
    print ('cone = {}'.format(cone))
    if cone is not None:
        ndoffset = ndoffset[(slice(None),)+cone]
        ndonset = ndonset[(slice(None),)+cone]
        ndx = ndx[cone]
    res = ndseparator(ndx,ndonset,ndoffset,epsilon,gamma,mu)
    if cone is not None:
        res = unslice_itp(cone,res)
    return res

# Given model evaluators for the training and test sets, compute an
# interpolant at layer `l1` for a predicate at layer `l2` with
# precision `1-epsilon`, recall `gamma`. The interpolant must be
# expressed over slice `cone` of layer `l1`, if `cone` is not None.
# If `samps` is not None, it is take as the training sample split,
# i.e., the pair `(pos,neg)` where `pos` is the set of positve
# training samples at layer `l1`, and `neg` is the set of negative
# training samples. Otherwise the training split is computed.  Returns
# the interpolant, the training error and the test error.

show_predictions = False

def interpolant_int(train_eval,test_eval,l1,x,l2,pred,
                    epsilon,gamma,mu,cone=None,samps=None):
    global ttime
    print ('l2 = {}'.format(l2))
    train_eval.set_pred(l2,pred)
    before = time.time()
    psamps2,nsamps2 = train_eval.split(l1) if samps is None else samps
    res = slice_ndseparator(x,nsamps2,psamps2,epsilon,gamma,mu,cone)
    ttime += (time.time()-before)
    print ("interpolant: {}".format(res))
#    check_itp_pred(res,x)
    train_error = check_itp(train_eval,l1,itp_pred(res),l2,pred)
    describe_error("training",train_error)
    test_error = check_itp(test_eval,l1,itp_pred(res),l2,pred)
    describe_error("test",test_error)
    if show_predictions:
        show_positives(train_eval,l1,itp_pred(res),l2,pred,10,res)
    return res,train_error,test_error

ttime = 0.0

def interpolant(train_eval:ModelEval,test_eval:ModelEval,l1:int,inp:np.ndarray,
                lpred:LayerPredicate,
                epsilon:float=0.05,gamma:float=0.6,mu:float=0.9) -> Tuple[LayerPredicate,Stats]:

    global ttime
    ttime = 0.0
    l2,pred = lpred.layer,lpred.pred
    x = train_eval.eval_one(l1,inp)
    cone = get_pred_cone(train_eval.model,lpred,l1)
    res,train_error,test_error = interpolant_int(train_eval,test_eval,l1,x,l2,pred,
                                                 epsilon,gamma,mu,cone=cone)
    conjs = [BoundPredicate(*x) for x in res]
    stats = Stats()
    stats.train_acc = train_error
    stats.test_acc = test_error
    stats. time = ttime 
    return LayerPredicate(l1,And(*conjs)),stats 

def describe_error(s,error):
    F,N,P = error
    print ("on {} set: F = {}, N = {}, P = {}, precision={}, recall={}"
           .format(s,F,N,P,1.0 - float(F)/(N+1),float((N+1)-F)/P))

def unflatten_unit(input_shape,unit):
    unit = unit[0] if isinstance(unit,tuple) else unit
    res = tuple()
    while len(input_shape) > 0:
        input_shape,dim = input_shape[:-1],input_shape[-1]
        res = (unit%dim,) + res
        unit = unit//dim
    return res

# Get the slice at layer `n` (-1 for input) that is relevant to a
# slice `slc` at layer `n1`. TODO: doesn't really belong here since it
# is toolkit-dependent. TODO: Not sure if this gives correct result
# for even convolutional kernel sizes in case of padding == 'same', as
# Keras docs don't say how padding is done in this case (i.e., whether
# larger padding is used on left/bottom or right/top).

def get_cone(model,n,n1,slc) -> Tuple:
    while n1 > n:
        layer = model.layers[n1]
        if isinstance(layer,Conv2D):
            weights = layer.get_weights()[0]
            row_size,col_size,planes,units = weights.shape
            print ('layer = {}, row_size = {}, col_size = {}'.format(n1,row_size,col_size))
            shp = layer.input_shape
            if layer.padding == 'same':
                rp = -(row_size // 2)
                cp = -(col_size // 2)
                slc = ((max(0,slc[0][0]+rp),max(0,slc[0][1]+cp),0),
                       (min(shp[1]-1,slc[1][0]+row_size-1+cp),
                        min(shp[2]-1,slc[1][1]+col_size-1+cp),planes-1))
                print ("conv slice = {}".format(slc))
            else:
                slc = ((slc[0][0],slc[0][1],0),
                       (slc[1][0]+row_size-1,slc[1][1]+col_size-1,planes-1))
        elif isinstance(layer,MaxPooling2D):
            wrows,wcols = layer.pool_size
            def foo(i):
                return (slc[i][0] * wrows,slc[i][1] * wcols,slc[i][2])
            slc = ((slc[0][0] * wrows,slc[0][1] * wcols,slc[0][2]),
                   ((slc[1][0]+1)*wrows-1,(slc[1][1]+1)*wcols-1,slc[1][2]))
        elif isinstance(layer,Flatten):
            shape = layer.input_shape[1:]
            slc = (unflatten_unit(shape,slc[0]),unflatten_unit(shape,slc[1]))
        elif isinstance(layer,Dense):
            shape = layer.input_shape[1:]
            slc = (tuple(0 for x in shape),tuple(x-1 for x in shape))
            print ('Dense slc = {}'.format(slc))
        elif layer.input_shape == layer.output_shape:
            pass
        else:
            print ("Cannot compute kernel image for layer of type {}.".format(type(layer)))
            exit(1)
        n1 -= 1    
    return tuple(slice(x,y+1) for x,y in zip(slc[0],slc[1]))
    
def get_pred_cone(model,lpred,layer = -1) -> Tuple:
    shape = model.input_shape if lpred.layer == -1 else model.layers[lpred.layer].output_shape
    cone = lpred.pred.cone(shape[1:])
    print ('cone= {}'.format(cone))
    slc = (tuple(x.start for x in cone),tuple(x.stop-1 for x in cone))
    print ('slc= {}'.format(slc))
    return get_cone(model,layer,lpred.layer,slc)

# Given a predicate `pred` over a slice `slc`, return the
# corresponding predicate over the whole tensor.

def unslice_itp(slc,pred):
    return [(unslice_coord(slc,idx),val,pos) for idx,val,pos in pred]

def unslice_coord(slc,coord):
    return tuple(x.start+y for x,y in zip(slc,coord))

# Predicate saying output o1 is greater than output o2. Note, o1 and
# o2 may be tuples, refering to elements of tenors of dimension greater
# than 1.

def prefer_output(o1,o2):
    return lambda x: x[o1] >= x[o2]

    
# Computes the error for a predication over a data set. The error is
# represented as a triple `(F,N,P)`, where `F` is the number of
# failed predications, `N` is the number predictions, and `P` is the
# number of positive results in the dataset. Thus, the precision is
# `(1-F)/N` and the recall is `N/P`.
#
# The procedure takes a model evaluator `m`, the prediction predicate
# `p1` at layer `l1` and the result predicate `p2` at layer `l2`.
# 

def check_itp(m,l1,p1,l2,p2):
    prediction = vect_eval(p1,m.eval(l1))
    print ('checkitp: l2 = {}'.format(l2))
    result = vect_eval(p2,m.eval(l2))
    failure = np.logical_and(prediction,np.logical_not(result))
    F = np.count_nonzero(failure)
    N = np.count_nonzero(prediction)
    P = np.count_nonzero(result)
    return (F,N,P)

