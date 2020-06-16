#
# Copyright (c) Microsoft Corporation.
#

from .model_mgr import unflatten_unit, ModelEval, DataModel
import numpy as np
import math
from .itp import itp_pred, bound, LayerPredicate
from .itp import BoundPredicate, And
import time
import random
from typing import Tuple, List, Optional, Callable

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

# Find the upper gamma-th fractile of all units, over a vector of
# activations. If pos is true, find the lower fractile.

def fractile_all(gamma,pos,offset):
    assert len(offset) > 0 and gamma > 0.0
    minoff = int(math.ceil(gamma * len(offset))) - 1
    offvals = offset.copy()
    offvals.sort(axis=0)
    return (np.flip(offvals,axis=0) if pos else offvals)[minoff]


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

def discrimination(A,onset,offset,pos,gamma):
    y = fractile_all(gamma,pos,offset)
    x = fractile_all(gamma,pos,A)
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

def separator(A,onset,offset,epsilon,gamma,mu):
    res = []
    orig_gamma = gamma
    while len(onset) > epsilon * (len(onset)+len(offset)) and gamma > 0.001:
        pda = discrimination(A,onset,offset,True,gamma)
        nda = discrimination(A,onset,offset,False,gamma)
        pos = cost(pda[0]) <= cost(nda[0])
        idx,val,d = pda[0] if pos else nda[0]
        if d[0] < len(onset):
            pred = (idx,val,pos)
            res.append(pred)
            A = remove(A,pred)
            onset = remove(onset,pred)
            offset = remove(offset,pred)
            gamma = gamma * mu
        else:
#            print ("*** no improvement ***")
            gamma = orig_gamma * gamma
    return res

#
# Same as the above, but computes the result over a random subset of
# the features of size `samp_size`. Default sample size is `sqrt(len(x))`.

_use_random_subspace = False
_random_subspace_size = None

def randseparator(A,onset,offset,epsilon,gamma,mu):
    if _use_random_subspace:
        N = A.shape[1]
        samp_size = (_random_subspace_size(N) if _random_subspace_size is not None
                     else int(math.sqrt(N)))
        subset = np.array(random.sample(list(range(N)),samp_size))
        sA = A[:,subset]
        sonset = onset[:,subset]
        soffset = offset[:,subset]
        res = separator(sA,sonset,soffset,epsilon,gamma,mu)
        res = [(subset[idx],val,pos) for idx,val,pos in res]
    else:
        res = separator(A,onset,offset,epsilon,gamma,mu)
    return res


# Same as above, but creates an ensemble of predictors of size `_ensemble_size`

_ensemble_size = 1

def ensembleseparator(A,onset,offset,epsilon,gamma,mu):
    res = []
    for idx in range(_ensemble_size):
        res.extend(randseparator(A,onset,offset,epsilon,gamma,mu))
    return res

#
# Same as above, but allows the values to be N-dimentional. Unlike the above
# functions, the coordinates in the result are tuples, even if the input is
# one-dimensional. 
#

def ndseparator(A,onset,offset,epsilon,gamma,mu) -> List[Tuple[Tuple,float,bool]]:
    shape = A.shape[1:]
    if len(shape) > 1:
#        x = np.ravel(x)
        A = A.reshape(len(A),-1)
        onset = onset.reshape(len(onset),-1)
        offset = offset.reshape(len(offset),-1)
    res = ensembleseparator(A,onset,offset,epsilon,gamma,mu)
    if len(shape) > 1:
        res = [(unflatten_unit(shape,idx),val,pos) for idx,val,pos in res]
    else:
        res = [((idx,),val,pos) for idx,val,pos in res]
    return res # type: ignore
    
#
# Same as above, but separator is expressed only over a slice `cone`
# of the input tensors. If `cone` is None, no slicing is done.
#

def slice_ndseparator(ndA,ndonset,ndoffset,epsilon,gamma,mu,cone=None):
    if cone is not None:
        ndoffset = ndoffset[(slice(None),)+cone]
        ndonset = ndonset[(slice(None),)+cone]
        ndA = ndA[(slice(None),)+cone]
    res = ndseparator(ndA,ndonset,ndoffset,epsilon,gamma,mu)
    if cone is not None:
        res = unslice_itp(cone,res)
    return res

# Internal implementation of `interpolant()`, see below. If we are given weights,
# we keep all of the positive samples with non-zero weights. TODO: we should pass the weights
# down and use them to compute the fractiles and precisions. 

def interpolant_int(train_eval,test_eval,l1,A,l2,pred,
                    epsilon,gamma,mu,cone=None,samps=None,weights=None):
    global _ttime,_weights
    train_eval.set_pred(l2,pred)
    before = time.time()
    psamps2,nsamps2 = train_eval.split(l1) if samps is None else samps
    _weights = weights
    if _weights is not None:
        _weights =  np.compress(train_eval.cond,_weights,axis=0)
        assert len(_weights) == len(psamps2)
        psamps2 = np.compress(_weights,psamps2,axis=0)
    res = slice_ndseparator(A,nsamps2,psamps2,epsilon,gamma,mu,cone)
    _ttime += (time.time()-before)
    print ("interpolant: {}".format(res))
    train_error = check_itp(train_eval,l1,itp_pred(res),l2,pred)
    describe_error("training",train_error)
    test_error = check_itp(test_eval,l1,itp_pred(res),l2,pred)
    describe_error("test",test_error)
#    if show_predictions:
#        show_positives(train_eval,l1,itp_pred(res),l2,pred,10,res)
    return res,train_error,test_error

_ttime = 0.0

# Compute a Bayesian interpolant.
#
# Required parameters:
#
# - `data_model` : DataModel object providing model with training and test data
# - `l1`: index of layer at which interpolant should be computed
# - `inps`: input valuations (i.e., the inputs satisfying `A`)
# - `lpred` : LayerPredicate representing the conclusion `B`
#
# Options:
#
# - `alpha`: precision parameter (default: 0.98)
# - `gamma`: recall parameter (default: 0.6)
# - `mu`: recall reduction parameter (default: 0.9)
# - `ensemble size`: size of ensemble (default: 1)
# - `random_subspace_size`: function from feature space size to random subspace size
#
# Return value: (itp,stats), where
#
# - `itp` is a LayerPredicate representing the interpolant
# - `stats` is a Stats object giving interpolation statistics
#
# Notes:
#
# - If `ensemble_size > 1`, the interpolant is computed as an ensemble
#   (a conjunction) of independent interpolants. 
# - If `random_subspace_size` is not `None`, then each interpolant
#   in the ensemble is computed using a random sample without
#   replacement of the features of size `random_subspace_size(N)`
#   where `N` is the number of units in the interpolant layer `l1`.
#   The default function is `N/ensemble_size`.
#   

def interpolant(data_model:DataModel,l1:int,inps:np.ndarray,
                lpred:LayerPredicate,alpha:float=0.98,gamma:float=0.6,
                mu:float=0.9,ensemble_size:int=1,
                random_subspace_size:Optional[Callable[[int],int]]=None,
                weights = None,
                ) -> Tuple[LayerPredicate,Stats]:

    print ('inps.shape = {}'.format(inps.shape))
    global _ttime,_ensemble_size,_use_random_subspace,_random_subspace_size
    train_eval,test_eval = data_model._train_eval,data_model._test_eval
    epsilon = 1.0 - alpha
    _ttime = 0.0
    _ensemble_size = ensemble_size
    _use_random_subspace = ensemble_size > 1
    _random_subspace_size = random_subspace_size or (lambda N: N//ensemble_size)
    l2,pred = lpred.layer,lpred.pred
    A = train_eval.eval_all(l1,inps)
    cone = get_pred_cone(train_eval.model,lpred,l1)
    res,train_error,test_error = interpolant_int(train_eval,test_eval,l1,A,l2,pred,
                                                 epsilon,gamma,mu,cone=cone,weights=weights)
    conjs = [BoundPredicate(*x) for x in res]
#    for inp in inps:
#        if not And(*conjs)(train_eval.eval_one(l1,inp)):
#            print ("interpolant not satisfied for input")
    stats = Stats()
    stats.train_acc = train_error
    stats.test_acc = test_error
    stats. time = _ttime 
    return LayerPredicate(l1,And(*conjs)),stats 

def describe_error(s,error):
    F,N,P = error
    print ("on {} set: F = {}, N = {}, P = {}, precision={}, recall={}"
           .format(s,F,N,P,1.0 - float(F)/(N+1),float((N+1)-F)/P))


# Get the slice at layer `n` (-1 for input) that is relevant to a
# slice `slc` at layer `n1`.

def get_cone(model,n,n1,slc) -> Tuple:
    return model.get_cone(n,n1,slc) # type: ignore
    
# Get the slice of layer `layer` that is relevant to LayerPredicate `lpred`.

def get_pred_cone(model,lpred,layer = -1) -> Tuple:
    shape = model.layer_shape(lpred.layer)
    cone = lpred.pred.cone(shape[1:])
    slc = (tuple(x.start for x in cone),tuple(x.stop-1 for x in cone))
    return get_cone(model,layer,lpred.layer,slc)

# Given a predicate `pred` over a slice `slc`, return the
# corresponding predicate over the whole tensor.

def unslice_itp(slc,pred):
    return [(unslice_coord(slc,idx),val,pos) for idx,val,pos in pred]

def unslice_coord(slc,coord):
    return tuple(x.start+y for x,y in zip(slc,coord))

    
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
    print ('p1 = {}, l1 = {}'.format(p1,l1))
    prediction = vect_eval(p1,m.eval(l1))
    result = vect_eval(p2,m.eval(l2))
    failure = np.logical_and(prediction,np.logical_not(result))
    F = np.count_nonzero(failure)
    N = np.count_nonzero(prediction)
    P = np.count_nonzero(result)
    return (F,N,P)


# Computes the gamma-th fractile of a unit over a dataset:
#
# - model: The model and datatset
# - lidx: the layer index (-1 for input)
# - unit: the tensor index
# - pos: true for upper fractile, false for lower
# - gamma: the fraction
#

def fractile(model:ModelEval,lidx:int,unit:Tuple[int,...],pos:bool,gamma:float):
    data = model.eval(lidx)
    data = data[(slice(None),)+unit]
    data = np.sort(data)
    if not pos:
        data = np.flip(data)
    off = int(math.ceil(gamma * len(data))) - 1 if gamma > 0.0 else 0
    return LayerPredicate(lidx,BoundPredicate(unit,data[off],pos))

# Computes the fraction of samples satisfying a predicate:
#
# - model: The model and datatset
# - lpred: The predicate
#

def fraction(model:ModelEval,lpred:LayerPredicate):
    data = lpred.eval(model)
    return np.count_nonzero(data) / len(data)
    
params = {
    'alpha':float,
    'gamma':float,
    'mu':float,
    'ensemble_size':int
}
    
