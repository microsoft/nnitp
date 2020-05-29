#
# Copyright (c) Microsoft Corporation.
#

#
# Methods for compressing a network using an ensemble of interpolants
#

import sys
import numpy as np
from .model_mgr import DataModel
from .bayesitp import interpolant, params as itp_params
from .error import BadParameter,Error, filter_params
from .itp import output_category_predicate


def compress(name,**kwargs):
    data_model = DataModel()
    data_model.load(name)
    params = data_model.params.copy()
    params.update(kwargs)
    size = params.get('size',len(data_model.x_train))
    layers = params.get('layers',[])
    category = params.get('category',0)
    if not layers:
        raise BadParameter('layers')
    l1 = layers[-1]
    data_model.set_sample_size(size)
    conc = output_category_predicate(data_model,category)
    print ('conc: {}'.format(conc))
#    inputs = conc.sat(data_model._train_eval)
#    print ('len(inputs): {}'.format(len(inputs)))
#    inpidx = params.get('input',0)
#    if inpidx >= len(inputs):
#        raise BadParameter('input')
    params = filter_params(params,itp_params)

    model = data_model._train_eval
    inputs = model.data
    pred = conc.eval(data_model._train_eval)
    itps = []
    
    for idx in range(len(inputs)):
        print ('Remaining: {}'.format(np.count_nonzero(pred)))
        if pred[idx]:
            itp,stats = interpolant(data_model,l1,inputs[idx],conc,**params)
            itps.append(itp)
            pred = np.logical_and(pred,np.logical_not(itp.eval(model)))
    print ('Interpolants:')
    for itp in itps:
        print (itp)
    
                

def main():
    if len(sys.argv) <= 2:
        print ('usage: nnitp compress option=value ... model_name')
        exit(1)
    name = sys.argv[-1]
    try:            
        compress(name)
    except Error as err:
        print (err)
        exit(1)


if __name__ == '__main__':
    main()

    
