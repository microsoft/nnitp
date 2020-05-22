#
# Copyright (c) Microsoft Corporation.
#

#
# Keras backend for nnitp
#

from keras import backend as K
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from typing import Tuple
from .model_mgr import unflatten_unit

# This class is the interface to Keras models.

class Wrapper(object):

    # Constructor from a Keras model.

    def __init__(self,model):
        self.model = model
        self._backend_session = K.get_session()
        
    # To use this model in given model in a thread, we have to set it
    # up as the default Keras session and also set up the tensorflow
    # default graph. This method returns a context object suitable for
    # this purpose.  To run inference using model `foo`, you have to
    # use `with session(): ...`. TODO: Really need to expose this?

    def session(self):
        K.set_session(self._backend_session)
        return self._backend_session.graph.as_default()

    # Use tensorflow to compute the activation of layer `lidx` in
    # model `model` over input data `test`. Layer index `-1` stands
    # for the input data.

    def compute_activation(self,lidx,test):
        if lidx < 0:
            return test
        inp = self.model.input
        functor = K.function([inp, K.learning_phase()], [self.model.layers[lidx].output] )
        return functor([test])[0]

    # Get the shape of a given layer's tensor, or the input shape if
    # layer is -1.

    def layer_shape(self,layer):
        return self.model.input_shape if layer == -1 else self.model.layers[layer].output_shape

    # Return  list of layer names.

    @property
    def layers(self):
        return [layer.name for layer in self.model.layers]

    # Get the slice at layer `n` (-1 for input) that is relevant to a
    # slice `slc` at layer `n1`. TODO: doesn't really belong here since it
    # is toolkit-dependent. TODO: Not sure if this gives correct result
    # for even convolutional kernel sizes in case of padding == 'same', as
    # Keras docs don't say how padding is done in this case (i.e., whether
    # larger padding is used on left/bottom or right/top).

    # TODO: change the `slc` argument to a list of python slice objects.

    def get_cone(self,n,n1,slc) -> Tuple:
        model = self.model
        while n1 > n:
            layer = model.layers[n1]
            if isinstance(layer,Conv2D):
                weights = layer.get_weights()[0]
                row_size,col_size,planes,units = weights.shape
                shp = layer.input_shape
                if layer.padding == 'same':
                    rp = -(row_size // 2)
                    cp = -(col_size // 2)
                    slc = ((max(0,slc[0][0]+rp),max(0,slc[0][1]+cp),0),
                           (min(shp[1]-1,slc[1][0]+row_size-1+cp),
                            min(shp[2]-1,slc[1][1]+col_size-1+cp),planes-1))
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
            elif layer.input_shape == layer.output_shape:
                pass
            else:
                print ("Cannot compute dependency cone for layer of type {}.".format(type(layer)))
                exit(1)
            n1 -= 1    
        return tuple(slice(x,y+1) for x,y in zip(slc[0],slc[1]))

    
