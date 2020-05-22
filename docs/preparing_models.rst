Preparing models and data
=========================

To produce explanations, nnitp needs both the model and the data from
which the model was trained. To provide this information to nnitp,
prepare a Python file with a name like `foo_model.py`, where `foo` is
the model name. This file should contain these items:

- function `get_data()` returns the training and test datasets in this form::

    (x_train, y_train), (x_test,y_test)

  where

  - `x_train` is the vector of training input samples
  - `y_train` is the vector of training sample labels
  - `x_test` is the vector of test input samples
  - `y_test` is the vector of test sample labels.

  All of these should be represented as `numpy` arrays.

  The input data should be the exact data used to train the model. For
  example, if some preprocessing steps were applied to the original
  dataset, the same preprocessing should be applied by `get_data`.

- function `get_model()` returns the model, in an appropriate wrapper

  The wrapper used depends on the model type:

  - For Keras models, use `nnitp.keras.Wrapper`. 

- dictionary `params` mapping parameter names to values.

As an example, here is a Python file `mnist_model.py` that loads a
Keras model of the MNIST digit recognition dataset from a file
`mnist_model.h5`::

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


This file should be found in the nnitp's load path, which includes the
current working directory and source subdirectory `models`.

Models found in this path can be selected in the GUI.


