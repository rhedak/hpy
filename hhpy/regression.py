"""
hhpy.regression.py
~~~~~~~~~~~~~~~~~~

Contains regression models. Mostly convenience wrappers of other frameworks.

"""

# ---- imports
# --- standard imports
import os
import sys
import numpy as np

# --- third party imports

# --- optional imports
try:
    # suppress 'Using TensorFlow backend.' message
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    import keras
    sys.stderr = stderr
except ImportError:
    keras = None

# --- local imports
from hhpy.main import BaseClass, export, SequenceOrScalar
from hhpy.modelling import to_keras_3d


# ---- classes
# noinspection PyPep8Naming
@export
class Conv1DNN(BaseClass):

    # --- attributes
    __name__ = 'Conv1DNN'
    __attributes__ = ['window', 'epochs', 'batch_size', 'filters', 'kernel_size', 'dropout', 'pool_size', 'optimizer',
                      'epochs_trained', 'input_size', 'output_size']

    # --- functions
    def __init__(self, window: int = 10, epochs: int = 50, batch_size: int = 32, filters: int = 32,
                 kernel_size: int = 5, dropout: float = .5, pool_size: int = 2, optimizer: str = 'adam',
                 activation: str = 'relu', loss: str = 'mse'):

        # -- assert

        if keras is None:
            raise ImportError('Missing dependency keras')

        # -- assign
        # set attributes
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.pool_size = pool_size
        self.optimizer = optimizer
        self.activation = activation
        self.loss = loss

        # not-set attributes
        self.model = None
        self.epochs_trained = 0
        self.input_size = None
        self.output_size = None

    def _init_model(self, X: np.ndarray, y: np.ndarray):

        self.input_size = X.shape[2]
        self.output_size = y.shape[1]

        # -- init model
        _model = keras.Sequential()
        # -- add double conv 1d layer
        _model.add(keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation,
                                       input_shape=(self.window, self.input_size)))
        _model.add(keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation=self.activation))
        # -- add dropout layer
        if self.dropout:
            _model.add(keras.layers.Dropout(self.dropout))
        # -- add pool layer
        if self.pool_size:
            _model.add(keras.layers.MaxPooling1D(pool_size=self.pool_size))
        # -- flatten and dense to get to output shape
        _model.add(keras.layers.Flatten())
        _model.add(keras.layers.Dense(self.output_size, activation=self.activation))
        # -- compile
        _model.compile(loss=self.loss, optimizer=self.optimizer)
        # -- return
        return _model

    def reset(self):

        self.model = None
        self.epochs_trained = 0
        self.input_size = None
        self.output_size = None

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = None, reset: bool = False, verbose: int = 0,
            groupby: SequenceOrScalar = None):

        # -- assert
        X, y = to_keras_3d(x=X, window=self.window, y=y, groupby=groupby)

        # -- init
        # - defaults
        if epochs is None:
            epochs = self.epochs

        # - reset
        if reset or self.model is None:
            self.model = self._init_model(X=X, y=y)
            self.epochs_trained = 0

        # -- main
        # - call member model fit
        self.model.fit(x=X, y=y, epochs=epochs, batch_size=self.batch_size, verbose=verbose)
        self.epochs_trained += epochs

    def predict(self, X, groupby: SequenceOrScalar = None):

        # -- assert
        _X = to_keras_3d(X, window=self.window, dropna=False, groupby=groupby)

        # -- main
        # - call member model predict
        _y = self.model.predict(x=_X)

        return _y

        # # check if data was lost -> append NaN
        # _X_shape_diff = X.shape[0] - _X.shape[0]
        # if _X.shape[0] != X.shape[0]:

