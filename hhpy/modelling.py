"""
hhpy.modelling.py
~~~~~~~~~~~~~~~~~

Contains a model class that is based on pandas DataFrames and wraps around sklearn and other frameworks
to provide convenient train test functions.

"""

# ---- imports
# --- standard imports
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- third party imports
from copy import deepcopy
from typing import Sequence, Mapping, Union, Callable, Optional, Any, Tuple, List

from sklearn.exceptions import DataConversionWarning
from sklearn.base import clone

# ---- optional imports
try:
    from IPython.core.display import display
except ImportError:
    display = print

# --- local imports
from hhpy.main import GROUPBY_DUMMY, export, BaseClass, is_list_like, assert_list, tprint, DocstringProcessor, \
    SequenceOrScalar, DFOrArray, list_exclude, list_merge, list_intersection
from hhpy.ds import docstr as docstr_ds, assert_df, k_split, df_score, drop_duplicate_cols
from hhpy.plotting import ax_as_list, legend_outside, rcParams as hpt_rcParams, docstr as docstr_hpt
from hhpy.ipython import display_df

# ---- variables
# --- validations
validations = {
    'Model__predict__return_type': ['y', 'df', 'DataFrame'],
    'Models__predict__return_type': ['y', 'df', 'DataFrame', 'self'],
    'Models__score__return_type': ['self', 'df', 'DataFrame'],
    'Models__score__scores': ['r2', 'rmse', 'mae', 'stdae', 'medae', 'pae'],
    'fit__fit_type': ['train_test', 'k_cross', 'final'],
    'setattr__target': ['all', 'self', 'children', 'a', 's', 'c']
}
# --- docstr
docstr = DocstringProcessor(
    df='Pandas DataFrame containing the training and testing data. '
       'Can be saved to the Model object or supplied on an as needed basis.',
    X_ref='List of features (predictors) used for training the model',
    y_ref='List of labels (targets) to be predicted',
    X='The feature (predictor) data used for training as DataFrame, np.array or column names',
    y='The label (target) data used for training as DataFrame, np.array or column names',
    X_test='The feature (predictor) data used for testing as DataFrame, np.array or column names',
    y_test='The label (target) data used for testing as DataFrame, np.array or column names',
    df_train='Pandas DataFrame containing the training data, optional if array like data is passed for X/y',
    df_test='Pandas DataFrame containing the testing data, optional if array like data is passed for X/y test',
    X_predict='The feature (predictor) data used for predicting as DataFrame, np.array or column names',
    y_predict='The label (target) data used for predicting as DataFrame, np.array or column names. Specifying y '
              'is only necessary for convolutional or time-series type models [optional]',
    df_predict='Pandas DataFrame containing the predict data, optional if array like data is passed for X_predict',
    model_in='Any model object that implements .fit and .predict',
    name='Name of the model, used for naming columns [optional]',
    dropna='Whether to drop rows containing NA in the training data [optional]',
    scaler_X='Scalar object that implements .transform and .inverse_transform, applied to the features (predictors)'
             'before training and inversely after predicting [optional]',
    scaler_y='Scalar object that implements .transform and .inverse_transform, applied to the labels (targets)'
             'before training and inversely after predicting [optional]',
    do_print='Whether to print the steps to console [optional]',
    display_score='Whether to display the score DataFrame [optional]',
    ensemble='if True also predict with Ensemble like combinations of models. If True or mean calculate'
             'mean of individual predictions. If median calculate median of individual predictions. [optional]',
    printf='print function to use for logging [optional]',
    groupby=docstr_ds.params['groupby'],
    key='key of attribute to set',
    value='value of attribute to set',
    **validations
)


# ---- classes
# noinspection PyPep8Naming
@docstr
@export
class Model(BaseClass):
    """
    A unified modeling class that is extended from sklearn, accepts any model that implements .fit and .predict

    :param model: %(model_in)s
    :param name: %(name)s
    :param X_ref: %(X_ref)s
    :param y_ref: %(y_ref)s
    :param groupby: %(groupby)s
    """

    # --- globals
    __name__ = 'Model'
    __attributes__ = ['name', 'model', 'X_ref', 'y_ref', 'groupby', 'is_fit']

    # --- functions
    def __init__(self, model: Any = None, name: str = 'pred', X_ref: SequenceOrScalar = None,
                 y_ref: SequenceOrScalar = None, groupby: SequenceOrScalar = None) -> None:

        # -- assert
        if X_ref is not None:
            X_ref = assert_list(X_ref)
        if y_ref is not None:
            y_ref = assert_list(y_ref)
        # _intersection_X_y = list_intersection(X_ref, y_ref)
        # if len(_intersection_X_y) > 0:
        #     raise ValueError(f"{_intersection_X_y} are in both X_ref and y_ref")

        # -- assign
        self.name = name
        if isinstance(model, str):
            model = eval(model)
        self.model = [model]
        self.X_ref = X_ref
        self.y_ref = y_ref
        self.groupby = groupby
        self.is_fit = False

    def reset(self) -> None:
        """
        reset self to unfit state and delete copy of model

        :return: None
        """

        self.is_fit = False
        self.model = [clone(self.model[0])]

    @docstr
    def fit(self, X: Union[DFOrArray, SequenceOrScalar] = None, y: Union[DFOrArray, SequenceOrScalar] = None,
            df: pd.DataFrame = None, dropna: bool = True, X_test: Union[DFOrArray, SequenceOrScalar] = None,
            y_test: Union[DFOrArray, SequenceOrScalar] = None, df_test: pd.DataFrame = None,
            groupby: SequenceOrScalar = None, k: int = 0, **kwargs) -> None:
        """
        generalized fit method extending on model.fit
        
        :param X: %(X)s
        :param y: %(y)s
        :param df: %(df_train)s
        :param dropna: %(dropna)s
        :param X_test: %(X_test)s
        :param y_test: %(y_test)s
        :param df_test: %(df_test)s
        :param groupby: %(groupby)s
        :param k: index of the model to fit [optional]
        :param kwargs: additional keyword arguments to pass to model.fit [optional]
        :return: None
        """

        # -- assert
        # - groupby
        if groupby is None:
            groupby = self.groupby
        groupby = assert_list(groupby)
        self.groupby = groupby
        # - df
        # if X and y are passed separately: concat them to a DataFrame
        if df is None:

            _df_X = assert_df(X)
            _df_y = assert_df(y)
            # if both X and y are DataFrames we can use their column names
            if isinstance(X, pd.DataFrame) and isinstance(y, (pd.DataFrame, pd.Series)):
                df = pd.concat([_df_X, _df_y], axis=1)
            # if not then we override the default column names when concatting to avoid duplicates
            else:
                df = pd.concat([_df_X, _df_y], ignore_index=True, sort=False, axis=1)
            # save column names to self
            self.X_ref = _df_X.columns
            self.y_ref = _df_y.columns
            # in this case groupby is not supported
            groupby = []
        # get X_ref and y_ref
        else:
            if self.X_ref is None:
                self.X_ref = assert_list(X)
            if self.y_ref is None:
                self.y_ref = assert_list(y)

        df = assert_df(df)
        if df_test is not None:
            df_test = assert_df(df_test)

        # handle dropna
        if dropna:
            _dropna_cols = list_intersection(df.columns, list_merge(self.X_ref, self.y_ref, groupby))
            df = df.dropna(subset=_dropna_cols)
            if df_test is not None:
                df_test = df_test.dropna(subset=_dropna_cols)

        # -- init
        # - X / y train
        _X = df[self.X_ref]
        _y = df[self.y_ref]

        # - X / y test
        if df_test is None:
            _X_test = X_test
            _y_test = y_test
        else:
            _X_test = df_test[self.X_ref]
            _y_test = df_test[self.y_ref]

        # -- fit
        # append a copy of the model if needed
        while k >= len(self.model):
            self.model.append(deepcopy(self.model[0]))
        # get model by index
        _model = self.model[k]
        _varnames = _model.fit.__code__.co_varnames

        # pass X / y test only if fit can handle it
        if 'groupby' not in kwargs.keys() and 'groupby' in _varnames and groupby not in [None, []]:
            _X = pd.concat([_X, df[groupby]], axis=1)
            if _X_test is not None:
                _X_test = pd.concat([_X_test, df_test[groupby]], axis=1)
            kwargs['groupby'] = groupby
        if 'X_test' not in kwargs.keys() and 'X_test' in _varnames:
            kwargs['X_test'] = _X_test
        if 'y_test' not in kwargs.keys() and 'y_test' in _varnames:
            kwargs['y_test'] = _y_test

        warnings.simplefilter('ignore', (FutureWarning, DataConversionWarning))
        _model.fit(X=_X, y=_y, **kwargs)
        warnings.simplefilter('default', (FutureWarning, DataConversionWarning))
        self.is_fit = True

    @docstr
    def predict(self, X: Union[DFOrArray, SequenceOrScalar] = None, y: Union[DFOrArray, SequenceOrScalar] = None,
                df: pd.DataFrame = None, return_type: str = 'y', k_index: pd.Series = None,
                groupby: SequenceOrScalar = None, handle_na: bool = True, multi: SequenceOrScalar = None, **kwargs
                ) -> Union[pd.Series, pd.DataFrame]:
        """
        Generalized predict method based on model.predict
        
        :param X: %(X)s
        :param y: %(y)s
        :param df: %(df)s
        :param return_type: one of %(Model__predict__return_type)s, if 'y' returns a pandas Series / DataFrame
            with only the predictions, if one of 'df','DataFrame' returns the full DataFrame with predictions added
        :param k_index: If specified and model is k_cross split: return only the predictions for each test subset
        :param groupby: %(groupby)s
        :param handle_na: Whether to handle NaN values (prediction will be NaN) [optional]
        :param multi: Postfixes to use for multi output models [optional]
        :param kwargs: additional keyword arguments passed to child model's predict [optional]
        :return: see return_type
        """

        # -- assert
        # - model is fit
        if not self.is_fit:
            raise ValueError('Model is not fit yet')
        # - valid return type
        if return_type not in validations['Model__predict__return_type']:
            raise ValueError(f"return type must be one of {validations['Model__predict__return_type']}")
        # - df
        if df is not None:
            df = assert_df(df)
        else:
            df = pd.concat([assert_df(X), assert_df(y)], axis=1)
        # - groupby
        if groupby in [None, []]:
            groupby = self.groupby
        groupby = assert_list(groupby)

        _y_ref_pred = ['{}_pred'.format(_) for _ in self.y_ref]

        # special case if there is only one target (most algorithms support only one)
        if len(self.y_ref) == 1:
            _y_ref_pred = _y_ref_pred[0]

        # - predict using sklearn api
        _ks = len(self.model)
        _y_labs = []
        for _k, _model in enumerate(self.model):

            # get y lab
            if _ks == 1:
                _y_lab = _y_ref_pred
                _y_labs = [_y_ref_pred]
            else:
                if k_index is None:
                    _y_lab = [f"{_}_{_k}" for _ in assert_list(_y_ref_pred)]
                    _y_labs += _y_lab
                else:
                    _y_labs = [_y_ref_pred]

            # - handle kwargs
            _varnames = _model.predict.__code__.co_varnames
            # groupby
            if 'groupby' not in kwargs.keys() and 'groupby' in _varnames:
                _X = df[self.X_ref + groupby]  # .dropna()
                kwargs['groupby'] = groupby
            else:
                _X = df[self.X_ref]
            # noinspection PyUnresolvedReferences
            _na_indices = _X[_X.isna().any(axis=1)].index.tolist()
            # y
            if 'y' not in kwargs.keys() and 'y' in _varnames:
                _y = df[self.y_ref]
                # noinspection PyUnresolvedReferences
                _na_indices = list_merge(_na_indices, _X[_X.isna().any(axis=1)].index.tolist())
                if handle_na:  # drop na
                    _y = _y.drop(_na_indices)
                kwargs['y'] = _y
            if handle_na:  # drop na (has to be here so y na indices have been handled)
                _X = _X.drop(_na_indices)
            # - predict
            try:
                _y_pred = _model.predict(_X, **kwargs)
            except TypeError:
                # sometimes the model accepts y as keyword argument but cannot actually handle it
                # I'm looking at you sklearn.multioutput
                kwargs.pop('y')
                _y_pred = _model.predict(_X, **kwargs)

            # cast to pandas
            if len(_y_pred) != len(_X):
                raise ValueError(f"The lengths of y_pred ({len(_y_pred)}) and X ({len(_X)}) do not match")
            _y_pred = pd.DataFrame(_y_pred, index=_X.index)
            # bring back dropped nans
            if handle_na:
                _y_pred = pd.concat([_y_pred, pd.DataFrame(np.nan, index=_na_indices, columns=_y_pred.columns)])\
                    .sort_index()

            # feed back to df
            if len(_y_labs) < _y_pred.shape[1]:  # some regressors (TimeSeriesRegressors)
                # might return multi output
                _y_labs_new = []
                if multi is None:
                    multi = _y_pred.shape[1] // len(_y_labs)
                for _y_lab in _y_labs:
                    for _it in range(multi):
                        _y_labs_new.append(f"{_y_lab}_{_it}")
                _y_labs = _y_labs_new + []

            if k_index is None:  # all
                for _it, _y_lab in enumerate(_y_labs):
                    df[_y_lab] = _y_pred[_y_pred.columns[_it]]
            else:
                # during first iteration: assert output columns are available
                if _k == 0:
                    for __y_ref_pred in assert_list(_y_ref_pred):
                        if __y_ref_pred not in df.columns:
                            df[__y_ref_pred] = np.nan
                # write output
                # we loop the columns because _y is possible to be multi output or single output while _y_pred is a DF
                # and np where does weird stuff if you pass a single column on the left and a DF of the right of the ==
                for _col, __y_ref_pred in enumerate(assert_list(_y_ref_pred)):
                    df[__y_ref_pred] = np.where(k_index == _k, _y_pred[_y_pred.columns[_col]], df[__y_ref_pred])

        if return_type == 'y':
            return df[_y_labs]
        else:
            return df

    @docstr
    def setattr(self, key: str, value: Any, target: str = 'all') -> None:
        """
        Set attribute on self and or all k instances of one's model

        :param key: %(key)s
        :param value: %(value)s
        :param target: One of %(setattr__target)s, for 'self' sets only on self, for 'children' sets for children and
            for 'all' tries to set on both
        :return: None
        """

        # -- assert
        if target not in validations['setattr__target']:
            raise ValueError(f"target must be one of {validations['setattr__target']}")

        # -- main
        # - self
        if target in ['all', 'a', 'self', 's'] and hasattr(self, key):
            setattr(self, key, value)

        # - children
        if target in ['all', 'a', 'children', 'c']:
            for _model in self.model:
                if hasattr(_model, key):
                    setattr(_model, key, value)


# noinspection PyPep8Naming
@docstr
@export
class Models(BaseClass):
    """
    Collection of Models that allow for fitting and predicting with multiple Models at once,
    comparing accuracy and creating Ensembles
        
    :param args: multiple Model objects that will form a Models Collection
    :param name: name of the collection
    :param df: %(df)s
    :param X_ref: %(X_ref)s
    :param y_ref: %(y_ref)s
    :param scaler_X: %(scaler_X)s
    :param scaler_y: %(scaler_y)s
    :param printf: %(printf)s
    """

    # --- globals
    __name__ = 'Models'
    __attributes__ = ['models', 'fit_type', 'df', 'X_ref', 'y_ref', 'y_pred', 'groupby', 'scaler_X', 'scaler_y',
                      'model_names', 'df_score', 'k_tests', 'printf']
    __dependent_classes__ = [Model]

    # --- functions
    def __init__(self, *args: Any, df: pd.DataFrame = None, X_ref: SequenceOrScalar = None,
                 y_ref: SequenceOrScalar = None, groupby: SequenceOrScalar = None, scaler_X: Any = None,
                 scaler_y: Any = None, printf: Callable = tprint) -> None:

        # -- assert
        if isinstance(scaler_X, type):
            scaler_X = scaler_X()
        if isinstance(scaler_y, type):
            scaler_y = scaler_y()
        if X_ref is not None:
            X_ref = assert_list(X_ref)
        if y_ref is not None:
            y_ref = assert_list(y_ref)
        if groupby is not None:
            groupby = assert_list(groupby)

        # -- init
        _models = []
        _model_names = []
        # ensure hhpy.modelling.Model
        _it = -1
        if len(args) == 0:
            warnings.warn('No Models passed')
        for _arg in args:
            for _model in assert_list(_arg):

                _it += 1

                if not isinstance(_model, Model):
                    if hasattr(_model, 'name'):
                        _name = _model.name
                    elif hasattr(_model, '__name__'):
                        _name = f"{_model.__name__}_{_it}"
                    else:
                        _name = f"model_{_it}"
                    _model = Model(_model, name=_name, X_ref=X_ref, y_ref=y_ref, groupby=groupby)
                _models.append(deepcopy(_model))
                if _model.name not in _model_names:
                    _model_names.append(_model.name)

        # -- assign
        self.models = _models
        self.fit_type = None
        if df is not None:
            self.df = df.copy()
        else:
            self.df = None
        self.y_ref = y_ref
        if X_ref is not None:
            self.X_ref = X_ref
        elif df is not None:
            # default to all columns not in y_ref / groupby
            self.X_ref = [_ for _ in df.columns if _ not in y_ref + groupby]
        else:
            self.X_ref = []
        self.groupby = groupby
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.model_names = _model_names
        self.printf = printf

        # - not given attributes
        self.df_score = None
        self.k_tests = None
        self.y_pred = []
        self.multi = None

    @property
    def df_test(self) -> Union[None, pd.DataFrame, List[pd.DataFrame]]:
        # None case
        if self.df is None or self.fit_type is None:
            return None
        # train-test case
        if self.fit_type == 'train_test':
            return self.df[lambda _: _['_k_index'] == self.k_tests[0]]
        # k-cross case
        _df_tests = []
        for _k_test in self.k_tests:
            _df_tests.append(self.df[lambda _: _['_k_index'] == _k_test])
        return _df_tests

    def reset(self) -> None:
        """
        reset all child models to an unfit state

        :return: None
        """

        # - loop child models and call reset
        for _model in self.models:
            _model.reset()
        # - reset df score
        self.df_score = None
        # - reset k tests
        self.k_tests = None
        # - drop predictions from df
        for _y_pred in list_intersection(self.y_pred, assert_list(self.df.columns)):
            self.df = self.df.drop(_y_pred, axis=1)

    @docstr
    def k_split(self, **kwargs):
        """
        apply hhpy.ds.k_split to self to create train-test or k-cross ready data
        
        :param kwargs: keyword arguments passed to :func:`~hhpy.ds.k_split`
        :return: None
        """

        # -- init
        if self.df is None:
            raise ValueError('You must specify a df')

        # -- split
        _k_index = k_split(df=self.df, return_type='s', **kwargs)
        self.df['_k_index'] = _k_index

    @docstr
    def model_by_name(self, name: Union[list, str]) -> Union[Model, list]:
        """
        extract a list of Models from the collection by their names

        :param name: name of the Model
        :return: list of Models
        """

        _models = []
        for _model in self.models:
            if _model.name in assert_list(name):
                if not is_list_like(name):
                    return _model
                else:
                    _models.append(_model)
        return _models

    @docstr
    def fit(self, fit_type: str = 'train_test',  k_test: Optional[int] = 0, groupby: SequenceOrScalar = None,
            do_print: bool = True, **kwargs):
        """
        fit all Model objects in collection
        
        :param fit_type: one of %(fit__fit_type)s
        :param k_test: which k_index to use as test data
        :param groupby: %(groupby)s
        :param do_print: %(do_print)s
        :param kwargs: Other keyword arguments passed to :func:`~Model.fit`
        :return: None
        """

        # -- assert
        if fit_type not in validations['fit__fit_type']:
            raise ValueError(f"fit type must be one of {validations['fit__fit_type']}")

        # -- init
        if groupby is None:
            groupby = self.groupby
        # get df
        _df = self.df.copy()
        # groupby and k index
        # -- apply scaler to X and y separately
        warnings.simplefilter('ignore', DataConversionWarning)
        if self.scaler_X is not None and _df[self.X_ref].shape[1] > 0:  # 0 column features cannot be scaled
            self.scaler_X = self.scaler_X.fit(_df[self.X_ref])
            _df[self.X_ref] = self.scaler_X.transform(_df[self.X_ref])
        else:
            self.scaler_X = None
        if self.scaler_y is not None:
            self.scaler_y = self.scaler_y.fit(_df[self.y_ref])
            _df[self.y_ref] = self.scaler_y.transform(_df[self.y_ref])
        warnings.simplefilter('default', DataConversionWarning)
        # split
        if fit_type == 'train_test':
            _k_tests = [k_test]
            # self.df['_test'] = self.df['_k_index'] == k_test
        elif fit_type == 'k_cross':
            _k_tests = sorted(_df['_k_index'].unique())
        else:  # fit_type == 'final'
            _k_tests = [-1]
        # get df train and df test
        for _k_test in _k_tests:

            _df_train = _df[lambda _: _['_k_index'] != _k_test]
            _df_test = _df[lambda _: _['_k_index'] == _k_test]

            for _model in self.models:

                if do_print:
                    self.printf('fitting model {}...'.format(_model.name))

                _model.fit(X=self.X_ref, y=self.y_ref, df=_df_train, df_test=_df_test, groupby=groupby,
                           k=_k_test, **kwargs)

        self.fit_type = fit_type
        self.k_tests = _k_tests

    @docstr
    def predict(
            self, X: Union[DFOrArray, SequenceOrScalar] = None, y: Union[DFOrArray, SequenceOrScalar] = None,
            df: pd.DataFrame = None, return_type: str = 'self', ensemble: bool = False, k_predict_type: str = 'test',
            groupby: SequenceOrScalar = None, multi: SequenceOrScalar = None, do_print: bool = True, **kwargs
    ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """
        predict with all models in collection
        
        :param X: %(X_predict)s
        :param y: %(y_predict)s
        :param df: %(df_predict)s
        :param return_type: one of %(Models__predict__return_type)s
        :param ensemble: %(ensemble)s
        :param k_predict_type: 'test' or 'all'
        :param groupby: %(groupby)s
        :param multi: postfixes to use for multi output [optional]
        :param do_print: %(do_print)s
        :param kwargs: other keyword arguments passed to Model.predict [optional]
        :return: if return_type is self: None, else see Model.predict
        """

        # -- assert
        # return_type
        _valid_return_types = ['self', 'df', 'df_full', 'DataFrame']
        assert(return_type in _valid_return_types), f"return_type must be one of {_valid_return_types}"
        # fit_type
        if not self.fit_type:
            raise ValueError('Model is not fit yet')
        # X / y
        if df is None:
            if isinstance(X, pd.DataFrame):
                if isinstance(y, pd.DataFrame):
                    df = pd.concat([X, y], axis=1)
                else:
                    df = X
            else:
                df = self.df.copy()

        # - handle scaler transformation
        warnings.simplefilter('ignore', DataConversionWarning)
        if self.scaler_X is not None:
            df[self.X_ref] = self.scaler_X.transform(df[self.X_ref])
        if self.scaler_y is not None:
            df[self.y_ref] = self.scaler_y.transform(df[self.y_ref])

        _y_preds = []
        _y_ref_preds = []
        _model_names = []

        for _model in self.models:

            if do_print:
                self.printf('predicting model {}...'.format(_model.name))

            _model_names.append(_model.name)

            # get config for k_cross
            _k_index = None
            if self.fit_type == 'k_cross':
                # k_cross_type test: return only test prediction for each k
                if k_predict_type == 'test':
                    _k_index = self.df['_k_index']
                    _y_ref_preds += [f"{_}_{_model.name}" for _ in _model.y_ref]
                # k_cross_type all: return all predictions
                else:
                    for _k in range(len(_model.model)):
                        _y_ref_preds += [f"{_}_{_model.name}_{_k}" for _ in _model.y_ref]
            else:
                _y_ref_preds += [f"{_}_{_model.name}" for _ in _model.y_ref]

            _y_pred_scaled = _model.predict(df=df, groupby=groupby, k_index=_k_index, **kwargs)

            # - handle scaler inverse transformation
            if self.scaler_y is None:
                _y_pred = deepcopy(_y_pred_scaled)
            else:
                _df_y_pred_scaled = pd.DataFrame(_y_pred_scaled)
                _df_y_pred_scaled.columns = _model.y_ref
                # if the model is predicting only a subset of ys we need to reshape before we scale
                if _model.y_ref != self.y_ref:
                    for _col in self.y_ref:
                        if _col not in _model.y_ref:
                            _df_y_pred_scaled[_col] = np.nan
                    # ensure correct order
                    _df_y_pred_scaled = _df_y_pred_scaled[self.y_ref]

                _df_y_pred = self.scaler_y.inverse_transform(_df_y_pred_scaled)
                _df_y_pred = pd.DataFrame(_df_y_pred)
                _df_y_pred.columns = self.y_ref
                _y_pred = _df_y_pred[_model.y_ref]
            _y_preds.append(_y_pred)
        warnings.simplefilter('default', DataConversionWarning)

        _df = pd.concat(_y_preds, axis=1, sort=False)

        # feed back to df
        if len(_y_ref_preds) < _df.shape[1]:  # some regressors (TimeSeriesRegressors)
            # might return multi output
            _y_ref_preds_new = []
            if multi is None:
                multi = range(_df.shape[1] // len(_y_ref_preds))
            for _y_ref_pred in _y_ref_preds:
                for _multi in multi:
                    _y_ref_preds_new.append(f"{_y_ref_pred}_{_multi}")
            _y_ref_preds = _y_ref_preds_new + []
        else:
            multi = None

        # save
        self.multi = multi

        # to df
        _df.columns = _y_ref_preds
        _df.index = df.index

        if ensemble:  # supports up to 3 model ensembles (does not support multi regression)
            # drop duplicates
            _model_names = list(set(_model_names))
            # get all combinations
            for _comb in list(itertools.combinations(_model_names, 2)) + list(itertools.combinations(_model_names, 3)):
                for _y_ref in self.y_ref:
                    _comb_name = '_'.join(_comb)
                    _y_comb_name = '{}_{}'.format(_y_ref, _comb_name)
                    _cols = []
                    for _element in _comb:
                        _cols.append('{}_{}'.format(_y_ref, _element))
                    _df[_y_comb_name] = _df[_cols].mean(axis=1)
                    if _comb_name not in self.model_names:
                        self.model_names.append(_comb_name)

        if return_type == 'self':
            for _col in _df.columns:
                self.df[_col] = _df[_col]
                if _col not in self.y_pred:
                    self.y_pred.append(_col)
        elif return_type in ['df', 'DataFrame']:
            return _df
        elif return_type in ['df_full']:
            for _col in list_intersection(df.columns, _df.columns):
                df = df.drop(_col, axis=1)
            return pd.concat([df, _df], axis=1)

    @docstr
    def score(
            self, return_type: str = 'self', pivot: bool = False, groupby: SequenceOrScalar = None,
            do_print: bool = True,  display_score: bool = True, display_format: str = ',.3f', **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        calculate score of the Model predictions

        :param return_type: one of %(Models__score__return_type)s
        :param pivot: whether to pivot the DataFrame for easier readability [optional]
        :param do_print: %(do_print)s
        :param display_score: %(display_score)s
        :param display_format: Format to use when displaying the score DataFrame [optional]
        :param groupby: %(groupby)s
        :param kwargs: other keyword arguments passed to :func:`~hhpy.ds.df_score`
        :return: if return_type is 'self': None, else: pandas DataFrame containing the scores
        """

        # -- assert
        _valid_return_types = ['self', 'df', 'df_full', 'DataFrame']
        assert(return_type in _valid_return_types),\
            f"return_type must be one of {_valid_return_types}"
        groupby = assert_list(groupby)
        _df = self.df.copy()

        # -- init
        if do_print:
            self.printf('scoring...')

        if self.fit_type is None:
            raise ValueError('Model is not fit yet')
        elif self.fit_type == 'k_cross':
            groupby.append('_k_index')
        elif self.fit_type == 'train_test':
            _df = _df[lambda _: _['_k_index'].isin(self.k_tests)]

        _df_score = df_score(df=_df, y_true=self.y_ref, pred_suffix=self.model_names, pivot=pivot, groupby=groupby,
                             multi=self.multi, **kwargs)

        if display_score:
            if pivot:
                _display_score = _df_score
            else:
                _display_score = _df_score.pivot_table(index=['y_true', 'y_pred'], columns='score', values='value')
            display_df(_display_score, float_format=display_format)

        if return_type == 'self':
            self.df_score = _df_score
        else:
            return _df_score

    @docstr
    def train(
            self, df: pd.DataFrame = None, k: int = 5, groupby: Union[Sequence, str] = None,
            sortby: Union[Sequence, str] = None, random_state: int = None, fit_type: str = 'train_test',
            k_test: Optional[int] = 0, ensemble: bool = False, scores: Union[Sequence, str, Callable] = None,
            multi: SequenceOrScalar = None,
            scale: float = None, do_predict: bool = True, do_score: bool = True, do_split: bool = True,
            do_fit: bool = True, do_print: bool = True, display_score: bool = True, kws_fit: dict = None,
            kws_pred: dict = None, kws_score: dict = None
    ) -> None:
        """
        wrapper method that combined k_split, train, predict and score

        :param df: %(df)s
        :param k: see hhpy.ds.k_split see hhpy.ds.k_split
        :param groupby: %(groupby)s
        :param sortby: see hhpy.ds.k_split
        :param random_state: see hhpy.ds.k_split
        :param fit_type: see .fit
        :param k_test: see .fit
        :param ensemble: %(ensemble)s
        :param scores: see .score [optional]
        :param multi: postfixes to use for multi output [optional]
        :param scale: see .score
        :param do_print: %(do_print)s
        :param display_score: %(display_score)s
        :param do_split: whether to apply k_split [optional]
        :param do_fit: whether to fit the Models [optional]
        :param do_predict: whether to add predictions to DataFrame [optional]
        :param do_score: whether to create self.df_score [optional]
        :param kws_fit: additional keyword arguments passed to fit [optional]
        :param kws_pred: additional keyword arguments passed to pred [optional]
        :param kws_score: additional keyword arguments passed to score [optional]
        :return: None
        """
        # -- defaults
        if df is not None:
            self.df = df
        if kws_fit is None:
            kws_fit = {}
        if kws_pred is None:
            kws_pred = {}
        if kws_score is None:
            kws_score = {}

        # -- main
        if do_split:
            self.k_split(k=k, groupby=groupby, sortby=sortby, random_state=random_state, do_print=do_print)
        if do_fit:
            self.fit(fit_type=fit_type, k_test=k_test, do_print=do_print, groupby=groupby, **kws_fit)
        if do_predict:
            self.predict(ensemble=ensemble, do_print=do_print, groupby=groupby, multi=multi, **kws_pred)
        if do_score:
            self.score(scores=scores, scale=scale, do_print=do_print, display_score=display_score, **kws_score)
        if do_print:
            self.printf('training done')

    @docstr_hpt
    def scoreplot(
            self, x='y_ref', y='value', hue='model', hue_order=None, row='score', row_order=None,
            palette=None, width=16, height=9 / 2, scale=None, query=None, return_fig_ax=False,
            **kwargs
    ) -> Optional[tuple]:
        """
        plot the score(s) using sns.barplot

        :param x: %(x)s
        :param y: %(y)s
        :param hue: %(hue)s
        :param hue_order: %(order)s
        :param row: the variable to wrap around the rows [optional]
        :param row_order: %(order)s
        :param palette: %(palette)s
        :param width: %(subplot_width)s
        :param height: %(subplot_height)s
        :param scale: scale the values [optional
        :param query: query to be passed to pd.DataFrame.query before plotting [optional]
        :param return_fig_ax: %(return_fig_ax)s
        :param kwargs: other keyword arguments passed to sns.barplot
        :return: see return_fig_ax
        """

        # -- init
        if row_order is None:
            row_order = ['r2', 'rmse', 'mae', 'stdae', 'medae', 'pae']
        if hue_order is None:
            hue_order = self.model_names
        if palette is None:
            palette = hpt_rcParams['palette']
        _row_order = assert_list(row_order)

        fig, ax = plt.subplots(nrows=len(_row_order), figsize=(width, height * len(_row_order)))
        _ax_list = ax_as_list(ax)

        _df_score = self.score(return_type='df', scale=scale, do_print=False, display_score=False)
        if query is not None:
            _df_score = _df_score.query(query)

        _row = -1
        for _row_value in _row_order:
            _row += 1
            _ax = _ax_list[_row]
            sns.barplot(x=x, y=y, hue=hue, hue_order=hue_order, data=_df_score.query('{}==@_row_value'.format(row)),
                        ax=_ax, palette=palette, **kwargs)
            _y_label = _row_value
            if scale:
                _y_label += '_scale={}'.format(scale)
            _ax.set_ylabel(_y_label)
            _ax.set_xlabel('')

        legend_outside(ax)

        if return_fig_ax:
            return fig, ax
        else:
            plt.show(fig)

    @docstr
    def setattr(self, key: str, value: Any, target: str = 'all', child_target: str = 'all') -> None:
        """
        Set attribute on self and or all child models

        :param key: %(key)s
        :param value: %(value)s
        :param target: One of %(setattr__target)s, for 'self' sets only on self, for 'children' sets for children and
            for 'all' tries to set on both
        :param child_target: Same as target but passed to :func:`~Model.setattr`
        :return: None
        """

        # -- assert
        if target not in validations['setattr__target']:
            raise ValueError(f"target must be one of {validations['setattr__target']}")

        # -- main
        # - self
        if target in ['all', 'a', 'self', 's'] and hasattr(self, key):
            setattr(self, key, value)

        # - children
        if target in ['all', 'a', 'children', 'c']:
            for _model in self.models:
                _model.setattr(key, value, target=child_target)


# ---- functions
@export
def assert_array(a: Any, return_name: bool = False, name_default: str = 'name') -> Union[Tuple[np.ndarray, str],
                                                                                         np.ndarray]:
    """
    Take any python object and turn it into a 2d numpy array (if possible). Useful for training neural networks.

    :param a: any python object
    :param return_name: Whether the name should be returned
    :param name_default: The name to fall back to if the object has no name attribute.
    :return: numpy array, if return_name: Tuple [numpy.array, name]
    """
    _name = name_default
    if return_name:
        if isinstance(a, pd.DataFrame):
            _name = ';'.join(a.columns)
        elif hasattr(a, 'name'):
            _name = a.name
    # assert array
    a = np.array(a).copy()
    if len(a.shape) == 1:
        a = a.reshape(-1, 1)
    if return_name:
        return a, _name
    else:
        return a


@export
def dict_to_model(dic: Mapping) -> Model:
    """
    restore a Model object from a dictionary

    :param dic: dictionary containing the model definition
    :return: Model
    """
    # init empty model
    if dic['__name__'] == 'Model':
        _model = Model()
    else:
        raise ValueError('__name__ not recognized')
    # reconstruct from dict
    _model.from_dict(dic)

    return _model


@export
def assert_model(model: Any) -> Model:
    """
    takes any Model, model object or dictionary and converts to Model

    :param model: Mapping or object containing  a model
    :return: Model
    """

    if not isinstance(model, Mapping):
        if not isinstance(model, Model):
            return Model(model)
        return model

    if '__name__' in model.keys():
        if model['__name__'] in ['Model']:
            _model = dict_to_model(model)
        else:
            raise ValueError('__name__ not recognized')
    else:
        raise KeyError('no element named __name__')

    return _model


def force_model(*args, **kwargs):
    warnings.warn('force_model is deprecated, please use assert_model instead', DeprecationWarning)
    return assert_model(*args, **kwargs)


@export
def get_coefs(model: Any, y: SequenceOrScalar):
    """
    get coefficients of a linear regression in a sorted data frame

    :param model: model object
    :param y: name of the coefficients
    :return: pandas DataFrame containing the coefficient names and values
    """

    if isinstance(model, Model):
        _model = model.model
    else:
        _model = model

    assert(hasattr(_model, 'coef_')), 'Attribute coef_ not available, did you specify a linear model?'

    _coef = _model.coef_.tolist()
    _coef.append(model.intercept_)

    _df = pd.DataFrame()
    _df['feature'] = assert_list(y) + ['intercept']
    _df['coef'] = _coef
    _df = _df.sort_values(['coef'], ascending=False).reset_index(drop=True)

    return _df


@export
def get_feature_importance(
        model: object, predictors: Union[Sequence, str], features_to_sum: Mapping = None
) -> pd.DataFrame:
    """
    get feature importance of a decision tree like model in a sorted data frame

    :param model: model object
    :param predictors: names of the predictors properly sorted
    :param features_to_sum: if you want to sum features please provide name mappings
    :return: pandas DataFrame containing the feature importances
    """

    def _get_feature_importance_rf(__model, __predictors, __features_to_sum: Mapping = None):

        # init df for feature importances
        ___df = pd.DataFrame({
            'feature': __predictors,
            'importance': __model.feature_importances_
        })

        # if applicable: sum features
        if __features_to_sum is not None:
            for _key in list(__features_to_sum.keys()):
                ___df['feature'] = np.where(___df['feature'].isin(__features_to_sum[_key]), _key, ___df['feature'])
                ___df = ___df.groupby('feature').sum().reset_index()

        ___df = ___df.sort_values(['importance'], ascending=False).reset_index(drop=True)
        ___df['importance'] = np.round(___df['importance'], 5)

        return ___df

    # get feature importance of a decision tree like model in a sorted data frame
    def _get_feature_importance_xgb(_f_model, _f_predictors, _f_features_to_sum=None):
        # get f_score
        _f_score = _f_model.get_booster().get_fscore()

        # init df to return
        __df = pd.DataFrame()

        # get predictor code
        __df['feature_code'] = list(_f_score.keys())
        __df['feature_code'] = __df['feature_code'].str[1:].astype(int)

        # code importance
        __df['importance_abs'] = list(_f_score.values())
        __df['importance'] = __df['importance_abs'] / __df['importance_abs'].sum()

        __df = __df.sort_values(['feature_code']).reset_index(drop=True)

        __df['feature'] = [_f_predictors[x] for x in range(len(_f_predictors)) if x in __df['feature_code']]

        if _f_features_to_sum is not None:

            for _key in list(_f_features_to_sum.keys()):
                __df['feature'] = np.where(__df['feature'].isin(_f_features_to_sum[_key]), _key, __df['feature'])
                __df = __df.groupby('feature').sum().reset_index()

        __df = __df.sort_values(['importance'], ascending=False).reset_index(drop=True)

        __df['importance'] = np.round(__df['importance'], 5)

        __df = __df[['feature', 'importance']]

        return __df

    # -- main
    try:
        # noinspection PyTypeChecker
        _df = _get_feature_importance_rf(model, predictors, features_to_sum)
        # this is supposed to also work for XGBoost but it was broken in a recent release
        # so below serves as fallback
    except ValueError:
        # noinspection PyTypeChecker
        _df = _get_feature_importance_xgb(model, predictors, features_to_sum)

    return _df


@export
def to_keras_3d(x: DFOrArray, window: int, y: DFOrArray = None, groupby: SequenceOrScalar = None,
                groupby_to_dummy: bool = False, dropna: bool = True, reshape: bool = True)\
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    reformat a DataFrame / 2D array to become a keras compatible 3D array. If dropna is True the first window
    observations get dropped since they will contain NaN values in the required shifted elements.

    :param x: numpy array or DataFrame
    :param window: series-window, how many iterations to convolve
    :param y: accompanying target / label DataFrame or numpy 2d array. If specified a modified version of y will be
        returned to match x's shape where the first window elements have been dropped. [optional]
    :param groupby: column to group by (shift observations in each group) [optional]
    :param groupby_to_dummy: Whether to include the groupby value as pandas Dummy [optional]
    :param dropna: Whether to drop na rows [optional]
    :param reshape: Whether to reshape to keras format observations - timestamps - features [optional]
    :return: if y is None: x as 3d array, else: Tuple[x, y]
    """

    # -- assert
    # if x input is already a 3d-array: return it as is
    if hasattr(x, 'shape') and len(x.shape) == 3:
        if y is None:
            return x
        else:
            return x, y

    # -- init
    # - convert x and y to DataFrames
    x = assert_df(x)
    # - groupby
    if groupby in [None, []]:
        groupby = GROUPBY_DUMMY
        x[GROUPBY_DUMMY] = 1
    groupby = assert_list(groupby)
    # handle groupby to dummy (needs to be here)
    if groupby_to_dummy:
        x = pd.concat([x, pd.get_dummies(x[groupby])], axis=1)
    _x_cols = list_exclude(x.columns, groupby)
    # handle y
    if y is None:
        _df = x
        _y_cols = None
    else:
        y = assert_df(y)
        _y_cols = list_exclude(y.columns, groupby)
        _df = pd.concat([x, y], axis=1)
        _df = drop_duplicate_cols(_df)

    # -- main
    # - init new x as list of window lists
    _x = [[] for _ in range(window)]
    # - init new y as list
    _y = []

    # - groupby loop
    for _index, _df_i in _df.groupby(groupby):
        # shift loop
        for _step in range(window):
            # shift by _step+1 since shifting by 0 would mean using the target (label) value as predictor (feature)
            _x_shift_i = _df_i[_x_cols].shift(_step+1)
            if dropna:
                # drop the first window elements (since they are na in at least 1 cast)
                _x_shift_i = _x_shift_i.iloc[window:]
            _x[_step].append(_x_shift_i)
        # drop the first window elements of y
        if y is not None:
            _y_i = _df_i[_y_cols]
            if dropna:
                _y_i = _y_i.iloc[window:]
            _y.append(_y_i)
    # - use pd concat to make lists of dfs into dfs
    _x = [pd.concat(_) for _ in _x]
    if y is not None:
        _y = pd.concat(_y).values
    # - turn x into keras 3d array using numpy.dstack
    _x = np.dstack(_x)
    # - reshape into the keras format observations - timestamps - features
    if reshape:
        _x_copy = _x.copy()
        _x = np.zeros((_x_copy.shape[0], _x_copy.shape[2], _x_copy.shape[1]))
        for _row in range(_x_copy.shape[0]):
            _x[_row] = _x_copy[_row].T
    # -- return
    if y is None:
        return _x
    else:
        return _x, _y

# TODO - make these members of Models
# def grid_search(model, grid, target, n_random=100, n_random_state=0, goal='acc_test', f=cla, opt_class=None,
#                 opt_score_type=None, ascending=None, do_print=True, display_score=False, float_format=',.4f',
#                 **kwargs):
#     # -- defaults --
#
#     if opt_score_type is None:
#         if opt_class is not None:
#             opt_score_type = 'f1_pr'
#         else:
#             opt_score_type = 'score'
#
#     if ascending is None:
#         # goal can be one of scores keys
#         if goal in ['r2_train', 'r2_test', 'acc_train', 'acc_test', 'true_pos', 'true_neg']:
#             _ascending = False
#         else:
#             _ascending = True
#     else:
#         _ascending = ascending
#
#     # -- init --
#
#     if is_list_like(target):
#         target = target[0]
#         warnings.warn('target cannot be a list for grid_search, using {}'.format(target))
#
#     # you can pass a DataFrame or a dictionary as grid
#     if isinstance(grid, pd.DataFrame):
#         _grid = grid.copy()
#         del grid
#     elif isinstance(grid, Mapping):
#
#         # expand gridf
#
#         _grid = expand_dict_to_df(grid)
#         del grid
#
#     else:
#         raise 'grid should be a dictionary or DataFrame'
#
#     # if a number of random combinations to try was passed, use it to sample grid
#     if n_random is not None:
#         if n_random < _grid.shape[0]:
#             _grid = _grid.sample(n=n_random, random_state=n_random_state).reset_index(drop=True)
#
#     # -- main --
#
#     # iter _grid rows
#     _df_out = []
#
#     _i = 0
#     for _index, _row in _grid.iterrows():
#
#         _it_df = pd.DataFrame({'it': [_i]})
#         _i += 1
#
#         _model = deepcopy(model)
#         _string = ''
#
#         for _key in _row.keys():
#
#             _value = _row[_key]
#             if isinstance(_value, float):
#                 if _value.is_integer(): _value = int(_value)
#
#             if len(_string) > 0: _string += ', '
#             _string += '{} = {}'.format(_key, _value)
#
#             _model.set_params(**{_key: _value})
#             _it_df[_key] = _value
#
#         if do_print: tprint('{} / {} - {}'.format(_i, _grid.shape[0], _string))
#
#         # execute function
#         _dict = f(model=_model, target=target, do_print=False, display_score=display_score, **kwargs)
#
#         # parse score df
#         if opt_score_type == 'score':
#             _score = _dict['score']
#         elif opt_score_type == 'f1_pr':
#             # switch case for k_cross
#             if 'f1_pr' in _dict['cm'][target].keys():
#                 _score = _dict['cm'][target]['f1_pr']
#             else:
#                 _score = _dict['cm'][target]['f1_pr_test']
#         elif opt_score_type == 'cm':
#             # switch case for k_cross
#             if 'cm' in _dict['cm'][target].keys(): _score = _dict['cm'][target]['f1_pr']
#             _score = _dict['cm'][target]['cm_test']
#         else:
#             raise ValueError('Unknown opt_score_type {}'.format(opt_score_type))
#
#         if opt_class is not None: _score = _score[_score.index == opt_class]
#
#         for _col in _score.columns: _it_df[_col] = _score[_col].mean()
#
#         _df_out.append(_it_df)
#
#     _df_out = pd.concat(_df_out, ignore_index=True, sort=False).sort_values(by=goal, ascending=_ascending)
#
#     # find best combination
#     if do_print:
#         tprint('')
#         print('best combination:')
#         display_df(_df_out.head(1), float_format=float_format)
#
#     return _df_out
#
#
# # wrapper for grid_search for regression
# def grid_reg(*args, goal='rmse_test_scaled', f=reg, **kwargs):
#     return grid_search(*args, goal=goal, f=f, **kwargs)
#
#
# # wrapper for grid_search for regression
# def grid_cla(*args, goal='acc_test', display_cm=False, display_f1_pr=False, display_score=False, f=cla, **kwargs):
#     return (
#         grid_search(*args, goal=goal, display_cm=display_cm, display_f1_pr=display_f1_pr, display_score=display_score,
#                     f=f, **kwargs))
#
#
# # just a wrapper for grid_reg that uses k_cross validation by default
# def grid_reg_k(f=k_cross_reg, goal='rmse', **kwargs):
#     return grid_search(goal=goal, f=f, **kwargs)
#
#
# def multi_modelling(f, goal, ascending, df_train, df_test, target, predictors_it, predictors_fix=None,
# fixed_only=True,
#                     do_print=True, print_prefix='', **kwargs):
#     # init
#     if predictors_fix is None:
#         predictors_fix = []
#     if not is_list_like(target): target = [target]
#     if not is_list_like(predictors_it): predictors_it = [predictors_it]
#     if not is_list_like(predictors_fix): predictors_fix = [predictors_fix]
#
#     # a predictor cannot be both fix and it
#     if len(predictors_fix) > 0:
#         _predictors_it = [_ for _ in predictors_it if _ not in predictors_fix]
#         if fixed_only: _predictors_it += [{'fixed_only': []}]
#     else:
#         _predictors_it = predictors_it
#
#     ## empty dict so that it can become a DtaFrame
#     _df_score = []
#
#     _i = 0
#     _i_max = len(_predictors_it)
#
#     for _predictor in _predictors_it:
#
#         _i += 1
#
#         if isinstance(_predictor, Mapping):
#             _predictor_name = list(_predictor.keys())[0]
#             _predictor_value = _predictor[_predictor_name]
#         else:
#             _predictor_name = _predictor
#             _predictor_value = _predictor
#
#         if not is_list_like(_predictor_value):
#             _predictors_score = [_predictor_value]
#         else:
#             _predictors_score = [] + _predictor_value
#
#         if predictors_fix is not None: _predictors_score = _predictors_score + predictors_fix
#
#         _print_prefix = '{}iteration {} / {} : {} - '.format(print_prefix, _i, _i_max, _predictor_name)
#
#         _rmse_test = 0
#         _rmse_train = 0
#
#         _score = \
#             f(df_train=df_train, df_test=df_test, target=target, predictors=_predictors_score,
#               print_prefix=_print_prefix,
#               do_print=do_print, display_score=False, **kwargs)['score']
#         _score = pd.DataFrame(_score[_score.index.isin(target)].mean()).T
#         _score['predictor'] = _predictor_name
#         _score['predictor_value'] = _predictor_value
#
#         _df_score.append(_score)
#
#     _df_score = pd.concat(_df_score, ignore_index=True, sort=False).sort_values(by=goal,
#                                                                                 ascending=ascending).reset_index(
#         drop=True)
#
#     if do_print:
#         tprint('done')
#
#     return _df_score


# # wrapper
# def multi_cla(f=cla, goal='acc_test', ascending=False, **kwargs):
#     return multi_modelling(f=f, goal=goal, ascending=ascending, **kwargs)
#
#
# # wrapper
# def multi_reg(f=reg, goal='rmse_test_scaled', ascending=True, **kwargs):
#     return multi_modelling(f=f, goal=goal, ascending=ascending, **kwargs)
#
#
# # iteratively build a model by trying all predictors one by one, finding the best one and then trying
# # all again (repeat)
# def iter_modelling(f, goal, df_train, df_test, target, predictors_it, predictors_fix=None, max_depth=10, cutoff=0,
#                    do_print=True, **kwargs):
#     if predictors_fix is None:
#         predictors_fix = []
#     if cutoff is None: cutoff = - np.inf
#
#     # init
#     _predictors_fix = deepcopy(predictors_fix)
#     _predictors_it = deepcopy(predictors_it)
#     if not is_list_like(_predictors_fix): _predictors_fix = [_predictors_fix]
#     _predictors_fix = list(_predictors_fix)
#     if max_depth > len(predictors_it): max_depth = len(predictors_it)
#
#     # a predictor cannot be both fix and it
#     _predictors_it = [_ for _ in _predictors_it if _ not in _predictors_fix]
#
#     _prev_rmse = 9 ** 9
#
#     for _depth in range(max_depth):
#
#         _print_prefix = 'depth {} / {} - '.format(_depth + 1, max_depth)
#
#         _df_score = f(df_train=df_train, df_test=df_test, target=target, predictors_it=_predictors_it,
#                       predictors_fix=_predictors_fix, fixed_only=False, do_print=do_print, print_prefix=_print_prefix,
#                       **kwargs)
#         _best = _df_score.iloc[0]
#         _best_goal = _best[goal]
#         _best_predictor = _best['predictor_value']
#
#         print('{}best result: {}'.format(_print_prefix, format_iloc(_best)))
#
#         if _prev_rmse - _best_goal > cutoff:
#
#             _prev_rmse = _best_goal
#
#             # update predictors
#             if not is_list_like(_best_predictor):
#
#                 _predictors_it = [_ for _ in _predictors_it if _ != _best_predictor]
#                 _predictors_fix += [_best_predictor]
#
#             else:
#
#                 _predictors_it_new = []
#
#                 for _predictor in _predictors_it:
#
#                     if not isinstance(_predictor, Mapping):
#                         _predictors_it_new.append(_predictor)
#                     else:
#                         if not _best['predictor'] in _predictor.keys(): _predictors_it_new.append(_predictor)
#
#                 _predictors_it = deepcopy(_predictors_it_new)
#                 _predictors_fix += _best_predictor
#
#         else:
#
#             print('{}cutoff {} not reached, stopping'.format(_print_prefix, cutoff))
#             break
#
#     return None
#
#
# # wrapper
# def iter_cla(f=multi_cla, goal='acc_test', **kwargs):
#     return iter_modelling(f=f, goal=goal, **kwargs)
#
#
# # wrapper
# def iter_reg(f=multi_reg, goal='rmse_test_scaled', **kwargs):
#     return iter_modelling(f=f, goal=goal, **kwargs)
#
#
# # iteratively build a model by trying all predictors one by one and accepting those who improve more than cutoff
# # (faster than iter_reg)
# def forward_reg(df_train, df_test, predictors_it, predictors_fix, max_depth=10, cutoff=.001, do_print=True, f=reg,
#                 **kwargs):
#     _predictors_fix = deepcopy(predictors_fix)
#     _predictors_it = deepcopy(predictors_it)
#     if not is_list_like(_predictors_fix): _predictors_fix = [_predictors_fix]
#     _predictors_fix = list(_predictors_fix)
#     _predictors_final = [] + _predictors_fix
#
#     # a predictor cannot be both fix and it
#     _predictors_it = [_ for _ in _predictors_it if _ not in _predictors_fix]
#
#     if do_print: tprint('training base model...')
#
#     _dict = f(
#         df_train=df_train,
#         df_test=df_test,
#         predictors=predictors_fix,
#         do_print=False,
#         **kwargs
#     )
#
#     _base = _dict['score'].iloc[0]
#
#     _prev_rmse = _base['rmse_test_scaled']
#
#     print('base score: rmse_scaled train / test : {:.4f} / {:.4f} - rmse train / test : {:.4f} / {:.4f}'.format(
#         _base['rmse_train_scaled'], _base['rmse_test_scaled'], _base['rmse_train'], _base['rmse_test']))
#
#     for _depth in range(max_depth):
#
#         _print_prefix = 'depth {} / {} - '.format(_depth + 1, max_depth)
#
#         # do a multi_reg
#         _df_score = multi_reg(df_train=df_train, df_test=df_test, predictors_it=_predictors_it,
#                               predictors_fix=_predictors_fix, fixed_only=False, do_print=do_print,
#                               print_prefix=_print_prefix, f=reg, **kwargs)
#
#         # find base model
#         _df_score_base = _df_score.query('@_prev_rmse-rmse_test_scaled>@cutoff')
#
#         # if the len is zero no improvements are possible -> quit
#         if _df_score_base.shape[0] == 0:
#             if do_print: print('{}no improvements possible, quitting'.format(_print_prefix))
#             break
#
#         # get all predictors that made improvements:
#         _predictors_pos_name = _df_score_base['predictor'].tolist()
#         _predictors_pos = _df_score_base['predictor_value'].tolist()
#         # print(_predictors_pos)
#         _predictors_fix_new = _predictors_fix + flatten(_predictors_pos)
#
#         if do_print: print('{}accepted predictors: {}'.format(_print_prefix, _predictors_pos_name))
#
#         if do_print: tprint('{}training new base model...'.format(_print_prefix))
#
#         # get new base
#         _dict_new = f(
#             df_train=df_train,
#             df_test=df_test,
#             predictors=_predictors_fix_new,
#             do_print=False,
#             **kwargs
#         )
#
#         _base_new = _dict_new['score'].iloc[0]
#
#         print('{}new score: rmse_scaled train / test : {:.4f} / {:.4f} - rmse train / test : {:.4f} / {:.4f}'.format(
#             _print_prefix, _base_new['rmse_train_scaled'], _base_new['rmse_test_scaled'], _base_new['rmse_train'],
#             _base_new['rmse_test']))
#
#         # new rmse
#         if _prev_rmse - _base_new['rmse_test_scaled'] > cutoff:
#
#             _prev_rmse = deepcopy(_base_new['rmse_test_scaled'])
#             _base = _base_new.copy()
#             _dict = deepcopy(_dict_new)
#             _predictors_fix = deepcopy(_predictors_fix_new)
#             _predictors_final += _predictors_pos_name
#
#             # update predictors
#             for _new_predictor, _new_predictor_name in zip(_predictors_pos, _predictors_pos_name):
#
#                 if not is_list_like(_new_predictor):
#
#                     _predictors_it = [_ for _ in _predictors_it if _ != _new_predictor]
#
#                 else:
#
#                     _predictors_it_new = []
#
#                     for _predictor in _predictors_it:
#
#                         if not isinstance(_predictor, Mapping):
#                             _predictors_it_new.append(_predictor)
#                         else:
#                             if not _new_predictor_name in _predictor.keys(): _predictors_it_new.append(_predictor)
#
#                     _predictors_it = deepcopy(_predictors_it_new)
#         else:
#             print('{}overall score improvement below cutoff {}, quitting'.format(_print_prefix, cutoff))
#             break
#
#         if len(_predictors_it) == 0:
#             print('{}accepted all predictors, quitting'.format(_print_prefix))
#             break
#
#     print('')
#     print('final predictors: {}'.format(_predictors_final))
#     print('discarded predictors: {}'.format(_predictors_it))
#     print('final score: rmse_scaled train / test : {:.4f} / {:.4f} - rmse train / test : {:.4f} / {:.4f}'.format(
#         _base['rmse_train_scaled'], _base['rmse_test_scaled'], _base['rmse_train'], _base['rmse_test']))
#
#     return _dict
#
#
# # iteratively build a model by trying all predictors and removing one by one and discarding those whose gain
# # was less than cutoff (faster than iter_reg)
# def backward_modelling(f, goal, ascending, cutoff, df_train, df_test, target, predictors_it, predictors_fix=None,
#                        max_depth=3, do_print=True, float_format=',.4f', **kwargs):
#     # -- init --
#     if predictors_fix is None:
#         predictors_fix = []
#     _predictors_fix = deepcopy(predictors_fix)
#     _predictors_it = deepcopy(predictors_it)
#     if not is_list_like(_predictors_fix): _predictors_fix = [_predictors_fix]
#     _predictors_fix = list(_predictors_fix)
#     _predictors_final = [] + _predictors_fix
#
#     # a predictor cannot be both fix and it
#     _predictors_it = [_ for _ in _predictors_it if _ not in _predictors_fix]
#     _predictors_all = predictors_fix + _predictors_it
#
#     # -- base --
#     if do_print: tprint('training base model...')
#
#     _dict = f(
#         df_train=df_train,
#         df_test=df_test,
#         target=target,
#         predictors=_predictors_all,
#         do_print=False,
#         display_score=False,
#         **kwargs
#     )f
#
#     _score = _dict['score']
#     _score = _score[_score.index.isin(target)]
#     _base = _score.mean()
#
#     _best_goal = _base[goal]
#
#     print('base score: {}'.format(format_iloc(_base, float_format=float_format)))
#
#     # -- main loop --
#     for _depth in range(max_depth):
#
#         _print_prefix = 'depth {} / {} - '.format(_depth + 1, max_depth)
#
#         tprint('{}init...'.format(_print_prefix))
#
#         _i = 0
#         _i_max = len(_predictors_it)
#         _predictors_it_new = deepcopy(_predictors_it)
#         _removed = 0
#
#         # loop all remaining predictors
#         for _predictor in _predictors_it:
#
#             _i += 1
#
#             _predictors = [_ for _ in _predictors_all if _ != _predictor]
#
#             # try:
#             if True:
#
#                 _dict_new = f(
#                     df_train=df_train,
#                     df_test=df_test,
#                     target=target,
#                     predictors=_predictors,
#                     do_print=False,
#                     display_score=False,
#                     **kwargs
#                 )
#
#                 _score_new = _dict_new['score']
#                 _score_new = _score_new[_score_new.index.isin(target)]
#                 _base_new = _score_new.mean()
#
#                 _goal = _base_new[goal]
#                 _full_score = format_iloc(_base_new, float_format=float_format)
#                 _new_score = _base_new[goal]
#                 _new_score_formatted = format_iloc(_base_new[[goal]], float_format=float_format)
#
#                 if ascending:
#                     _goal_diff = _best_goal - _new_score
#                 else:
#                     _goal_diff = _new_score - _best_goal
#
#                 tprint(
#                     '{}predictor {} / {} : {} ; {} - best: {} - diff: {}'.format(_print_prefix, _i, _i_max,
#                     _predictor,
#                                                                                  _new_score_formatted,
#                                                                                  format(_best_goal, float_format),
#                                                                                  format(_goal_diff, float_format)))
#
#                 if _goal_diff <= cutoff:
#
#                     _predictors_all = deepcopy(_predictors)
#                     _predictors_it_new.remove(_predictor)
#                     _removed += 1
#                     _dict = deepcopy(_dict_new)
#
#                     # if the diff is below 0 a new best score was reached
#                     if _goal_diff < 0: _best_goal = deepcopy(_new_score)
#
#             # except:
#             #    print('')
#             #    print('predictor {} failed'.format(_predictor))
#             #    pass
#
#         if _removed == 0:
#             tprint('')
#             print('no improvements possible, quitting')
#             break
#
#         _predictors_it = deepcopy(_predictors_it_new)
#
#         tprint('')
#         print('{}predictors {} ; {}'.format(_print_prefix, _predictors_all, _full_score))
#
#         if len(_predictors_all) == 1:
#             print('only 1 predictor left, quitting')
#             break
#
#     return _dict
#
#
# def backward_cla(f=cla, goal='acc_test', ascending=True, cutoff=0, float_format=',.3f', **kwargs):
#     return backward_modelling(f=f, goal=goal, ascending=ascending, cutoff=cutoff, float_format=float_format,
#     **kwargs)
#
#
# def backward_reg(f=reg, goal='rmse_test_scaled', ascending=False, cutoff=0, **kwargs):
#     return backward_modelling(f=f, goal=goal, ascending=ascending, cutoff=cutoff, **kwargs)
