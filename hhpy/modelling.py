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
from typing import Sequence, Mapping, Union, Callable, Optional, Any

from sklearn.exceptions import DataConversionWarning

try:
    from IPython.core.display import display
except ImportError:
    display = print

# --- local imports
from hhpy.main import export, BaseClass, is_list_like, force_list, tprint, DocstringProcessor, SequenceOrScalar
from hhpy.ds import _assert_df, k_split, df_score
from hhpy.plotting import ax_as_list, legend_outside, rcParams as hpt_rcParams, docstr as docstr_hpt

# ---- variables
# --- validations
validations = {
    'Model__predict__return_type': ['y', 'df', 'DataFrame'],
    'Models__predict__return_type': ['y', 'df', 'DataFrame', 'self'],
    'Models__score__return_type': ['self', 'df', 'DataFrame'],
    'Models__score__scores': ['r2', 'rmse', 'mae', 'stdae', 'medae'],
    'fit__fit_type': ['train_test', 'k_cross', 'final'],
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
    df_predict='Pandas DataFrame containing the predict data, optional if array like data is passed for X_predict',
    model_in='Any model object that implements .fit and .predict',
    name='Name of the model, used for naming columns [optional]',
    dropna='Whether to drop rows containing NA in the training data [optional]',
    scaler_X='Scalar object that implements .transform and .inverse_transform, applied to the features (predictors)'
             'before training and inversely after predicting',
    scaler_y='Scalar object that implements .transform and .inverse_transform, applied to the labels (targets)'
             'before training and inversely after predicting',
    do_print='Whether to print the steps to console',
    display_score='Whether to display the score DataFrame',
    ensemble='if True also predict with Ensemble like combinations of models. If True or mean calculate'
             'mean of individual predictions. If median calculate median of individual predictions.',
    printf='print function to use for logging',
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
    """

    # --- globals
    __name__ = 'Model'
    __attributes__ = ['name', 'model', 'X_ref', 'y_ref', 'is_fit']

    # --- functions
    def __init__(self, model: Any = None, name: str = 'pred', X_ref: SequenceOrScalar = None,
                 y_ref: SequenceOrScalar = None) -> None:

        # -- init
        if X_ref is not None:
            X_ref = force_list(X_ref)
        if y_ref is not None:
            y_ref = force_list(y_ref)

        # -- assign
        self.name = name
        if isinstance(model, str):
            model = eval(model)
        self.model = model
        self.X_ref = X_ref
        self.y_ref = y_ref
        self.is_fit = False

    @docstr
    def fit(self, X: Union[np.ndarray, Sequence, str] = None, y: Union[np.ndarray, Sequence, str] = None,
            df: pd.DataFrame = None, dropna: bool = True, X_test: Union[np.ndarray, Sequence, str] = None,
            y_test: Union[np.ndarray, Sequence, str] = None, df_test: pd.DataFrame = None) -> None:
        """
        generalized fit method extending on model.fit
        
        :param X: %(X)s
        :param y: %(y)s
        :param df: %(df_train)s
        :param dropna: %(dropna)s
        :param X_test: %(X_test)s
        :param y_test: %(y_test)s
        :param df_test: %(df_test)s
        :return: None
        """

        if df is None:

            if isinstance(X, pd.DataFrame) and (isinstance(y, pd.DataFrame) or isinstance(y, pd.Series)):

                _df_X = X.copy()
                _df_y = pd.DataFrame(y.copy())
                df = pd.concat([_df_X, _df_y], axis=1)

            else:

                _df_X = pd.DataFrame(X)
                _df_y = pd.DataFrame(y)
                df = pd.concat([_df_X, _df_y], ignore_index=True, sort=False, axis=1)

            self.X_ref = _df_X.columns
            self.y_ref = _df_y.columns

            del _df_X, _df_y

        else:

            if self.X_ref is None:
                self.X_ref = force_list(X)
            if self.y_ref is None:
                self.y_ref = force_list(y)
            df = _assert_df(df)

        # handle dropna
        if dropna:
            df = df.dropna(subset=self.X_ref + self.y_ref)
            if df_test is None:
                df_test = df_test.dropna(subset=self.X_ref + self.y_ref)

        # - init X / y train
        _X = df[self.X_ref]
        _y = df[self.y_ref]

        # - init X / y test
        if df_test is None:
            _X_test = X_test
            _y_test = y_test
        else:
            _X_test = df_test[self.X_ref]
            _y_test = df_test[self.y_ref]

        # -- fit

        _kwargs = {'X': _X, 'y': _y}

        # pass X / y test only if fit can handle it
        if 'X_test' in self.model.fit.__code__.co_varnames:
            _kwargs['X_test'] = _X_test
        if 'y_test' in self.model.fit.__code__.co_varnames:
            _kwargs['y_test'] = _y_test

        warnings.simplefilter('ignore', (FutureWarning, DataConversionWarning))
        self.model.fit(**_kwargs)
        warnings.simplefilter('default', (FutureWarning, DataConversionWarning))
        self.is_fit = True

    @docstr
    def predict(self, X=None, df=None, return_type='y') -> Union[pd.Series, pd.DataFrame]:
        """
        Generalized predict method based on model.predict
        
        :param X: %(X)s
        :param df: %(df)s
        :param return_type: one of %(Model__predict__return_type)s, if 'y' returns a pandas Series / DataFrame
            with only the predictions, if one of 'df','DataFrame' returns the full DataFrame with predictions added
        :return: see return_type
        """

        if not self.is_fit:
            raise ValueError('Model is not fit yet')

        _valid_return_types = validations['Model__predict__return_type']
        assert(return_type in _valid_return_types), 'return type must be one of {}'.format(_valid_return_types)

        # ensure pandas df
        if df is None:
            _df = pd.DataFrame(X)
        else:
            _df = df.copy()
            del df

        _X = _df[self.X_ref].dropna()

        _y_ref = ['{}_pred'.format(_) for _ in self.y_ref]

        # special case if there is only one target (most algorithms support only one)
        if len(self.y_ref) == 1:
            _y_ref = _y_ref[0]

        # predict using sklearn api
        _y_pred = self.model.predict(_X)

        # cast to pandas
        _y_pred = pd.DataFrame(_y_pred)
        _y_pred.index = _X.index

        # feed back to df
        _df[_y_ref] = _y_pred

        if return_type == 'y':
            return _df[_y_ref]
        elif return_type in ['df', 'DataFrame']:
            return _df
        else:
            raise ValueError('{} is not a valid return_type'.format(return_type))


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
    __attributes__ = ['models', 'fit_type', 'df', 'X_ref', 'y_ref', 'scaler_X', 'scaler_y', 'model_names', 'df_score']

    # --- functions
    def __init__(self, *args: Any, df: pd.DataFrame = None, X_ref: SequenceOrScalar = None,
                 y_ref: SequenceOrScalar = None, scaler_X: Any = None, scaler_y: Any = None,
                 printf: Callable = tprint) -> None:

        # -- assert
        if isinstance(scaler_X, type):
            scaler_X = scaler_X()
        if isinstance(scaler_y, type):
            scaler_y = scaler_y()

        # -- init
        _models = []
        _model_names = []
        # ensure hhpy.modelling.Model
        _it = -1
        for _arg in args:
            for _model in force_list(_arg):

                _it += 1

                if not isinstance(_model, Model):
                    if 'name' in dir(_model):
                        _name = _model.name
                    elif '__name__' in dir(_model):
                        _name = '{}_{}'.format(_model.__name__, _it)
                    else:
                        _name = 'model_{}'.format(_it)
                    _model = Model(_model, name=_name)
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
        self.y_ref = force_list(y_ref)
        if X_ref:
            self.X_ref = force_list(X_ref)
        elif df is not None:
            # default to all columns not in y_ref
            self.X_ref = [_ for _ in df.columns if _ not in self.y_ref]
        else:
            self.X_ref = []
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.model_names = _model_names
        self.df_score = None
        self.printf = printf

    @docstr
    def k_split(self, **kwargs):
        """
        apply hhpy.ds.k_split to self to create train-test or k-cross ready data
        
        :param kwargs: keyword arguments passed to hhpy.ds.k_split
        :return: None
        """

        # -- init
        assert self.df is not None, 'You must specify a df'

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
            if _model.name in force_list(name):
                if not is_list_like(name):
                    return _model
                else:
                    _models.append(_model)
        return _models

    @docstr
    def fit(self, fit_type: str = 'train_test',  k_test: Optional[int] = None, do_print: bool = True):
        """
        fit all Model objects in collection
        
        :param fit_type: one of %(fit__fit_type)s
        :param k_test: which k_index to use as test data
        :param do_print: %(do_print)s
        :return: None
        """

        # TODO - CROSS VALIDATION

        _valid_fit_types = validations['fit__fit_type']
        assert fit_type in _valid_fit_types, 'fit type must be one of {}'.format(_valid_fit_types)
        assert not ((fit_type != 'final') and ('_k_index' not in self.df.columns)), 'DataFrame is not k_split yet'
        assert fit_type != 'cross', 'cross validation not implemented yet'

        _df_X = self.df[self.X_ref]
        _df_y = self.df[self.y_ref]
        # -- apply scaler
        warnings.simplefilter('ignore', DataConversionWarning)
        if self.scaler_X is not None:
            self.scaler_X = self.scaler_X.fit(_df_X)
            _df_X_scaled = pd.DataFrame(self.scaler_X.transform(_df_X), columns=_df_X.columns)
        else:
            _df_X_scaled = _df_X.copy()
        if self.scaler_y is not None:
            self.scaler_y = self.scaler_y.fit(_df_y)
            _df_y_scaled = pd.DataFrame(self.scaler_y.transform(_df_y), columns=_df_y.columns)
        else:
            _df_y_scaled = _df_y.copy()
        warnings.simplefilter('default', DataConversionWarning)

        _df_scaled = pd.concat([_df_X_scaled, _df_y_scaled], axis=1)

        _df_scaled['_k_index'] = self.df['_k_index']

        if fit_type == 'train_test':
            _k_tests = [_df_scaled['_k_index'].max()]
            if k_test is None:
                k_test = _k_tests[0]
            self.df['_test'] = self.df['_k_index'] == k_test
        elif fit_type == 'k_cross':
            _k_tests = sorted(_df_scaled['_k_index'].unique())
        else:
            _k_tests = [-1]

        for _k_test in _k_tests:

            _ = _k_test
            _df_train = _df_scaled.query('_k_index!=@_k_test')
            _df_test = _df_scaled.query('_k_index==@_k_test')

            for _model in self.models:

                if do_print:
                    self.printf('fitting model {}...'.format(_model.name))

                _model.fit(X=self.X_ref, y=self.y_ref, df=_df_train, df_test=_df_test)

        self.fit_type = fit_type

    @docstr
    def predict(
            self, X=None, df=None, return_type='self', ensemble=False, do_print=True
    ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """
        predict with all models in collection
        
        :param X: %(X_predict)s
        :param df: %(df_predict)s
        :param return_type: one of %(Models__predict__return_type)s
        :param ensemble: %(ensemble)s
        :param do_print: %(do_print)s
        :return: if return_type is self: None, else see Model.predict
        """

        _valid_return_types = ['self', 'df', 'DataFrame']

        assert(return_type in _valid_return_types), 'return_type must be one of {}'.format(_valid_return_types)

        if not self.fit_type:
            raise ValueError('Model is not fit yet')

        if X is None:
            if df is None:
                X = self.df
            else:
                X = df

        if df is not None:
            _X = df[X]
        elif isinstance(X, pd.DataFrame):
            _X = X[self.X_ref]
        else:
            _X = X

        if self.scaler_X is not None:
            # noinspection PyUnresolvedReferences
            _X = self.scaler_X.transform(_X)
        _X = pd.DataFrame(_X, columns=self.X_ref)

        _y_preds = []
        _y_ref_preds = []
        _model_names = []

        for _model in self.models:

            if do_print:
                self.printf('predicting model {}...'.format(_model.name))

            _model_names.append(_model.name)
            _y_ref_preds += ['{}_{}'.format(_, _model.name) for _ in _model.y_ref]
            _y_pred_scaled = _model.predict(X=_X)
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

        # drop duplicates
        _model_names = list(set(_model_names))

        _df = pd.concat(_y_preds, axis=1, sort=False)
        _df.columns = _y_ref_preds

        if ensemble:  # supports up to 3 model ensembles
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
        elif return_type in ['df', 'DataFrame']:
            return _df

    @docstr
    def score(
            self, return_type: str = 'self', pivot: bool = False, do_print: bool = True,  display_score: bool = True,
            **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        calculate score of the Model predictions

        :param return_type: one of %(Models__score__return_type)s
        :param pivot: whether to pivot the DataFrame for easier readability [optional]
        :param do_print: %(do_print)s
        :param display_score: %(display_score)s
        :param kwargs: other keyword arguments passed to :func: `~hhpy.ds.df_score`
        :return: if return_type is 'self': None, else: pandas DataFrame containing the scores
        """

        if do_print:
            self.printf('scoring...')

        _valid_return_types = ['self', 'df', 'DataFrame']
        assert(return_type in _valid_return_types),\
            'return_type must be one of {}'.format(_valid_return_types)

        if not self.fit_type:
            raise ValueError('Model is not fit yet')
        elif self.fit_type == 'train_test':
            _df = self.df.query('_test')
        else:  # self.fit_type == 'final'
            _df = self.df.copy()

        _df_score = df_score(df=_df, y_true=self.y_ref, pred_suffix=self.model_names, pivot=pivot, **kwargs)

        if display_score:
            if pivot:
                _display_score = _df_score
            else:
                _display_score = _df_score.pivot_table(index=['y_ref', 'model'], columns='score', values='value')
            display(_display_score)

        if return_type == 'self':
            self.df_score = _df_score
        else:
            return _df_score

    @docstr
    def train(
            self, df: pd.DataFrame = None, k: int = 5, groupby: Union[Sequence, str] = None,
            sortby: Union[Sequence, str] = None, random_state: int = None, fit_type: str = 'train_test',
            k_test: Optional[int] = None, ensemble: bool = False, scores: Union[Sequence, str] = None,
            scale: float = None, do_predict: bool = True, do_score: bool = True, do_split: bool = True,
            do_fit: bool = True, do_print: bool = True, display_score: bool = True
    ) -> None:
        """
        wrapper method that combined k_split, train, predict and score

        :param df: %(df)s
        :param k: see hhpy.ds.k_split see hhpy.ds.k_split
        :param groupby: see hhpy.ds.k_split
        :param sortby: see hhpy.ds.k_split
        :param random_state: see hhpy.ds.k_split
        :param fit_type: see .fit
        :param k_test: see .fit
        :param ensemble: %(ensemble)s
        :param scores: see .score
        :param scale: see .score
        :param do_print: %(do_print)s
        :param display_score: %(display_score)s
        :param do_split: whether to apply k_split [optional]
        :param do_fit: whether to fit the Models [optional]
        :param do_predict: whether to add predictions to DataFrame [optional]
        :param do_score: whether to create self.df_score [optional]
        :return: None
        """

        if df is not None:
            self.df = df
        if do_split:
            self.k_split(k=k, groupby=groupby, sortby=sortby, random_state=random_state, do_print=do_print)
        if do_fit:
            self.fit(fit_type=fit_type, k_test=k_test, do_print=do_print)
        if do_predict:
            self.predict(ensemble=ensemble, do_print=do_print)
        if do_score:
            self.score(scores=scores, scale=scale, do_print=do_print, display_score=display_score)
        if do_print:
            self.printf('training done')

    @docstr_hpt
    def scoreplot(
            self, x='y_ref', y='value', hue='model', hue_order=None, row='score', row_order=None,
            palette=hpt_rcParams['palette'], width=16, height=9 / 2, scale=None, query=None, return_fig_ax=False,
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
            row_order = ['r2', 'rmse', 'mae', 'stdae', 'medae']
        if hue_order is None:
            hue_order = self.model_names
        _row_order = force_list(row_order)

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


# ---- functions
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
def force_model(model: Any) -> Model:
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
    _df['feature'] = force_list(y) + ['intercept']
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
#     )
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
