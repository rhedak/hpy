"""
hhpy.ds.py
~~~~~~~~~~

Contains DataScience functions extending on pandas and sklearn

"""
# ---- imports
# --- standard imports
import numpy as np
import pandas as pd
import warnings
import os

# --- third party imports
from copy import deepcopy
from scipy import stats, signal
from scipy.spatial import distance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.preprocessing import StandardScaler
from typing import Mapping, Sequence, Callable, Union, List, Optional, Tuple, Any
from io import StringIO
from datetime import datetime

# --- local imports
from hhpy.main import export, BaseClass, assert_list, tprint, progressbar, qformat, list_intersection, round_signif, \
    is_list_like, dict_list, append_to_dict_list, concat_cols, DocstringProcessor, reformat_string, dict_inv, \
    list_exclude, docstr as docstr_main, SequenceOfScalars, SequenceOrScalar, STRING_NAN, is_scalar, GROUPBY_DUMMY, \
    assert_scalar, list_merge

# ---- variables
# --- constants
ROW_DUMMY = '__row__'
# --- validations
validations = {
    'DFMapping__from_df__return_type': ['self', 'tuple'],
    'DFMapping__to_excel__if_exists': ['error', 'replace', 'append']
}
# --- docstr
docstr = DocstringProcessor(
    # - general
    df='Pandas DataFrame containing the data, other objects are implicitly cast to DataFrame',
    x='Main variable, name of a column in the DataFrame or vector data',
    hue='Name of the column to split by level [optional]',
    top_nr='Number of unique levels to keep when applying :func:`~top_n_coding` [optional]',
    other_name='Name of the levels grouped inside other [optional]',
    other_to_na='Whether to cast all other elements to NaN [optional]',
    inplace='Whether to modify the DataFrame inplace [optional]',
    printf='The function used for printing in-function messages. Set to None or False to suppress printing [optional]',
    groupby='The columns used for grouping, passed to pandas.DataFrame.groupby [optional]',
    window='Size of the rolling window, see pandas.Series.rolling [optional]',
    # - specific
    DFMapping__col_names='Whether to transform the column names [optional]',
    DFMapping__values='Whether to transform the column values [optional]',
    DFMapping__columns='Columns to transform, defaults to all columns [optional]',
    # - imported
    warn=docstr_main.params['warn'],
    # - validations
    **validations
)
# --- dtypes
dtypes = {
    'Int': ['Int8', 'Int16', 'Int32', 'Int64', 'UInt8', 'UInt16', 'UInt32', 'UInt64'],
    'UInt': ['UInt8', 'UInt16', 'UInt32', 'UInt64'],
    'int': ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'],
    'uint': ['uint8', 'uint16', 'uint32', 'uint64'],
    'float': ['float8', 'float16', 'float32', 'float64'],
    'string': ['string'],
    'object': ['object'],
    'boolean': ['boolean'],
    'category': ['category'],
    'datetime': ['datetime64[ns]'],
    'datetimez': ['datetime64[ns, <tz>]'],
    'period': ['period[<freq>]']
}
dtypes['Iint'] = dtypes['Int'] + dtypes['int']
dtypes['number'] = dtypes['Iint'] + dtypes['float']
dtypes['datetime64'] = dtypes['datetime']


# ---- classes
@export
class DFMapping(BaseClass):
    """
        Mapping object bound to a pandas DataFrame that standardizes column names and values according to the chosen
        conventions. Also implements google translation. Can be used like an sklearn scalar object.
        The mapping can be saved and later used to restore the original shape of the DataFrame.
        Note that the index is exempt.

        :param name: name of the object [Optional]
        :param df: a DataFrame to init on or path to a saved DFMapping object [Optional]
        :param kwargs: other arguments passed to the respective init function
    """

    # --- globals
    __name__ = 'DFMapping'
    __attributes__ = ['col_mapping', 'value_mapping']

    # --- functions
    def __init__(self, df: Union[pd.DataFrame, dict, str] = None, **kwargs) -> None:

        self.col_mapping = {}
        self.value_mapping = {}

        # -- defaults
        # - if the function is called with only one argument attempt to parse it's type and act accordingly
        # DataFrame is passed: init from it
        if isinstance(df, pd.DataFrame):
            self.from_df(df, **kwargs)
        # Dict is passed: init from it
        elif isinstance(df, dict):
            self.from_dict(df)
        # path to excel or pickle file is passed: init from it
        elif isinstance(df, str):
            if '.xlsx' in df:
                self.from_excel(df)
            else:
                self.from_pickle(df)

    @docstr
    def from_df(self, df: pd.DataFrame, col_names: bool = True, values: bool = True,
                columns: Optional[List[str]] = None, return_type: str = 'self', printf: Callable = tprint,
                duplicate_limit: int = 10, warn: bool = True, **kwargs) -> Optional[Tuple[dict, dict]]:
        """
        Initialize the DFMapping from a pandas DataFrame.

        :param df: %(df)s
        :param col_names: %(DFMapping__col_names)s
        :param values:  %(DFMapping__values)s
        :param columns: %(DFMapping__columns)s
        :param return_type: if 'self': writes to self, 'tuple' returns (col_mapping, value_mapping) [optional]
        :param printf: %(printf)s
        :param duplicate_limit: allowed number of reformated duplicates per column, each duplicate is suffixed with '_'
            but if you have too many you likely have a column of non allowed character strings and the mapping
            would take a very long time. The duplicate handling therefore stops and a warning is triggered
            since the transformation is no longer invertible. Consider excluding the column or using cat codes
            [optional]
        :param warn: %(warn)s
        :param kwargs: Other keyword arguments passed to :func:`~hhpy.main.reformat_string` [optional]
        :return: see return_type
        """

        # -- assert
        df = assert_df(df)

        # -- init
        # assert
        if return_type not in validations['DFMapping__from_df__return_type']:
            if warn:
                warnings.warn(f'Unknown return_type {return_type}, falling back to self')
            return_type = 'self'

        # -- main
        # extract columns
        if columns:
            _columns = columns
        else:
            _columns = df.columns

        # init mappings
        _col_mapping = {}
        _value_mapping = {}
        _str_columns = df.select_dtypes(['object', 'category']).columns

        # loop columns
        for _it, _column in enumerate(_columns):
            # progressbar
            if printf:
                progressbar(_it, len(_columns), printf=printf, print_prefix=f'{_column}: ')
            # map col name
            if col_names:
                _reformated_column = reformat_string(_column, **kwargs)
                # careful: it is possible that the reformated string is a duplicate, in this case we append '_' to the
                # string until it is no longer a duplicate
                _it_ = 0
                while _reformated_column in _col_mapping.values():
                    _reformated_column += '_'
                    _it_ += 1
                    if _it_ == duplicate_limit:
                        if warn:
                            warnings.warn(f'too many reformated duplicates in column names')
                        break
                # assign to dict
                _col_mapping[_column] = _reformated_column
            # check if column is string like
            if _column in _str_columns:
                # get unique values
                _uniques = df[_column].drop_duplicates().values
                # map
                if values:
                    _value_mapping[_column] = {}
                    _it_u_max = len(_uniques)
                    for _it_u, _unique in enumerate(_uniques):
                        # progressbar
                        if printf:
                            progressbar(_it, len(_columns), printf=printf,
                                        print_prefix=f'{_column}: {_it_u} / {_it_u_max}')
                        # reformat
                        _reformated_unique = reformat_string(_unique, **kwargs)
                        # careful: it is possible that the reformated string is a duplicate, in this case we
                        # append '_' to the string until it is no longer a duplicate
                        _it_ = 0
                        while _reformated_unique in _value_mapping[_column].values():
                            _reformated_unique += '_'
                            _it_ += 1
                            if _it_ == duplicate_limit:
                                if warn:
                                    warnings.warn(f'too many reformated duplicates in column {_column}')
                                break
                        # assign to dict
                        _value_mapping[_column][_unique] = _reformated_unique
        # progressbar 100%
        if printf:
            progressbar(printf=printf)

        if return_type == 'self':
            self.col_mapping = _col_mapping
            self.value_mapping = _value_mapping
        else:  # return_type == 'tuple'
            return self.col_mapping, self.value_mapping

    def fit(self, *args, **kwargs) -> Optional[Tuple[dict, dict]]:
        """
        Alias for :meth:`~DFMapping.from_df` to be inline with sklearn conventions

        :param args: passed to from_df
        :param kwargs: passed to from_df
        :return: see from_df
        """

    @docstr
    def transform(self, df: pd.DataFrame, col_names: bool = True, values: bool = True,
                  columns: Optional[List[str]] = None, inverse: bool = False,
                  inplace: bool = False) -> Optional[pd.DataFrame]:
        """
        Apply a mapping created using :func:`~create_df_mapping`. Intended to make a DataFrame standardized and
        human readable. The same mapping can also be applied with inverse=True to restore the original form
        of the transformed DataFrame.

        :param df: %(df)s
        :param col_names: %(DFMapping__col_names)s
        :param values:  %(DFMapping__values)s
        :param columns: %(DFMapping__columns)s
        :param inverse: Whether to apply the mapping in inverse order to restore the original form of the DataFrame
            [optional]
        :param inplace: %(inplace)s
        :return: if inplace: None, else: Transformed DataFrame
        """
        # -- init
        # handle inplace
        if not inplace:
            df = assert_df(df)
        # get helpers
        if col_names:
            _col_mapping = self.col_mapping
        else:
            _col_mapping = {}
        if values:
            _value_mapping = self.value_mapping
        else:
            _value_mapping = {}
        if columns:
            _columns = columns
        else:
            _columns = df.columns

        # -- main
        # if inverse: rename columns first
        if _col_mapping:
            if inverse:
                _col_mapping = dict_inv(_col_mapping, duplicates='drop')
                df.columns = [_col_mapping.get(_, _) for _ in _columns]
            else:
                _columns = [_col_mapping.get(_, _) for _ in _columns]

        # replace values
        for _key, _mapping in _value_mapping.items():

            # if applicable: inverse mapping
            if inverse:
                _mapping = dict_inv(_mapping, duplicates='drop')
            # replace column values
            df[_key] = df[_key].replace(_mapping)

        # if not inverse: rename columns last
        if not inverse:
            df.columns = _columns

        # -- return
        if inplace:
            # noinspection PyProtectedMember
            df._update_inplace(df)
        else:
            return df

    def inverse_transform(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """
        wrapper for :meth:`DFMapping.transform` with inverse=True

        :param args: passed to transform
        :param kwargs: passed to transform
        :return: see transform
        """

        return self.transform(*args, inverse=True, **kwargs)

    @docstr
    def fit_transform(self, df: pd.DataFrame, col_names: bool = True, values: bool = True,
                      columns: Optional[List[str]] = None, kwargs_fit: Mapping = None,
                      **kwargs) -> Optional[pd.DataFrame]:
        """
        First applies :meth:`DFMapping.from_df` (which has alias fit) and then :meth:`DFMapping.transform`

        :param df: pandas DataFrame to fit against and then transform.
        :param col_names: %(DFMapping__col_names)s
        :param values:  %(DFMapping__values)s
        :param columns: %(DFMapping__columns)s
        :param kwargs: passed to transform
        :param kwargs_fit: passed to fit
        :return: see transform
        """
        if kwargs_fit is None:
            kwargs_fit = {}
        self.fit(df=df, col_names=col_names, values=values, columns=columns, **kwargs_fit)
        return self.transform(df=df, col_names=col_names, values=values, columns=columns, **kwargs)

    def to_excel(self, path: str, if_exists: str = 'error') -> None:
        """
        Save the DFMapping object as an excel file. Useful if you want to edit the results of the automatically
        generated object to fit your specific needs.

        :param path: Path to save the excel file to
        :param if_exists: One of %(DFMapping__to_excel__if_exists)s, if 'error' raises exception, if 'replace' replaces
            existing files and if 'append' appends to file (while checking for duplicates)
        :return: None
        """
        # -- functions
        def _write_excel_sheet(writer, mapping, sheet_name):
            # create DataFrame and transpose
            _df_mapping = pd.DataFrame(mapping, index=[0]).T
            # handle append
            if (if_exists == 'append') and (sheet_name in _sheet_names):
                # new mapping data comes below existing ones, duplicates are dropped (keep old)
                _df_mapping = pd.read_excel(path, sheet_name, index_col=0).append(_df_mapping)\
                    .pipe(drop_duplicate_indices)
            # write excel
            _df_mapping.to_excel(writer, sheet_name=sheet_name)
        # -- init
        # - assert
        if if_exists not in validations['DFMapping__to_excel__if_exists']:
            raise ValueError(f"if_exists must be one of {validations['DFMapping__to_excel__if_exists']}")
        # - handle if_exists
        _sheet_names = []
        if os.path.exists(path):
            if if_exists == 'error':
                raise FileExistsError(f"file already exists, please specify if_exists as one of ")
            elif if_exists == 'append':
                _sheet_names = pd.ExcelFile(path).sheet_names
        # -- main
        # pandas ExcelWriter object (saves on close)
        with pd.ExcelWriter(path) as _writer:
            # col mapping
            _write_excel_sheet(writer=_writer, mapping=self.col_mapping, sheet_name='__columns__')
            # value mappings
            for _key, _mapping in self.value_mapping.items():
                _write_excel_sheet(writer=_writer, mapping=_mapping, sheet_name=_key)

    def from_excel(self, path: str) -> None:
        """
        Init the DFMapping object from an excel file. For example you could auto generate a DFMapping using googletrans
        and then adjust the translations you feel are inappropriate in the excel file. Then regenerate the object
        from the edited excel file.

        :param path: Path to the excel file
        :return: None
        """

        def _read_excel(xls, sheet_name):
            return pd.read_excel(xls, sheet_name, index_col=0).T.to_dict(orient='records')[0]

        # open ExcelFile
        with pd.ExcelFile(path) as _xls:
            self.col_mapping = _read_excel(xls=_xls, sheet_name='__columns__')
            self.value_mapping = {}
            for _sheet_name in list_exclude(_xls.sheet_names, '__columns__'):
                self.value_mapping[_sheet_name] = _read_excel(xls=_xls, sheet_name=_sheet_name)


# ---- functions
# --- export
@export
def assert_df(df: Any, groupby: Union[SequenceOrScalar, bool] = False, name: str = 'df',
              ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List]]:
    """
    assert that input is a pandas DataFrame, raise ValueError if it cannot be cast to DataFrame

    :param df: Object to be cast to DataFrame
    :param groupby: column to use as groupby
    :param name: name to use in the ValueError message, useful when calling from another function
    :return: pandas DataFrame
    """

    try:
        df = pd.DataFrame(df).copy()
    except Exception as _e:
        print(f"{_e.__class__.__name__}: {_e}")
        raise ValueError(f"{name} must be a DataFrame or castable to DataFrame")

    if isinstance(groupby, bool) and not groupby:
        return df
    elif groupby is None or groupby in [[], GROUPBY_DUMMY, [GROUPBY_DUMMY]]:
        groupby = [GROUPBY_DUMMY]
        df[GROUPBY_DUMMY] = 1
    else:
        groupby = assert_list(groupby)

    # drop duplicate columns
    df = drop_duplicate_cols(df)

    return df, groupby


@export
def optimize_pd(df: pd.DataFrame, c_int: bool = True, c_float: bool = True, c_cat: bool = True, cat_frac: float = .5,
                convert_dtypes: bool = True, drop_all_na_cols: bool = False) -> pd.DataFrame:
    """
    optimize memory usage of a pandas df, automatically downcast all var types and converts objects to categories

    :param df: pandas DataFrame to be optimized. Other objects are implicitly cast to DataFrame
    :param c_int: Whether to downcast integers [optional]
    :param c_float: Whether to downcast floats [optional]
    :param c_cat: Whether to cast objects to categories. Uses cat_frac as condition [optional]
    :param cat_frac: If c_cat: If the column has less than cat_frac percent unique values it will be cast to category
        [optional]
    :param convert_dtypes: Whether to call convert dtypes (pandas 1.0.0+) [optional]
    :param drop_all_na_cols: Whether to drop columns that contain only missing values [optional]
    :return: the optimized pandas DataFrame
    """
    # -- func
    # noinspection PyShadowingNames
    def _do_downcast(df, cols, downcast):
        if downcast is None:
            return df
        for _col in assert_list(cols):
            # downcast
            try:
                df[_col] = pd.to_numeric(df[_col], downcast=downcast)
            except Exception as _e:
                print(f"Downcast Error in {_col} - {_e.__class__}: {_e}")
        return df

    # -- init
    # avoid inplace operations
    df = assert_df(df)

    # pandas version flag
    _pandas_version_1_plus = int(pd.__version__.split('.')[0]) > 0
    # not convert_dtypes not support before pandas 1.0.0
    if not _pandas_version_1_plus:
        convert_dtypes = False

    # check for duplicate columns
    _duplicate_columns = get_duplicate_cols(df)
    if len(_duplicate_columns) > 0:
        warnings.warn('duplicate columns found: {}'.format(_duplicate_columns))
        df = drop_duplicate_cols(df)

    # if applicable: drop columns containing only na
    if drop_all_na_cols:
        df = df.drop(df.columns[df.isnull().all()], axis=1)

    # -- main
    # if applicable: convert float columns containing integer values to dtype int
    if convert_dtypes:
        # scalar object to str (doesn't seem to work automatically as of 1.0.0)
        for _col in df.select_dtypes('object').columns:
            df[_col] = df[_col].apply(lambda _: str(_) if is_scalar(_) else _)
        # df.convert_dtypes will be called after downcasting since it is not supported for some dtypes

    # casting
    if c_int:
        _include = dtypes['int']
        # Int does not support downcasting as of pandas 1.0.0 -> check again later
        # if _pandas_version_1_plus:
        #     _include += dtypes_Int
        _cols_int = df.select_dtypes(include=_include)
        # loop int columns
        for _col in _cols_int:
            # split integer columns in unsigned (all positive) and (unsigned)
            if df[_col].isna().sum() > 0:
                _downcast = None
            elif (df[_col] > 0).all():
                _downcast = 'unsigned'
            else:
                _downcast = 'signed'
            df = _do_downcast(df=df, cols=_col, downcast=_downcast)

    if c_float:
        df = _do_downcast(df=df, cols=df.select_dtypes(include=['float']).columns, downcast='float')

    if c_cat:
        _include = ['object']
        if _pandas_version_1_plus:
            _include += ['string']
        for _col in df.select_dtypes(include=_include).columns:
            # if there are less than 1 - cat_frac unique elements: cast to category
            _count_unique = df[_col].dropna().drop_duplicates().shape[0]
            _count_no_na = df[_col].dropna().shape[0]
            if _count_no_na > 0 and (_count_unique / _count_no_na < (1 - cat_frac)):
                df[_col] = df[_col].astype('category')

    # call convert dtypes to handle downcasted dtypes
    if convert_dtypes:
        # try except is needed due to some compatibility issues
        try:
            df = df.convert_dtypes()
        except Exception as _e:
            print(f"skipped convert_dtypes due to: f{_e.__class__}: {_e}")

    return df


@export
def get_df_corr(df: pd.DataFrame, columns: List[str] = None, target: str = None,
                groupby: Union[str, list] = None) -> pd.DataFrame:
    """
    Calculate Pearson Correlations for numeric columns, extends on pandas.DataFrame.corr but automatically
    melts the output. Used by :func:`~hhpy.plotting.corrplot_bar`

    :param df: input pandas DataFrame. Other objects are implicitly cast to DataFrame
    :param columns: Column to calculate the correlation for, defaults to all numeric columns [optional]
    :param target: Returns only correlations that involve the target column [optional]
    :param groupby: Returns correlations for each level of the group [optional]
    :return: pandas DataFrame containing all pearson correlations in a melted format
    """
    # -- assert
    # df / groupby
    df, groupby = assert_df(df=df, groupby=groupby)
    # -- init
    # if there is a column called index it will create problems so rename it to '__index__'
    df = df.rename({'index': '__index__'}, axis=1)
    # columns defaults to numeric columns
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns

    # -- main
    # init df as list of dfs
    _df_corr = []
    # loop groups
    for _index, _df_i in df.groupby(groupby):
        # get corr
        _df_corr_i = _df_i.corr().reset_index().rename({'index': 'col_0'}, axis=1)
        # set upper right half to nan
        for _i, _col in enumerate(columns):
            _df_corr_i[_col] = np.where(_df_corr_i[_col].index <= _i, np.nan, _df_corr_i[_col])
        # gather / melt
        _df_corr_i = pd.melt(_df_corr_i, id_vars=['col_0'], var_name='col_1', value_name='corr').dropna()
        # drop self correlation
        _df_corr_i = _df_corr_i[_df_corr_i['col_0'] != _df_corr_i['col_1']]
        # get identifier
        for _groupby in groupby:
            _df_corr_i[_groupby] = _df_i[_groupby].iloc[0]
        # append to list of dfs
        _df_corr.append(_df_corr_i)
    # merge
    _df_corr = concat(_df_corr)

    # clean dummy groupby
    if GROUPBY_DUMMY in _df_corr.columns:
        _df_corr.drop(GROUPBY_DUMMY, axis=1, inplace=True)
    else:
        # move groupby columns to front
        _df_corr = col_to_front(_df_corr, groupby)

    # reorder and keep only columns involving the target (if applicable)
    if target is not None:
        # if the target is col_1: switch it to col_0
        _target_is_col_1 = (_df_corr['col_1'] == target)
        _df_corr['col_1'] = np.where(_target_is_col_1, _df_corr['col_0'], _df_corr['col_1'])
        _df_corr['col_0'] = np.where(_target_is_col_1, target, _df_corr['col_0'])
        # keep only target in col_0
        _df_corr = _df_corr[_df_corr['col_0'] == target]

    # get absolute correlation
    _df_corr['corr_abs'] = np.abs(_df_corr['corr'])
    # sort descending
    _df_corr = _df_corr.sort_values(['corr_abs'], ascending=False).reset_index(drop=True)

    return _df_corr


@export
def drop_zero_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns with all 0 or None Values from DataFrame. Useful after applying one hot encoding.

    :param df: pandas DataFrame
    :return: pandas DataFrame without 0 columns.
    """
    # noinspection PyUnresolvedReferences
    return df[df.columns[(df != 0).any()]]


@export
def get_duplicate_indices(df: pd.DataFrame) -> pd.Series:
    """
    Returns duplicate indices from a pandas DataFrame

    :param df: pandas DataFrame
    :return: List of indices that are duplicate
    """
    return df.index[df.index.duplicated()]


@export
def get_duplicate_cols(df: pd.DataFrame) -> pd.Series:
    """
    Returns names of duplicate columns from a pandas DataFrame

    :param df: pandas DataFrame
    :return: List of column names that are duplicate
    """
    return df.columns[df.columns.duplicated()]


@export
def drop_duplicate_indices(df: pd.DataFrame, warn: bool = True) -> pd.DataFrame:
    """
    Drop duplicate indices from pandas DataFrame

    :param df: pandas DataFrame
    :param warn: Whether to trigger a warning if duplicate indices are dropped
    :return: pandas DataFrame without the duplicates indices
    """
    if warn:
        _duplicate_indices = get_duplicate_indices(df).tolist()
        if _duplicate_indices:
            print(f"Dropping duplicate indices: {_duplicate_indices}")
    return df.loc[~df.index.duplicated(), :]


@export
def drop_duplicate_cols(df: pd.DataFrame, warn: bool = True) -> pd.DataFrame:
    """
    Drop duplicate columns from pandas DataFrame

    :param df: pandas DataFrame
    :param warn: Whether to trigger a warning if duplicate columns are dropped
    :return: pandas DataFrame without the duplicates columns
    """
    if warn:
        _duplicate_cols = get_duplicate_cols(df).tolist()
        if _duplicate_cols:
            warnings.warn(f"Dropping duplicate columns: {_duplicate_cols}")
    return df.loc[:, ~df.columns.duplicated()]


@export
def change_span(s: pd.Series, steps: int = 5) -> pd.Series:
    """
    return a True/False series around a changepoint, used for filtering stepwise data series in a pandas df
    must be properly sorted!

    :param s: pandas Series or similar
    :param steps: number of steps around the changepoint to flag as true
    :return: pandas Series of dtype Boolean
    """
    return pd.Series(s.shift(-steps).ffill() != s.shift(steps).bfill())


@export
def outlier_to_nan(df: pd.DataFrame, col: str, groupby: Union[list, str] = None, std_cutoff: np.number = 3,
                   reps: int = 1, do_print: bool = False) -> pd.DataFrame:
    """
    this algorithm cuts off all points whose DELTA (avg diff to the prev and next point) is outside of the n std range

    :param df: pandas DataFrame
    :param col: column to be filtered
    :param groupby: if provided: applies std filter by group
    :param std_cutoff: the number of standard deviations outside of which to set values to None
    :param reps: how many times to repeat the algorithm
    :param do_print: whether to print steps to console
    :return: pandas Series with outliers set to nan
    """
    df, groupby = assert_df(df=df, groupby=groupby)

    for _rep in range(reps):

        if do_print:
            tprint('rep = ' + str(_rep + 1) + ' of ' + str(reps))

        # grouped by df
        _df_grouped = df.groupby(groupby)

        # use interpolation to treat missing values
        df[col] = _df_grouped[col].transform(pd.DataFrame.interpolate)

        # calculate delta (mean of diff to previous and next value)
        _delta = .5 * (
                (df[col] - _df_grouped[col].shift(1).bfill()).abs() +
                (df[col] - _df_grouped[col].shift(-1).ffill()).abs()
        )

        df[col] = df[col].where((_delta - _df_grouped[col].mean()).abs() <= (std_cutoff * _df_grouped[col].std()))

    if GROUPBY_DUMMY in df.columns:
        df = df.drop(GROUPBY_DUMMY, axis=1)

    return df[col]


@export
def butter_pass_filter(data: pd.Series, cutoff: int, fs: int, order: int, btype: str = None, shift: bool = False):
    """
    Implementation of a highpass / lowpass filter using scipy.signal.butter

    :param data: pandas Series or 1d numpy Array
    :param cutoff: cutoff
    :param fs: critical frequencies
    :param order: order of the fit
    :param btype: The type of filter. Passed to scipy.signal.butter.  Default is ‘lowpass’.
        One of {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
    :param shift: whether to shift the data to start at 0
    :return: 1d numpy array containing the filtered data
    """

    def _f_butter_pass(_f_cutoff, _f_fs, _f_order, _f_btype):
        _nyq = 0.5 * _f_fs
        _normal_cutoff = _f_cutoff / _nyq
        # noinspection PyTupleAssignmentBalance
        __b, __a = signal.butter(_f_order, _normal_cutoff, btype=_f_btype, analog=False, output='ba')

        return __b, __a

    _data = np.array(data)

    if shift:
        _shift = pd.Series(data).iloc[0]
    else:
        _shift = 0

    _data -= _shift

    _b, _a = _f_butter_pass(_f_cutoff=cutoff, _f_fs=fs, _f_order=order, _f_btype=btype)

    _y = signal.lfilter(_b, _a, _data)

    _y = _y + _shift

    return _y


@export
def pass_by_group(df: pd.DataFrame, col: str, groupby: Union[str, list], btype: str, shift: bool = False,
                  cutoff: int = 1, fs: int = 20, order: int = 5):
    """
    allows applying a butter_pass filter by group

    :param df: pandas DataFrame
    :param col: column to filter
    :param groupby: columns to groupby
    :param btype: The type of filter. Passed to scipy.signal.butter.  Default is ‘lowpass’.
        One of {‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
    :param shift: shift: whether to shift the data to start at 0
    :param cutoff: cutoff
    :param fs: critical frequencies
    :param order: order of the filter
    :return: filtered DataFrame
    """
    df = assert_df(df)

    _df_out_grouped = df.groupby(groupby)

    # apply highpass filter
    df[col] = np.concatenate(
        _df_out_grouped[col].apply(butter_pass_filter, cutoff, fs, order, btype, shift).values).flatten()

    df = df.reset_index(drop=True)

    return df


@export
def lfit(x: SequenceOrScalar, y: SequenceOrScalar = None, w: SequenceOrScalar = None, df: pd.DataFrame = None,
         groupby: SequenceOrScalar = None, do_print: bool = True, catch_error: bool = False, return_df: bool = False,
         extrapolate: int = None) -> Union[pd.Series, pd.DataFrame]:
    """
    quick linear fit with numpy

    :param x: names of x variables in df or vector data, if y is None treated as target and fit against the index
    :param y: names of y variables in df or vector data [optional]
    :param w: names of weight variables in df or vector data [optional]
    :param df: pandas DataFrame containing x,y,w data [optional]
    :param groupby: If specified the linear fit is applied by group [optional]
    :param do_print: whether to print steps to console
    :param catch_error: whether to keep going in case of error [optional]
    :param return_df: whether to return a DataFrame or Series [optional]
    :param extrapolate: how many iteration to extrapolate [optional]
    :return: if return_df is True: pandas DataFrame, else: pandas Series
    """
    if df is None:
        if hasattr(x, 'name'):
            _x_name = x.name
        else:
            _x_name = 'x'
        if hasattr(y, 'name'):
            _y_name = y.name
        else:
            _y_name = 'y'
        if hasattr(w, 'name'):
            _w_name = w.name
        else:
            _w_name = 'w'
        _df = pd.DataFrame()
        _df[_x_name] = x
        _df[_y_name] = y
        _df[_w_name] = w
    else:
        _df = df.copy()
        del df
        _x_name = x
        _y_name = y
        _w_name = w
    _y_name_fit = f"{_y_name}_fit"

    if groupby is None:
        groupby = [GROUPBY_DUMMY]
        _df[GROUPBY_DUMMY] = 1
    groupby = assert_list(groupby)

    _it_max = _df[groupby].drop_duplicates().shape[0]

    _df_fit = []

    for _it, (_index, _df_i) in enumerate(_df.groupby(groupby)):

        if do_print and _it_max > 1:
            progressbar(_it, _it_max, print_prefix=qformat(_index))

        if y is None:
            _x = _df_i.index.to_series()
            _y = _df_i[_x_name]
        else:
            _x = _df_i[_x_name]
            _y = _df_i[_y_name]
        if w is not None:
            _w = _df_i[_w_name]
            _w = _w.astype(float)
        else:
            _w = None

        _x = _x.astype(float)
        _y = _y.astype(float)

        _idx = np.isfinite(_x) & np.isfinite(_y)

        if _w is not None:
            _w_idx = _w[_idx]
        else:
            _w_idx = None

        if catch_error:
            try:
                _fit = np.poly1d(np.polyfit(x=_x[_idx], y=_y[_idx], deg=1, w=_w_idx))
            except Exception as _exc:
                warnings.warn('handled exception: {}'.format(_exc))
                _fit = None
        else:
            _fit = np.poly1d(np.polyfit(x=_x[_idx], y=_y[_idx], deg=1, w=_w_idx))

        _x_diff = _x.diff().mean()
        _x = list(_x)
        _y = list(_y)

        if _fit is None:
            _y_fit = _y
        else:

            if extrapolate is not None:

                for _ext in range(extrapolate):
                    _x.append(np.max(_x) + _x_diff)
                    _y.append(np.nan)

            _y_fit = _fit(_x)

        # create df fit for iteration
        _df_fit_i = pd.DataFrame({
            _x_name: _x,
            _y_name: _y,
            _y_name_fit: _y_fit
        })

        _df_fit.append(_df_fit_i)

    _df_fit = concat(_df_fit)

    if do_print and _it_max > 1:
        progressbar()

    if return_df:
        return _df_fit
    else:
        return _df_fit[_y_name_fit]


@docstr
@export
def rolling_lfit(x: SequenceOrScalar, window: int, df: pd.DataFrame = None, groupby: SequenceOrScalar = None):
    """
    Rolling version of lfit: for each row of the DataFrame / Series look at the previous window rows, then perform an
    lfit and use this value as a prediction for this row. Useful as naive predictor for time series Data.

    :param x: %(x)s
    :param window: %(window)s
    :param df: %(df)s
    :param groupby:%(groupby)s
    :return: pandas Series containing the fitted values
    """

    # -- assert
    if df is None:
        if hasattr(x, 'name'):
            _x_name = x
        else:
            _x_name = 'x'
        df = pd.DataFrame({_x_name: x})
    else:
        _x_name = x

    # -- init
    if groupby is None:
        groupby = [GROUPBY_DUMMY]
        df[GROUPBY_DUMMY] = 1
    else:
        groupby = assert_list(groupby)

    # -- main
    # init output as dict
    _x_lfit = {}
    # - loop groups
    for _, _df_i in df.groupby(groupby):
        # get _x_i
        _x_i = _df_i[x]
        for _row, (_x_index, __) in enumerate(_x_i.iteritems()):
            # need at least 2 entries to lfit -> first two entries become na
            if _row < 2:
                _x_lfit[_x_index] = np.nan
                continue
            # if row < window start at 0
            _min_row = max([_row - window, 0])
            # subset series
            _x_row = _x_i.iloc[_min_row:_row]
            # fit
            _x_row_lfit = lfit(_x_row, extrapolate=1)
            # get extrapolated value and append to dict
            _x_lfit[_x_index] = (_x_row_lfit.iloc[-1])
    # dict to series
    _x_lfit = pd.Series(_x_lfit).sort_index()
    # -- return
    return _x_lfit


@export
def qf(df: pd.DataFrame, fltr: Union[pd.DataFrame, pd.Series, Mapping], rem_unused_categories: bool = True,
       reset_index: bool = False):
    """
    quickly filter a DataFrame based on equal criteria. All columns of fltr present in df are filtered
    to be equal to the first entry in filter_df.

    :param df: pandas DataFrame to be filtered
    :param fltr: filter condition as DataFrame or Mapping or Series
    :param rem_unused_categories: whether to remove unused categories from categorical dtype after filtering
    :param reset_index: whether to reset index after filtering
    :return: filtered pandas DataFrame
    """
    _df = df.copy()
    del df

    # filter_df can also be a dictionary, in which case pd.DataFrame.from_dict will be applied
    if isinstance(fltr, Mapping):
        _filter_df = pd.DataFrame(fltr, index=[0])
    # if the filter_df is a series, attempt to cast to data frame
    elif isinstance(fltr, pd.Series):
        _filter_df = pd.DataFrame(fltr).T
    # assume it to be a DataFrame
    else:
        _filter_df = fltr.copy()
        del fltr

    # drop columns not in
    _filter_df = _filter_df[list_intersection(_filter_df.columns, _df.columns)]

    # init filter
    _filter_iloc = _filter_df.iloc[0]

    # create a dummy boolean of all trues with len of df
    _filter_condition = (_df.index == _df.index)

    # logical and filter for all columns in filter df
    for _col in _filter_df.columns:
        _filter_condition = _filter_condition & (_df[_col] == _filter_iloc[_col])

    # create filtered df
    _df = _df[_filter_condition]

    # remove_unused_categories
    if rem_unused_categories:
        _df = remove_unused_categories(_df)

    if reset_index:
        _df = _df.reset_index(drop=True)

    # return
    return _df


@export
def quantile_split(s: pd.Series, n: int, signif: int = 2, na_to_med: bool = False):
    """
    splits a numerical column into n quantiles. Useful for mapping numerical columns to categorical columns

    :param s: pandas Series to be split
    :param n: number of quantiles to split into
    :param signif: number of significant digits to round to
    :param na_to_med: whether to fill na values with median values
    :return: pandas Series of dtype category
    """
    if len(s.unique()) <= n:
        return s

    _s = pd.Series(s).astype(float)
    _s = np.where(~np.isfinite(_s), np.nan, _s)
    _s = pd.Series(_s)

    _s_out = _s.apply(lambda _: np.nan)

    if na_to_med:
        _s = _s.fillna(_s.median())

    if signif is not None:
        _s = round_signif(_s, signif)

    if not isinstance(_s, pd.Series):
        _s = pd.Series(_s)

    _i = -1

    for _q in np.arange(0, 1, 1. / n):

        _i += 1

        __q_min = np.quantile(_s.dropna().values, _q)

        if _q + .1 >= 1:
            __q_max = _s.max()
        else:
            __q_max = np.quantile(_s.dropna().values, _q + .1)

        if np.round(_q + .1, 1) == 1.:
            __q_max_adj = np.inf
            _right_equal_sign = '<='
        else:
            __q_max_adj = __q_max
            _right_equal_sign = '<'

        _q_name = 'q{}: {}<=_{}{}'.format(_i, round_signif(__q_min, signif), _right_equal_sign,
                                          round_signif(__q_max, signif))

        _s_out = np.where((_s >= __q_min) & (_s < __q_max_adj), _q_name, _s_out)

    # get back the old properties of the series (or you'll screw the index)
    _s_out = pd.Series(_s_out)
    _s_out.name = s.name
    _s_out.index = s.index

    # convert to cat
    _s_out = _s_out.astype('category')

    return _s_out


@export
def acc(y_true: Union[pd.Series, str], y_pred: Union[pd.Series, str], df: pd.DataFrame = None) -> float:
    """
    calculate accuracy for a categorical label

    :param y_true: true values as name of df or vector data
    :param y_pred: predicted values as name of df or vector data
    :param df: pandas DataFrame containing true and predicted values [optional]
    :return: accuracy a percentage
    """
    if df is None:

        _y_true = y_true
        _y_pred = y_pred

    else:

        _y_true = df[y_true]
        _y_pred = df[y_pred]

    _acc = np.sum(_y_true == _y_pred) / len(_y_true)
    return _acc


@export
def rel_acc(y_true: Union[pd.Series, str], y_pred: Union[pd.Series, str], df: pd.DataFrame = None,
            target_class: str = None):
    """
    relative accuracy of the prediction in comparison to predicting everything as the most common group
    :param y_true: true values as name of df or vector data
    :param y_pred: predicted values as name of df or vector data
    :param df: pandas DataFrame containing true and predicted values [optional]
    :param target_class: name of the target class, by default the most common one is used [optional]
    :return: accuracy difference as percent
    """
    if df is None:

        _y_true = 'y_true'
        _y_pred = 'y_pred'

        _df = pd.DataFrame({
            _y_true: y_true,
            _y_pred: y_pred
        })

    else:

        _df = df.copy()

        _y_true = y_true
        _y_pred = y_pred

        del df, y_true, y_pred

    if target_class is None:
        # get acc of pred
        _acc = acc(_y_true, _y_pred, df=_df)
        # get percentage of most common value
        _acc_mc = _df[_y_true].value_counts()[0] / _df.shape[0]
    else:
        _df_target_class = _df.query('{}=="{}"'.format(_y_true, target_class))
        # get acc of pred for target class
        _acc = acc(_y_true, _y_pred, df=_df_target_class)
        # get percentage of target class
        _acc_mc = _df_target_class.shape[0] / _df.shape[0]

    # rel acc is diff of both
    return _acc - _acc_mc


@export
def cm(y_true: Union[pd.Series, str], y_pred: Union[pd.Series, str], df: pd.DataFrame = None) -> pd.DataFrame:
    """
    confusion matrix from pandas df
    :param y_true: true values as name of df or vector data
    :param y_pred: predicted values as name of df or vector data
    :param df: pandas DataFrame containing true and predicted values [optional]
    :return: Confusion matrix as pandas DataFrame
    """
    if df is None:

        _y_true = deepcopy(y_true)
        _y_pred = deepcopy(y_pred)

        if 'name' in dir(y_true):
            y_true = y_true.name
        else:
            y_true = 'y_true'
        if 'name' in dir(y_pred):
            y_pred = y_pred.name
        else:
            y_true = 'y_pred'
        df = pd.DataFrame({
            y_true: _y_true,
            y_pred: _y_pred
        })
    else:
        _y_true = df[y_true]
        _y_pred = df[y_pred]

    _cm = df.eval('_count=1').groupby([y_true, y_pred]).agg({'_count': 'count'}).reset_index() \
        .pivot_table(index=y_true, columns=y_pred, values='_count')
    _cm = _cm.fillna(0).astype(int)

    return _cm


@export
def f1_pr(y_true: Union[pd.Series, str], y_pred: Union[pd.Series, str], df: pd.DataFrame = None, target: str = None,
          factor: int = 100) -> pd.DataFrame:
    """
    get f1 score, true positive, true negative, missed positive and missed negative rate

    :param y_true: true values as name of df or vector data
    :param y_pred: predicted values as name of df or vector data
    :param df: pandas DataFrame containing true and predicted values [optional]
    :param target: level for which to return the rates, by default all levels are returned [optional]
    :param factor: factor by which to scale results, default 100 [optional]
    :return: pandas DataFrame containing f1 score, true positive, true negative, missed positive
        and missed negative rate
    """
    if df is None:

        _y_true = deepcopy(y_true)
        _y_pred = deepcopy(y_pred)

        if 'name' in dir(y_true):
            y_true = y_true.name
        else:
            y_true = 'y_true'
        if 'name' in dir(y_pred):
            y_pred = y_pred.name
        else:
            y_true = 'y_pred'
        df = pd.DataFrame({
            y_true: _y_true,
            y_pred: _y_pred
        })
    else:
        _y_true = df[y_true]
        _y_pred = df[y_pred]

    _cm = cm(y_true=y_true, y_pred=y_pred, df=df)

    if target is None:
        target = _cm.index.tolist()
    elif not is_list_like(target):
        target = [target]

    _f1_pr = []

    _tp_sum = 0
    _tn_sum = 0
    _mp_sum = 0
    _mn_sum = 0
    _count_true_sum = 0

    for _target in target:

        if _target in _cm.index:
            _count_true = _cm.loc[_target].sum()
        else:
            _count_true = 0

        _count_true_sum += _count_true

        if _target in _cm.columns:
            _count_pred = _cm[_target].sum()
        else:
            _count_pred = 0

        _perc_pred = _count_pred / _count_true * factor

        # true positive: out of predicted as target how many are actually target
        try:
            _tp_i = _cm[_target][_target]
            _tp_sum += _tp_i
        except ValueError:
            _tp_i = np.nan
        # false positive: out of predicted as not target how many are actually not target
        try:
            _tn_i = _cm.drop(_target, axis=1).drop(_target, axis=0).sum().sum()
            _tn_sum += _tn_i
        except ValueError:
            _tn_i = np.nan

        # missed positive: out of true target how many were predicted as not target
        try:
            _mp_i = _cm.drop(_target, axis=1).loc[_target].sum()
            _mp_sum += _mp_i
        except ValueError:
            _mp_i = np.nan
        # missed negative: out of true not target how many were predicted as target
        try:
            _mn_i = _cm.drop(_target, axis=0)[_target].sum()
            _mn_sum += _mn_i
        except ValueError:
            _mn_i = np.nan

        # precision
        try:
            _precision = _tp_i / (_tp_i + _mn_i) * 100
        except ValueError:
            _precision = np.nan

        # recall
        try:
            _recall = _tp_i / (_tp_i + _mp_i) * 100
        except ValueError:
            _recall = np.nan

        if np.isnan(_precision) or np.isnan(_recall):
            _f1 = np.nan
        else:
            _f1 = 200 * (_precision / 100. * _recall / 100.) / (_precision / 100. + _recall / 100.)

        # to df
        _cm_target = pd.DataFrame({
            y_true: [_target], 'count': [_count_true], 'F1': [_f1], 'precision': [_precision], 'recall': [_recall]
        }).copy()

        _f1_pr.append(_cm_target)

    _f1_pr = pd.concat(_f1_pr, ignore_index=True, sort=False).set_index(y_true)

    return _f1_pr


@export
def f_score(y_true: Union[pd.Series, str], y_pred: Union[pd.Series, str], df: pd.DataFrame = None, dropna: bool = True,
            f: Callable = r2_score, groupby: Union[list, str] = None, f_name: str = None) -> Union[pd.DataFrame, float]:
    """
    generic scoring function base on pandas DataFrame.

    :param y_true: true values as name of df or vector data
    :param y_pred: predicted values as name of df or vector data
    :param df: pandas DataFrame containing true and predicted values [optional]
    :param dropna: whether to dropna values [optional]
    :param f: scoreing function to apply, default is sklearn.metrics.r2_score, should return a scalar value. [optional]
    :param groupby: if supplied then the result is returned for each group level [optional]
    :param f_name: name of the scoreing function, by default uses .__name__ property of function [optional]
    :return: if groupby is supplied: pandas DataFrame, else: scalar value
    """
    if df is None:

        _df = pd.DataFrame()

        _y_true = 'y_true'
        _y_pred = 'y_pred'
        _df[_y_true] = y_true
        _df[_y_pred] = y_pred

    else:

        _y_true = assert_scalar(y_true)
        _y_pred = assert_scalar(y_pred)

        _df = df.copy()
        del df

    if dropna:
        _df = _df.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna(subset=[_y_true, _y_pred])
        if groupby is not None:
            _df = _df.dropna(subset=groupby)
    if _df.shape[0] == 0:
        return np.nan

    if groupby is None:

        return f(_df[_y_true], _df[_y_pred])

    else:

        _df_out = []

        for _i, _df_group in _df.groupby(groupby):

            _df_i = _df_group[assert_list(groupby)].head(1)
            if f_name is None:
                f_name = f.__name__
            _df_i[f_name] = f(_df_group[_y_true], _df_group[_y_pred])
            _df_out.append(_df_i)

        _df_out = concat(_df_out)

        return _df_out


# shorthand r2
@export
def r2(*args, **kwargs) -> Union[pd.DataFrame, float]:
    """
    wrapper for f_score using sklearn.metrics.r2_score

    :param args: passed to f_score
    :param kwargs: passed to f_score
    :return: if groupby is supplied: pandas DataFrame, else: scalar value
    """
    return f_score(*args, f=r2_score, **kwargs)


@export
def rmse(*args, **kwargs) -> Union[pd.DataFrame, float]:
    """
    wrapper for f_score using numpy.sqrt(skearn.metrics.mean_squared_error)

    :param args: passed to f_score
    :param kwargs: passed to f_score
    :return: if groupby is supplied: pandas DataFrame, else: scalar value
    """

    def _f_rmse(x, y):
        return np.sqrt(mean_squared_error(x, y))

    return f_score(*args, f=_f_rmse, **kwargs)


@export
def mae(*args, **kwargs) -> Union[pd.DataFrame, float]:
    """
    wrapper for f_score using skearn.metrics.mean_absolute_error

    :param args: passed to f_score
    :param kwargs: passed to f_score
    :return: if groupby is supplied: pandas DataFrame, else: scalar value
    """
    return f_score(*args, f=mean_absolute_error, **kwargs)


@export
def stdae(*args, **kwargs) -> Union[pd.DataFrame, float]:
    """
    wrapper for f_score using the standard deviation of the absolute error

    :param args: passed to f_score
    :param kwargs: passed to f_score
    :return: if groupby is supplied: pandas DataFrame, else: scalar value
    """

    def _f_stdae(x, y):
        return np.std(np.abs(x - y))

    return f_score(*args, f=_f_stdae, **kwargs)


@export
def medae(*args, **kwargs) -> Union[pd.DataFrame, float]:
    """
    wrapper for f_score using skearn.metrics.median_absolute_error

    :param args: passed to f_score
    :param kwargs: passed to f_score
    :return: if groupby is supplied: pandas DataFrame, else: scalar value
    """
    return f_score(*args, f=median_absolute_error, **kwargs)


@export
def mpae(*args, times_hundred: bool = True, pmax: int = 999, **kwargs) -> Union[pd.DataFrame, float]:
    """
    Wrapper for f_score using mean absolute error over mean.

    :param args: passed to f_score
    :param times_hundred: Whether to multiply by 100 for human readable percentages
    :param pmax: Max value for the percentage absolute error, used as a fallback because pae can go to infinity as
        y_true approaches zero
    :param kwargs: passed to f_score
    :return: if groupby is supplied: pandas DataFrame, else: scalar value
    """
    def _mpae(y_true, y_pred):

        _score = np.mean(np.abs(y_true - y_pred)) / np.mean(y_true)
        if times_hundred:
            _score *= 100
        return _score
    return f_score(*args, f=_mpae, **kwargs)


@export
def pae(*args, times_hundred: bool = True, pmax: int = 999, **kwargs) -> Union[pd.DataFrame, float]:
    """
    Wrapper for f_score using percentage absolute error. Does NOT work if y_true is 0 (returns np.nan in this case)

    :param args: passed to f_score
    :param times_hundred: Whether to multiply by 100 for human readable percentages
    :param pmax: Max value for the percentage absolute error, used as a fallback because pae can go to infinity as
        y_true approaches zero
    :param kwargs: passed to f_score
    :return: if groupby is supplied: pandas DataFrame, else: scalar value
    """
    def _pae(y_true, y_pred):

        _y_true = np.where(y_true == 0, np.nan, y_true)
        _frac = np.abs((y_pred / _y_true - 1))
        _score = np.nanmean(_frac)
        _score = np.where(_score > pmax, pmax, _score)
        if times_hundred:
            _score *= 100
        return _score
    return f_score(*args, f=_pae, **kwargs)


@export
def corr(*args, **kwargs) -> Union[pd.DataFrame, float]:
    """
    wrapper for f_score using pandas.Series.corr

    :param args: passed to f_score
    :param kwargs: passed to f_score
    :return: if groupby is supplied: pandas DataFrame, else: scalar value
    """

    def _f_corr(x, y): return pd.Series(x).corr(other=pd.Series(y))

    return f_score(*args, f=_f_corr, **kwargs)


@export
def df_score(df: pd.DataFrame, y_true: SequenceOrScalar, y_pred: SequenceOrScalar = None, pred_suffix: list = None,
             scores: List[Callable] = None, pivot: bool = True, scale: Union[dict, list, int] = None,
             groupby: Union[list, str] = None, multi: int = None, dropna: bool = True,
             ) -> pd.DataFrame:
    """
    creates a DataFrame displaying various kind of scores

    :param df: pandas DataFrame containing the true, pred data
    :param y_true: name of the true variable(s) inside df
    :param y_pred: name of the pred variable(s) inside df, specify either this or pred_suffix
    :param pred_suffix: name of the predicted variable suffixes. Supports multiple predictions.
        By default assumed suffix 'pred' [optional]
    :param scores: scoring functions to be used [optional]
    :param pivot: whether to pivot the DataFrame for easier readability [optional]
    :param scale: a scale for multiplying the scores, default 1 [optional]
    :param groupby: if supplied then the scores are calculated by group [optional]
    :param multi: how many multi outputs are there [optional]
    :param dropna: whether to drop na [optional]
    :return: pandas DataFrame containing al the scores
    """
    # -- assert
    if multi is None:
        multi = ['']
    else:
        multi = [f"_{_}" for _ in range(multi)]
    if pred_suffix is None:
        pred_suffix = ['pred']
    if scores is None:
        scores = [r2, rmse, mae, pae, mpae, corr]
    else:
        scores = assert_list(scores)
    df = assert_df(df)

    if groupby:
        groupby = assert_list(groupby)
    else:
        groupby = [GROUPBY_DUMMY]
        df[GROUPBY_DUMMY] = 1

    y_true = assert_list(y_true)
    pred_suffix = assert_list(pred_suffix)

    if y_pred is None:
        _y_true_new = []
        y_pred = []
        for _y_true in y_true:
            for _pred_suffix in pred_suffix:
                for _multi in multi:
                    _y_true_new.append(_y_true)
                    y_pred.append(f"{_y_true}_{_pred_suffix}{_multi}")
        y_true = _y_true_new
    else:
        y_pred = assert_list(y_pred)
        # check if y_pred is longer than y_true
        if len(y_pred) > len(y_true):
            warnings.warn('y_pred is longer than y_true, trailing entries will be dropped. If one y_true belongs'
                          'to multiple y_pred please specify it multiple times')
        elif len(y_true) > len(y_pred):
            warnings.warn('y_true is longer than y_pred, trailing entries will be dropped.')

    # -- init
    if dropna:
        df = df.dropna(subset=list_merge(y_true, y_pred))
    for _groupby in groupby:
        if df[_groupby].dtype.name == 'category':
            df[_groupby] = df[_groupby].cat.remove_unused_categories()

    if isinstance(scale, Mapping):
        for _y_true, _scale in scale.items():
            df[_y_true] *= _scale
            _index = y_true.index(_y_true)
            _y_pred = y_pred[_index]
            df[_y_pred] *= _scale
    elif is_list_like(scale):
        _i = -1
        for _scale, _y_true, _y_pred in zip(scale, y_true, y_pred):
            df[_y_true] *= _scale
            df[_y_pred] *= _scale
    elif scale is not None:
        for _y_true in y_true:
            df[_y_true] *= scale
        for _y_pred in y_pred:
            df[_y_pred] *= scale

    # -- main
    _df_score = dict_list(groupby + ['y_true', 'y_pred', 'y_ref', 'model', 'score', 'value'])

    for _y_true, _y_pred in zip(y_true, y_pred):

        if _y_pred not in df.columns:
            raise KeyError(f"{_y_pred} not in columns")

        for _score in scores:

            for _index, _df_i in df.groupby(groupby):

                _value = _score(_y_true, _y_pred, df=_df_i)

                _append_dict = {
                    'y_true': _y_true,
                    'y_pred': _y_pred,
                    'y_ref': _y_true,
                    'model': _y_pred,
                    'score': _score.__name__,
                    'value': _value
                }

                for _groupby_i in groupby:
                    _append_dict[_groupby_i] = _df_i[_groupby_i].iloc[0]

                append_to_dict_list(_df_score, _append_dict)

    _df_score = pd.DataFrame(_df_score)
    _df_score[['y_true', 'y_pred', 'score']] = _df_score[['y_true', 'y_pred', 'score']].astype(str)
    _df_score['value'] = _df_score['value'].astype(float)

    if _df_score.shape[0] == 0:
        raise ValueError("df_score is empty")

    _pivot_index = ['y_true', 'y_pred']

    if groupby != [GROUPBY_DUMMY]:
        _pivot_index += groupby
        _df_score[groupby] = _df_score[groupby].astype(str)
    else:
        _df_score = _df_score.drop([GROUPBY_DUMMY], axis=1)

    if pivot:
        _columns = _pivot_index + ['score', 'value']
        _df_score = _df_score[_columns]
        _df_score = pd.pivot_table(_df_score, index=_pivot_index, columns='score', values='value')

    return _df_score


@export
def rmsd(x: str, df: pd.DataFrame, group: str, return_df_paired: bool = False, agg_func: str = 'median',
         standardize: bool = False, to_abs: bool = False) -> Union[float, pd.DataFrame]:
    """
    calculated the weighted root mean squared difference for a reference columns x by a specific group. For a
    multi group DataFrame see :func:`df_rmsd`. For a plot see :func:`hhpy.plotting.rmsdplot`

    :param x: name of the column to calculate the rmsd for
    :param df: pandas DataFrame
    :param group: groups for which to calculate the rmsd
    :param return_df_paired: whether to return the paired DataFrame
    :param agg_func: which aggregation to use for the group value, passed to pd.DataFrame.agg
    :param standardize: whether to apply Standardization before calculating the rmsd
    :param to_abs: whether to cast x to abs before calculating the rmsd
    :return: if return_df_paired pandas DataFrame, else rmsd as float

    **Examples**

    Check out the `example notebook <https://colab.research.google.com/drive/1wvkYK80if0okXJGf1j2Kl-SxXZdl-97k>`_
    """

    _agg_by_group = '{}_by_group'.format(agg_func)

    _df = df.copy()

    if to_abs:
        _df[x] = _df[x].abs()
    if standardize:
        _df[x] = (_df[x] - _df[x].mean()) / _df[x].std()

    _df = _df.groupby([group]).agg({x: ['count', agg_func]}).reset_index()
    _df.columns = ['group', 'count', _agg_by_group]
    _df['dummy'] = 1

    _df_paired = pd.merge(_df, _df, on='dummy')
    _df_paired = _df_paired[_df_paired['group_x'] != _df_paired['group_y']]
    _df_paired['weight'] = _df_paired['count_x'] * _df_paired['count_y']
    _df_paired['difference'] = _df_paired[_agg_by_group + '_x'] - _df_paired[_agg_by_group + '_y']
    _df_paired['weighted_squared_difference'] = _df_paired['weight'] * _df_paired['difference'] ** 2

    if return_df_paired:
        return _df_paired
    else:
        return np.sqrt(_df_paired['weighted_squared_difference'].sum() / _df_paired['weight'].sum())


# get a data frame showing the root mean squared difference by group type
# noinspection PyShadowingNames
@export
def df_rmsd(x: str, df: pd.DataFrame, groups: Union[list, str] = None, hue: str = None, hue_order: list = None,
            sort_by_hue: bool = True, n_quantiles: int = 10, signif: int = 2, include_rmsd: bool = True,
            **kwargs) -> pd.DataFrame:
    """
    calculate :func:`rmsd` for reference column x with multiple other columns and return as DataFrame. For a
    plot see :func:`~hhpy.plotting.rmsdplot`

    :param x: name of the column to calculate the rmsd for
    :param df: pandas DataFrame containing the data
    :param groups: groups to calculate the rmsd or, defaults to all other columns in the DataFrame [optional]
    :param hue: further calculate the rmsd for each hue level [optional]
    :param hue_order: sort the hue levels in this order [optional]
    :param sort_by_hue: sort the values by hue rather than by group [optional]
    :param n_quantiles: numeric columns will be automatically split into this many quantiles [optional]
    :param signif: how many significant digits to use in quantile splitting [optional]
    :param include_rmsd: if False provide only a grouped DataFrame but don't actually calculate the rmsd,
        you can use include_rmsd=False to save computation time if you only need the maxperc (used in plotting)
    :param kwargs: passed to :func:`rmsd`
    :return: None

    **Examples**

    Check out the `example notebook <https://colab.research.google.com/drive/1wvkYK80if0okXJGf1j2Kl-SxXZdl-97k>`_
    """
    # avoid inplace operations
    _df = df.copy()

    _df_rmsd = pd.DataFrame()

    # x /  groups can be a list or a scaler
    if isinstance(x, list):
        _x_list = x
    else:
        _x_list = [x]

    if groups is None:
        groups = [_col for _col in _df.columns if _col not in _x_list]

    if isinstance(groups, list):
        _groups = groups
    else:
        _groups = [groups]

    if hue is not None:
        if hue in list(_df.select_dtypes(include=np.number)):
            _df[hue] = quantile_split(_df[hue], n=n_quantiles, signif=signif)
        _df[hue] = _df[hue].astype('category').cat.remove_unused_categories()
        _hues = _df[hue].cat.categories
    else:
        _hues = [None]

    # loop x
    for _x in _x_list:

        # loop groups
        for _group in _groups:

            # eliminate self dependency
            if _group == _x:
                continue

            # numerical data is split in quantiles
            if _group in list(_df.select_dtypes(include=np.number)):
                _df['_group'] = quantile_split(_df[_group], n_quantiles)
            # other data is taken as is
            else:
                _df['_group'] = _df[_group].copy()

            warnings.simplefilter(action='ignore', category=RuntimeWarning)

            # if hue is None, one calculation is enough
            for _hue in _hues:

                if hue is None:
                    _df_hue = _df
                else:
                    _df_hue = _df[_df[hue] == _hue]

                if include_rmsd:
                    _rmsd = rmsd(x=_x, df=_df_hue, group='_group', **kwargs)
                else:
                    _rmsd = np.nan

                _count = len(_df_hue['_group'])
                _maxcount = _df_hue['_group'].value_counts().reset_index()['_group'].iloc[0]
                _maxperc = _maxcount / _count
                _maxlevel = _df_hue['_group'].value_counts().reset_index()['index'].iloc[0]

                _df_rmsd_hue = pd.DataFrame(
                    {'x': _x, 'group': _group, 'rmsd': _rmsd, 'maxperc': _maxperc, 'maxlevel': _maxlevel,
                     'maxcount': _maxcount, 'count': _count}, index=[0])
                if hue is not None:
                    _df_rmsd_hue[hue] = _hue

                _df_rmsd = _df_rmsd.append(_df_rmsd_hue, ignore_index=True, sort=False)

    # postprocessing, sorting etc.
    if hue is not None:

        _df_rmsd[hue] = _df_rmsd[hue].astype('category')

        if hue_order is not None:
            _hues = hue_order
        else:
            _hues = _df_rmsd[hue].cat.categories

        _df_order = _df_rmsd[_df_rmsd[hue] == _hues[0]].sort_values(by=['rmsd'], ascending=False).reset_index(
            drop=True).reset_index().rename({'index': '_order'}, axis=1)[['group', '_order']]
        _df_rmsd = pd.merge(_df_rmsd, _df_order)

        if sort_by_hue:
            _df_rmsd = _df_rmsd.sort_values(by=[hue, '_order']).reset_index(drop=True).drop(['_order'], axis=1)
        else:
            _df_rmsd = _df_rmsd.sort_values(by=['_order', hue]).reset_index(drop=True).drop(['_order'], axis=1)
    else:
        _df_rmsd = _df_rmsd.sort_values(by=['rmsd'], ascending=False).reset_index(drop=True)

    return _df_rmsd


@export
def df_p(x: str, group: str, df: pd.DataFrame, hue: str = None, agg_func: str = 'mean', agg: bool = False,
         n_quantiles: int = 10):
    """
    returns a DataFrame with the p value. See hypothesis testing.
    :param x: name of column to evaluate
    :param group: name of grouping column
    :param df: pandas DataFrame
    :param hue: further split by hue level
    :param agg_func: standard agg function, passed to pd.DataFrame.agg
    :param agg: whether to include standard aggregation
    :param n_quantiles: numeric columns will be automatically split into this many quantiles [optional]
    :return: pandas DataFrame containing p values
    """
    # numeric to quantile
    _df, _groupby, _groupby_names, _vars, _df_levels, _levels = df_group_hue(df, group=group, hue=hue, x=x,
                                                                             n_quantiles=n_quantiles)

    _df_p = pd.DataFrame()

    # Loop levels
    for _i_1 in range(len(_levels)):
        for _i_2 in range(len(_levels)):

            _level_1 = _levels[_i_1]
            _level_2 = _levels[_i_2]

            if _level_1 != _level_2:

                _s_1 = _df[_df['_label'] == _level_1][x].dropna()
                _s_2 = _df[_df['_label'] == _level_2][x].dropna()

                # get t test / median test
                try:
                    if agg_func == 'median':
                        _p = stats.median_test(_s_1, _s_2)[1]
                    else:  # if not median then mean
                        _p = stats.ttest_ind(_s_1, _s_2, equal_var=False)[1]
                except ValueError:
                    _p = np.nan

                _df_dict = {}

                if hue is not None:

                    _df_dict[group] = _df_levels['_group'][_i_1]
                    _df_dict[group + '_2'] = _df_levels['_group'][_i_2]
                    _df_dict[hue] = _df_levels['_hue'][_i_1]
                    _df_dict[hue + '_2'] = _df_levels['_hue'][_i_1]

                else:

                    _df_dict[group] = _level_1
                    _df_dict[group + '_2'] = _level_2

                _df_dict['p'] = _p

                _df_p = _df_p.append(pd.DataFrame(_df_dict, index=[0]), ignore_index=True, sort=False)

    if agg:
        _df_p = _df_p.groupby(_groupby).agg({'p': 'mean'}).reset_index()

    return _df_p


# df with various aggregations
def df_agg(x, group, df, hue=None, agg=None, n_quantiles=10, na_to_med=False, p=True,
           p_test='mean', sort_by_count=False):
    if agg is None:
        agg = ['mean', 'median', 'std']
    if not isinstance(agg, list):
        agg = [agg]

    # numeric to quantile
    _df, _groupby, _groupby_names, _vars, _df_levels, _levels = df_group_hue(df, group=group, hue=hue, x=x,
                                                                             n_quantiles=n_quantiles,
                                                                             na_to_med=na_to_med)

    if hue is not None:
        _hue = '_hue'
    else:
        _hue = None

    # get agg
    _df_agg = _df.groupby(_groupby).agg({'_dummy': 'count', x: agg}).reset_index()
    _df_agg.columns = _groupby + ['count'] + agg
    if sort_by_count:
        _df_agg = _df_agg.sort_values(by=['count'], ascending=False)

    if p:
        _df_p = df_p(x=x, group='_group', hue=_hue, df=_df, agg_func=p_test, agg=True)
        _df_agg = pd.merge(_df_agg, _df_p, on=_groupby)

    _df_agg.columns = _groupby_names + [_col for _col in _df_agg.columns if _col not in _groupby]

    return _df_agg


# quick function to adjust group and hue to be categorical
def df_group_hue(df, group, hue=None, x=None, n_quantiles=10, na_to_med=False, keep=True):
    _df = df.copy()
    _hue = None

    if keep:
        _group = '_group'
        if hue is not None:
            _hue = '_hue'
    else:
        _group = group
        if hue is not None:
            _hue = hue

    _groupby = ['_group']
    _groupby_names = [group]
    _vars = [group]

    if hue is not None:
        _groupby.append('_hue')
        _groupby_names.append(hue)
        if hue not in _vars:
            _vars.append(hue)

    if x is not None:
        if x not in _vars:
            _vars = [x] + _vars

    _df = _df.drop([_col for _col in _df.columns if _col not in _vars], axis=1)

    _df[_group] = _df[group].copy()
    if hue is not None:
        _df[_hue] = _df[hue].copy()
    _df['_dummy'] = 1

    _df[_group] = _df[group].copy()
    if hue is not None:
        _df[_hue] = _df[hue].copy()

    # - numeric to quantile
    # group
    if _group in list(_df.select_dtypes(include=np.number)):
        _df[_group] = quantile_split(_df[group], n_quantiles, na_to_med=na_to_med)
    _df[_group] = _df[_group].astype('category').cat.remove_unused_categories()

    # hue
    if hue is not None:
        if _hue in list(_df.select_dtypes(include=np.number)):
            _df[_hue] = quantile_split(_df[hue], n_quantiles, na_to_med=na_to_med)
        _df[_hue] = _df[_hue].astype('category').cat.remove_unused_categories()
        _df['_label'] = concat_cols(_df, [_group, _hue]).astype('category')
        _df_levels = _df[[_group, _hue, '_label']].drop_duplicates().reset_index(drop=True)
        _levels = _df_levels['_label']
    else:
        _df['_label'] = _df[_group]
        _df_levels = _df[[_group, '_label']].drop_duplicates().reset_index(drop=True)
        _levels = _df_levels['_label']

    return _df, _groupby, _groupby_names, _vars, _df_levels, _levels


def order_cols(df, cols):
    return df[cols + [_col for _col in df.columns if _col not in cols]]


def df_precision_filter(df, col, precision):
    return df[(np.abs(df[col] - df[col].round(precision)) < (1 / (2 * 10 ** (precision + 1))))]


# grouped iterpolate method (avoids .apply failing if one sub group fails)
def grouped_interpolate(df, col, groupby, method=None):
    _df = df.copy()

    _dfs_i = []

    for _index_i, _df_i in df.groupby(groupby):

        try:
            _df_i[col] = _df_i[col].interpolate(method=method)
        except ValueError:  # do nothing
            _df_i[col] = _df_i[col]

        _dfs_i.append(_df_i)

    _df_interpolate = pd.concat(_dfs_i)

    return _df_interpolate[col]


def time_reg(df, t='t', y='y', t_unit='D', window=10, slope_diff_cutoff=.1, int_diff_cutoff=3, return_df_fit=False):
    if slope_diff_cutoff is None:
        slope_diff_cutoff = np.iinfo(np.int32).max
    if int_diff_cutoff is None:
        int_diff_cutoff = np.iinfo(np.int32).max

    _t_from = '{}_from'.format(t)
    _t_to = '{}_to'.format(t)
    _t_i = '{}_i'.format(t)
    _t_i_from = '{}_i_from'.format(t)
    _t_i_to = '{}_i_to'.format(t)
    _y_slope = '{}_slope'.format(y)
    _y_int = '{}_int'.format(y)
    _y_fit = '{}_fit'.format(y)
    _y_r2 = '{}_r2'.format(y)
    _y_rmse = '{}_rmse'.format(y)

    _df = df[[t, y]].copy().reset_index(drop=True)

    _t_min = _df[t].min()
    _t_max = _df[t].max()

    if isinstance(_df[t].iloc[0], pd.datetime):
        _df[_t_i] = (_df[t] - _t_min) / np.timedelta64(1, t_unit)
        _t_i_min = 0
        _t_i_max = (_df[t].max() - _t_min) / np.timedelta64(1, t_unit)
    else:
        _df[_t_i] = _df[t]
        _t_i_min = _t_min
        _t_i_max = _t_max

    _df['_y'] = (_df[y] - _df[y].mean()) / _df[y].std()

    _df['slope_rolling'] = _df[_t_i].rolling(window, min_periods=0).cov(other=_df['_y'], pairwise=False) / _df[
        _t_i].rolling(window, min_periods=0).var()
    _df['int_rolling'] = _df['_y'].rolling(window, min_periods=0).mean() - _df['slope_rolling'] * _df[_t_i].rolling(
        window, min_periods=0).mean()

    _df['slope_rolling_diff'] = np.abs(_df['slope_rolling'].diff())
    _df['int_rolling_diff'] = np.abs(_df['int_rolling'].diff())

    _df['slope_change'] = _df['slope_rolling_diff'] >= slope_diff_cutoff
    _df['int_change'] = _df['int_rolling_diff'] >= int_diff_cutoff
    _df['_change'] = (_df['slope_change']) | (_df['int_change'])

    _df_phases = _df[_df['_change']][[t, _t_i]]

    _df_phases.insert(0, _t_from, _df_phases[t].shift(1).fillna(_t_min))
    _df_phases.insert(2, _t_i_from, _df_phases[_t_i].shift(1).fillna(_t_i_min))

    _df_phases = _df_phases.rename({t: _t_to, _t_i: _t_i_to}, axis=1)

    # append row for last phase
    _df_phases = _df_phases.append(
        pd.DataFrame({
            _t_from: _df_phases[_t_from].max(),
            _t_to: _t_max,
            _t_i_from: _df_phases[_t_i_from].max(),
            _t_i_to: _t_i_max,
        }, index=[0]), ignore_index=True, sort=False
    )

    _df_phases[_y_slope] = np.nan
    _df_phases[_y_int] = np.nan
    _df_phases[_y_r2] = np.nan
    _df_phases[_y_rmse] = np.nan
    _df_phases['_keep'] = False

    _dfs = []

    _continue = False
    _t_i_from_row = None

    for _i, _row in _df_phases.iterrows():

        # check len of the phase: if len is less than window days it will be merged with next phase
        _t_i_to_row = _row[_t_i_to]

        if not _continue:
            _t_i_from_row = _row[_t_i_from]

        _df_t = _df[(_df[_t_i] >= _t_i_from_row) & (_df[_t_i] < _t_i_to_row)]

        _len_df_t = _df_t.index.max() - _df_t.index.min() + 1

        if _len_df_t < window:
            _continue = True
            continue
        else:
            _continue = False
            _df_phases['_keep'][_i] = True
            _df_phases[_t_i_from][_i] = _t_i_from_row

        # calculate slope
        _y_slope_i = _df_t[_t_i].cov(other=_df_t[y]) / _df_t[_t_i].var()
        # calculate intercept
        _y_int_i = _df_t[y].mean() - _y_slope_i * _df_t[_t_i].mean()

        # calculate y fit
        _df_t[_y_fit] = _y_int_i + _df_t[_t_i] * _y_slope_i

        _df_phases[_y_slope][_i] = _y_slope_i
        _df_phases[_y_int][_i] = _y_int_i
        _df_phases[_y_r2][_i] = r2_score(_df_t[y], _df_t[_y_fit])
        _df_phases[_y_rmse][_i] = np.sqrt(mean_squared_error(_df_t[y], _df_t[_y_fit]))

        _dfs.append(_df_t)

    _df_fit = pd.concat(_dfs)

    # postprocessing
    _df_phases = _df_phases[_df_phases['_keep']].reset_index(drop=True).drop(['_keep'], axis=1)

    if return_df_fit:
        return _df_fit
    else:
        return _df_phases


@docstr
@export
def col_to_front(df: pd.DataFrame, cols: SequenceOfScalars, inplace: bool = False) -> pd.DataFrame:
    """
    Brings one or more columns to the front (first n positions) of a DataFrame

    :param df: %(df)s
    :param cols: One or more column names to be brought to the front
    :param inplace: %(inplace)s
    :return: Modified copy of the DataFrame
    """
    _cols = assert_list(cols)
    _df = df[_cols + [_ for _ in df.columns if _ not in _cols]]

    if inplace:
        # noinspection PyProtectedMember
        df._update_inplace(_df)
    else:
        return _df


def lr(df, x, y, groupby=None, t_unit='D', do_print=True):
    # const
    _x_i = '_x_i'
    _y_slope = '{}_slope'.format(y)
    _y_int = '{}_int'.format(y)
    _y_fit = '{}_fit'.format(y)
    _y_error = '{}_error'.format(y)

    # -- init
    if do_print:
        tprint('init')

    _df = df[np.isfinite(df[x]) & np.isfinite(df[y])]

    # defaults
    if groupby:
        groupby = assert_list(groupby)
    else:
        _df['_dummy'] = 1
        groupby = ['_dummy']

    _df_out = dict_list(
        groupby + [_y_slope, _y_int, 'r2', 'rmse', 'error_mean', 'error_std', 'error_abs_mean', 'error_abs_std'])

    if isinstance(_df[x].iloc[0], pd.datetime):
        _df[_x_i] = (_df[x] - _df[x].min()) / np.timedelta64(1, t_unit)
    else:
        _df[_x_i] = _df[x]

    # loop groups

    _i = 0
    _i_max = _df[groupby].drop_duplicates().shape[0]

    for _index, _df_i in _df.groupby(groupby):

        _i += 1

        if do_print:
            tprint('Linear Regression Iteration {} / {}'.format(_i, _i_max))

        _slope = _df_i[_x_i].cov(other=_df_i[y]) / _df_i[_x_i].var()
        _int = _df_i[y].mean() - _slope * _df_i[_x_i].mean()
        _df_i[_y_fit] = _slope * _df_i[x] + _int
        _df_i[_y_error] = _df_i[_y_fit] - _df_i[y]

        _r2 = r2(_df_i[y], _df_i[_y_fit])
        _rmse = rmse(_df_i[y], _df_i[_y_fit])

        append_to_dict_list(_df_out, _index)
        append_to_dict_list(_df_out, {
            _y_slope: _slope,
            _y_int: _int,
            'r2': _r2,
            'rmse': _rmse,
            'error_mean': _df_i[_y_error].mean(),
            'error_std': _df_i[_y_error].std(),
            'error_abs_mean': _df_i[_y_error].abs().mean(),
            'error_abs_std': _df_i[_y_error].abs().std()
        })

    _df_out = pd.DataFrame(_df_out)

    if '_dummy' in _df_out.columns:
        _df_out = _df_out.drop(['_dummy'], axis=1)

    if do_print:
        tprint('Linear Regression done')

    return _df_out


def flatten(lst):
    # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists

    def _flatten_generator(_lst):

        for _x in _lst:
            if is_list_like(_x):
                for _sub_x in flatten(_x):
                    yield _sub_x
            else:
                yield _x

    return list(_flatten_generator(lst))


@export
def df_split(df: pd.DataFrame, split_by: Union[List[str], str], return_type: str = 'dict', print_key: bool = False,
             sep: str = '_', key_sep: str = '==') -> Union[list, dict]:
    """
    Split a pandas DataFrame by column value and returns a list or dict

    :param df: pandas DataFrame to be split
    :param split_by: Column(s) to split by, creates a sub-DataFrame for each level
    :param return_type: one of ['list', 'dict'], if list returns a list of sub-DataFrame, if dict returns a dictionary
        with each level as keys
    :param print_key: whether to include the column names in the key labels
    :param sep: separator to use in the key labels between columns
    :param key_sep: separator to use in the key labels between key and value
    :return: see return_type
    """

    _split_by = assert_list(split_by)

    if return_type == 'list':
        _dfs = []
    else:
        _dfs = {}

    for _i, _df in df.groupby(_split_by):

        if return_type == 'list':
            _dfs.append(_df)
        else:
            _key = qformat(pd.DataFrame(_df[_split_by]).head(1), print_key=print_key, sep=sep, key_sep=key_sep)
            _dfs[_key] = _df

    return _dfs


# concats a df, wrapper for pandas.concat
def concat(obj, ignore_index=True, sort=False, **kwargs):
    if isinstance(obj, pd.DataFrame):
        return obj
    elif len(obj) > 1:
        return pd.concat(obj, ignore_index=ignore_index, sort=sort, **kwargs)
    else:
        return obj[0]


@docstr
@export
def rank(df: pd.DataFrame, rankby: SequenceOrScalar, groupby: SequenceOrScalar = None,
         rank_ascending: bool = True, sortby: SequenceOrScalar = None,
         sortby_ascending: Union[bool, List[bool]] = None) -> pd.Series:
    """
    creates a ranking (without duplicate ranks) based on columns of a DataFrame

    :param df: %(df)s
    :param rankby: the column(s) to rankby
    :param groupby: %(groupby)s
    :param rank_ascending: Whether to rank in ascending order [optional]
    :param sortby: After the rankby column(s) the sortby columns will be sorted to break ties [optional]
    :param sortby_ascending: The sorting preference for each sortby column [optional]
    :return: pandas Series containing the rank (no duplicates)
    """

    # -- assert
    df, groupby = assert_df(df=df, groupby=groupby)
    rankby = assert_list(rankby)
    sortby = assert_list(sortby)

    # -- main
    # save row
    df[ROW_DUMMY] = range(df.shape[0])
    # handle ascending
    if sortby_ascending is None:
        _ascending = rank_ascending
    else:
        _ascending = assert_list(rank_ascending) + [True for _ in groupby] + assert_list(sortby_ascending)
    # sort
    _by = rankby + groupby + sortby
    df = df.sort_values(by=_by, ascending=_ascending).assign(rank=1)
    # rank
    df['__rank__'] = df.groupby(groupby)['rank'].cumsum()
    # sort back to original row order
    df = df.sort_values(by=ROW_DUMMY)

    # -- return
    return df['__rank__']


def kde(x, df=None, x_range=None, perc_cutoff=.1, range_cutoff=None, x_steps=1000):
    if df is not None:

        _df = df.copy()
        del df

        if x in ['value', 'perc', 'diff', 'sign', 'ex', 'ex_max', 'ex_min', 'mean', 'std', 'range',
                 'value_min', 'value_max', 'range_min', 'range_max']:
            raise ValueError('x cannot be named {}, please rename your variable'.format(x))
    else:
        _df = None

    # std cutoff = norm(0,1).pdf(1)/norm(0,1).pdf(0)
    # 1/e cutoff: range_cutoff = 1-1/e = .63
    # full width at half maximum: range_cutoff = .5
    if range_cutoff is None or range_cutoff in ['sigma', 'std']:
        _range_cutoff = stats.norm(0, 1).pdf(1) / stats.norm(0, 1).pdf(0)
    elif range_cutoff in ['e', '1/e', '1-1/e']:
        _range_cutoff = 1 - 1 / np.exp(1)
    elif range_cutoff in ['fwhm', 'FWHM', 'hm', 'HM']:
        _range_cutoff = .5
    else:
        _range_cutoff = range_cutoff + 0

    if _df is not None:
        _x = _df[x]
        _x_name = x
    else:
        _x = x
        if 'name' in dir(x):
            _x_name = x.name
        else:
            _x_name = 'x'

    assert (len(_x) > 0), 'Series {} has zero length'.format(_x_name)
    _x = pd.Series(_x).reset_index(drop=True)

    _x_name_max = f"{_x_name }_max"

    if x_range is None:
        x_range = np.linspace(np.nanmin(_x), np.nanmax(_x), x_steps)

    # -- fit kde
    _kde = stats.gaussian_kde(_x)

    # -- to df
    _df_kde = pd.DataFrame({_x_name: x_range, 'value': _kde.evaluate(x_range)})
    _df_kde['perc'] = _df_kde['value'] / _df_kde['value'].max()

    # -- get extrema
    _df_kde['diff'] = _df_kde['value'].diff()
    _df_kde['sign'] = np.sign(_df_kde['diff'])
    _df_kde['ex_max'] = _df_kde['sign'].diff(-1).fillna(0) > 0
    _df_kde['ex_min'] = _df_kde['sign'].diff(-1).fillna(0) < 0
    _df_kde['phase'] = _df_kde['ex_min'].astype(int).cumsum()

    if perc_cutoff:
        _df_kde['ex_max'] = _df_kde['ex_max'].where(_df_kde['perc'] > perc_cutoff, False)

    # -- get std
    # we get the extrema and do a full merge to find the closest one to each point
    _df_kde_ex = _df_kde.query('ex_max')[[_x_name, 'value', 'phase']].reset_index()
    _df_kde_ex['mean'] = np.nan
    _df_kde_ex['std'] = np.nan
    _df_kde_ex['range'] = np.nan
    _df_kde_ex['range_min'] = np.nan
    _df_kde_ex['range_max'] = np.nan
    _df_kde_ex['value_min'] = np.nan
    _df_kde_ex['value_max'] = np.nan

    for _index, _row in _df_kde_ex.iterrows():
        _df_kde_i = _df_kde[_df_kde['phase'] == _row['phase']]

        # Width of Peak range
        _df_kde_i = _df_kde_i[_df_kde_i['value'] >= _row['value'] * _range_cutoff]

        _x_min = _df_kde_i[_x_name].iloc[0]
        _x_max = _df_kde_i[_x_name].iloc[-1]

        _x_i = np.extract((_x > _x_min) & (_x < _x_max), _x)

        _mean, _std = stats.norm.fit(_x_i)

        _df_kde_ex['mean'].loc[_index] = _mean
        _df_kde_ex['std'].loc[_index] = _std

        _df_kde_ex['range'].loc[_index] = _x_max - _x_min
        _df_kde_ex['range_min'].loc[_index] = _x_min
        _df_kde_ex['range_max'].loc[_index] = _x_max
        _df_kde_ex['value_min'].loc[_index] = _df_kde_i['value'].iloc[0]
        _df_kde_ex['value_max'].loc[_index] = _df_kde_i['value'].iloc[-1]

    return _df_kde, _df_kde_ex


# wrapper to quickly aggregate df
def qagg(df: pd.DataFrame, groupby, columns=None, agg=None, reset_index=True):
    if agg is None:
        agg = ['mean', 'std']
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns

    _df_agg = df.groupby(groupby).agg({_: agg for _ in columns})
    _df_agg = _df_agg.set_axis(flatten([[_ + '_mean', _ + '_std'] for _ in columns]), axis=1, inplace=False)
    if reset_index:
        _df_agg = _df_agg.reset_index()
    return _df_agg


@export
def mahalanobis(point: Union[pd.DataFrame, pd.Series, np.ndarray], df: pd.DataFrame = None, params: List[str] = None,
                do_print: bool = True) -> Union[float, List[float]]:
    """
    Calculates the Mahalanobis distance for a single point or a DataFrame of points

    :param point: The point(s) to calculate the Mahalanobis distance for
    :param df: The reference DataFrame against which to calculate the Mahalanobis distance
    :param params: The columns to calculate the Mahalanobis distance for
    :param do_print: Whether to print intermediate steps to the console
    :return: if a single point is passed: Mahalanobis distance as float, else a list of floats
    """
    if df is None:
        df = point

    _df = df.copy()
    del df

    if params is None:
        params = _df.columns
    else:
        _df = _df[params]

    try:
        _vi = np.linalg.inv(_df.cov())
    except np.linalg.LinAlgError:
        return np.nan

    _y = _df.mean().values

    if isinstance(point, pd.DataFrame):

        _out = []

        _it = -1
        for _index, _row in point.iterrows():

            _it += 1

            if do_print:
                progressbar(_it, point.shape[0])

            _x = _row[params].values
            _out.append(distance.mahalanobis(_x, _y, _vi))

        if do_print:
            progressbar()
        return _out

    elif isinstance(point, pd.Series):
        _x = point[params].values
    else:
        _x = np.array(point)

    return distance.mahalanobis(_x, _y, _vi)


def multi_melt(df, cols, suffixes, id_vars, var_name='variable', sep='_', **kwargs):
    # for multi melt to work the columns must share common suffixes

    _df = df.copy()
    del df

    _df_out = []

    for _col in cols:
        _value_vars = ['{}{}{}'.format(_col, sep, _suffix) for _suffix in suffixes]

        _df_out_i = _df.melt(id_vars=id_vars, value_vars=_value_vars, value_name=_col, var_name=var_name, **kwargs)
        _df_out_i[var_name] = _df_out_i[var_name].str.slice(len(_col) + len(sep))
        _df_out_i = _df_out_i.sort_values(by=assert_list(id_vars) + [var_name]).reset_index(drop=True)
        _df_out.append(_df_out_i)

    _df_out = pd.concat(_df_out, axis=1).pipe(drop_duplicate_cols)

    return _df_out


# for resampling integer indexes
def resample(df, rule=1, on=None, groupby=None, agg='mean', columns=None, adj_column_names=True, factor=1, **kwargs):
    assert isinstance(df, pd.DataFrame), 'df must be a DataFrame'

    _df = df.copy()
    del df

    if on is not None:
        _df = _df.set_index(on)
    if columns is None:
        _columns = _df.select_dtypes(include=np.number).columns
    else:
        _columns = columns
    if groupby is not None:
        _columns = [_ for _ in _columns if _ not in assert_list(groupby)]
        _df = _df.groupby(groupby)

    # convert int to seconds to be able to use .resample
    _df.index = pd.to_datetime(_df.index * factor, unit='s')

    # resample as time series
    _df = _df.resample('{}s'.format(rule), **kwargs)

    # agg
    _adj_column_names = False
    if agg == 'mean':
        _df = _df.mean()
    elif agg == 'median':
        _df = _df.median()
    elif agg == 'sum':
        _df = _df.sum()
    else:
        _df = _df.agg({_: agg for _ in _columns})
        if adj_column_names:
            _adj_column_names = True

    # back to int
    _df.index = ((_df.index - pd.to_datetime('1970-01-01')).total_seconds() / factor)
    if _adj_column_names:
        _column_names = []
        for _col in _columns:
            for _agg in assert_list(agg):
                _column_names += ['{}_{}'.format(_col, _agg)]
        _df.columns = _column_names

    return _df


@docstr
@export
def df_count(x: str, df: pd.DataFrame, hue: Optional[str] = None, sort_by_count: bool = True, top_nr: int = 5,
             x_base: Optional[float] = None, x_min: Optional[float] = None, x_max: Optional[float] = None,
             other_name: str = 'other', other_to_na: bool = False, na: Union[bool, str] = 'drop') -> pd.DataFrame:
    """
    Create a DataFrame of value counts. Supports hue levels and is therefore useful for plots, for an application
    see :func:`~hhpy.plotting.countplot`

    :param x: %(x)s
    :param df: %(df)s
    :param hue: %(hue)s
    :param sort_by_count: Whether to sort the DataFrame by value counts [optional]
    :param top_nr: %(top_nr)s
    :param x_base: if supplied: cast x to integer multiples of x_base, useful when you have float data that would
        result in many unique counts for close numbers [optional]
    :param x_min: limit the range of valid numeric x values to be greater than or equal to x_min [optional]
    :param x_max: limit the range of valid numeric x values to be less than or equal to x_max [optional]
    :param other_name: %(other_name)s
    :param other_to_na: %(other_to_na)s
    :param na: whether to keep (True, 'keep') na values and implicitly cast to string
        or drop (False, 'drop') them [optional]
    :return: pandas DataFrame containing the counts by x (and by hue if it is supplied)
    """
    # -- init
    # avoid inplace operations
    df = assert_df(df)

    # if applicable: drop NaN
    if (not na) or (na == 'drop'):
        # true NaN
        df = df.dropna(subset=[x])
        # string NaN
        df = df[~df[x].isin(STRING_NAN)]
        if hue is not None:
            # true NaN
            df = df.dropna(subset=[hue])
            # string NaN
            df = df[~df[hue].isin(STRING_NAN)]

    # in case the original column is already called count it is renamed to count_org
    if x == 'count':
        x = 'count_org'
        df = df.rename({'count': 'count_org'}, axis=1)

    # -- preprocessing
    if x_base:

        # round to multiples of x_int
        df[x] = np.round(df[x] / x_base) * x_base
        if isinstance(x_base, int):
            df[x] = df[x].astype(int)

        # apply x limits
        if x_min is None:
            x_min = df[x].min()
        if x_max is None:
            x_max = df[x].max()
        _df_xs = pd.DataFrame({x: range(x_min, x_max, x_base)})
        _xs_on = [x]

        # init hues
        if hue is not None:
            _df_hues = df[[hue]].drop_duplicates().reset_index().assign(_dummy=1)
            _df_xs = pd.merge(_df_xs.assign(_dummy=1), _df_hues, on='_dummy').drop(['_dummy'], axis=1)
            _xs_on = _xs_on + [hue]

    else:
        # apply x limits (ignored if not numeric)
        if x in df.select_dtypes(np.number):
            if x_min:
                df[x] = df[x].where(lambda _: _ >= x_min, x_min)
            if x_max:
                df[x] = df[x].where(lambda _: _ <= x_max, x_max)

    # to string
    df[x] = df[x].astype(str)
    if hue is not None:
        df[hue] = df[hue].astype(str)

    # if applicable: apply top_n_coding (both x and hue)
    if top_nr:
        df[x] = top_n_coding(s=df[x], n=top_nr, other_name=other_name, other_to_na=other_to_na)
        if hue is not None:
            df[hue] = top_n_coding(s=df[hue], n=top_nr, other_name=other_name, other_to_na=other_to_na)

    # init groupby
    _groupby = [x]
    if hue is not None:
        _groupby = _groupby + [hue]

    # we use a dummy column called count and sum over it by group to retain the original x column values
    _df_count = df.assign(count=1).groupby(_groupby).agg({'count': 'sum'}).reset_index()

    # if applicable: append 0 entries for numerical x inside x_range
    if x_base:
        # was already called with same if before
        # noinspection PyUnboundLocalVariable
        _df_count = pd.merge(_df_count, _df_xs, on=_xs_on, how='outer')
        _df_count['count'] = _df_count['count'].fillna(0)

    # create total count (for perc)
    _count_x = 'count_{}'.format(x)
    _count_hue = 'count_{}'.format(hue)

    if hue is None:
        _df_count[_count_hue] = _df_count['count'].sum()
        _df_count[_count_x] = _df_count['count']
    else:
        _df_count[_count_x] = _df_count.groupby(x)['count'].transform(pd.Series.sum)
        _df_count[_count_hue] = _df_count.groupby(hue)['count'].transform(pd.Series.sum)

    # sort
    if sort_by_count:
        _df_count = _df_count.sort_values([_count_x], ascending=False).reset_index(drop=True)

    # add perc columns
    _df_count[f"perc_{x}"] = np.round(_df_count['count'] / _df_count[_count_x] * 100, 2)
    _df_count[f"perc_{hue}"] = np.round(_df_count['count'] / _df_count[_count_hue] * 100, 2)

    return _df_count


# return prediction accuracy in percent
def get_accuracy(class_true, class_pred):
    return np.where(class_true.astype(str) == class_pred.astype(str), 1, 0).sum() / len(class_true)


# takes a numeric pandas series and splits it into groups, the groups are labeled by INTEGER multiples of the step value
def numeric_to_group(pd_series, step=None, outer_limit=4, suffix=None, use_abs=False, use_standard_scaler=True):
    # outer limit is given in steps, only INTEGER values allowed
    outer_limit = int(outer_limit)

    # make a copy to avoid inplace effects
    _series = pd.Series(deepcopy(pd_series))

    # use standard scaler to center around mean with std +- 1
    if use_standard_scaler:
        _series = StandardScaler().fit(_series.values.reshape(-1, 1)).transform(_series.values.reshape(-1, 1)).flatten()

    # if step is none: use 1 as step
    if step is None:
        step = 1
    if suffix is None:
        if use_standard_scaler:
            suffix = 'std'
        else:
            suffix = 'step'

    if suffix != '':
        suffix = '_' + suffix

    # to absolute
    if use_abs:
        _series = np.abs(_series)
    else:
        # gather the +0 and -0 group to 0
        _series = np.where(np.abs(_series) < step, 0, _series)

    # group

    # get sign
    _series_sign = np.sign(_series)

    # divide by step, floor and integer
    _series = (np.floor(np.abs(_series) / step)).astype(int) * np.sign(_series).astype(int)

    # apply outer limit
    if outer_limit is not None:
        _series = np.where(_series > outer_limit, outer_limit, _series)
        _series = np.where(_series < -outer_limit, -outer_limit, _series)

    # make a pretty string
    _series = pd.Series(_series).apply(lambda x: '{0:n}'.format(x)).astype('str') + suffix

    # to cat
    _series = _series.astype('category')

    return _series


@export
def top_n(s: Sequence, n: Union[int, str], w: Optional[Sequence] = None, n_max: int = 20) -> list:
    """
    Select n elements form a categorical pandas series with the highest counts. Ties are broken by sorting
        s ascending

    :param s: pandas Series to select from
    :param n: how many elements to return, you can pass a percentage to return the top n %
    :param w: weights, if given the weights are summed instead of just counting entries in s [optional]
    :param n_max: how many elements to return at max if n is a percentage, set to None for no max [optional]
    :return: List of top n elements
    """

    # -- case int:
    if isinstance(n, int) or str(n).isnumeric():
        n = int(n)
        if w is None:
            return list(pd.Series(s).value_counts().reset_index()['index'][:n])
        else:
            return pd.DataFrame({'s': s, 'w': w}).groupby('s').agg({'w': 'sum'}) \
                       .sort_values(by='w', ascending=False).index.tolist()[:n]
    # -- case str (percent)
    elif isinstance(n, str):
        if '%' not in n:
            raise ValueError(f"Please specify n as integer or percent with percentage sign %")
        n = float(n.split('%')[0]) / 100.
        _df = pd.DataFrame({'s': s})
        # get weights
        if w is None:
            _df['w'] = 1
        else:
            _df['w'] = w
        # sum weights
        _df = _df.groupby('s').agg({'w': 'sum'}).reset_index().sort_values(by=['w', 's'], ascending=[False, True])
        # calculate cutoff
        _df['c'] = _df['w'].cumsum() / _df['w'].sum()
        _df = _df[_df['c'].shift(1).fillna(0) <= n]
        _n_list = _df['s'].tolist()
        if n_max is not None and len(_n_list) > n_max:
            _n_list = _n_list[:n_max]

        return _n_list


@docstr
@export
def top_n_coding(s: Sequence, n: int, other_name: str = 'other', na_to_other: bool = False,
                 other_to_na: bool = False, w: Optional[Sequence] = None) -> pd.Series:
    """
    Returns a modified version of the pandas series where all elements not in top_n become recoded as 'other'

    :param s: Pandas Series to adjust
    :param n: How many unique elements to keep
    :param other_name: Name of the other element [optional]
    :param na_to_other: Whether to cast missing elements to other [optional]
    :param other_to_na: %(other_to_na)s
    :param w: Weights, if given the weights are summed instead of just counting entries in s [optional]
    :return: Adjusted pandas Series
    """

    # we have to cast to string so we can set the other name
    _s = pd.Series(s).astype('str')
    _top_n = top_n(_s, n, w=w)
    if other_to_na:
        _s = pd.Series(np.where(_s.isin(_top_n), _s, 'nan'))
    else:
        _s = pd.Series(np.where(_s.isin(_top_n), _s, other_name))
    if na_to_other:
        _s = np.where(~_s.isin(STRING_NAN), _s, other_name)
    _s = pd.Series(_s)

    # get back the old properties of the series (or you'll screw the index)
    if isinstance(s, pd.Series):
        _s.name = s.name
        _s.index = s.index

    # convert to cat
    _s = _s.astype('category')

    return _s


@export
def k_split(df: pd.DataFrame, k: SequenceOrScalar = 5, groupby: Union[Sequence, str] = None,
            sortby: Union[Sequence, str] = None, random_state: int = None, do_print: bool = True,
            return_type: Union[str, int] = 0) -> Union[pd.Series, tuple]:
    """
    Splits a DataFrame into k (equal sized) parts that can be used for train test splitting or k_cross splitting

    :param df: pandas DataFrame to be split
    :param k: if integer: how many (equal sized) parts to split the DataFrame into
      if string: (timestamp) value where to split, requires sortby [optional]
    :param groupby: passed to pandas.DataFrame.groupby before splitting,
        ensures that each group will be represented equally in each split part [optional]
    :param sortby: if True the DataFrame is ordered by these column(s) and then sliced into parts from the top
        if False the DataFrame is sorted randomly before slicing [optional]
    :param random_state: random_state to be used in random sorting, ignore if sortby is True [optional]
    :param do_print: whether to print steps to console [optional]
    :param return_type: if one of ['Series', 's'] returns a pandas Series containing the k indices range(k)
        if integer < k returns tuple of shape (df_train, df_test) where the return_type'th part
        is equal to df_test and the other parts are equal to df_train
    :return: depending on return_type either a pandas Series or a tuple
    """

    if do_print:
        tprint(f"k_split: splitting 1:{k} ...")

    # -- assert
    df, groupby = assert_df(df=df, groupby=groupby)

    # -- main
    if is_list_like(k):
        # if k is list like then assume it is a lift of values to split at
        # check for sortby
        if sortby is None:
            raise ValueError(f"k={k} (string, datetime) requires sortby")
        # prepare output df
        _df_out = df.copy()
        # init k index on first k value
        _df_out['_k_index'] = np.where(_df_out[sortby] < k[0], len(k)+1, len(k))
        # compare with the others
        for _k_index in range(1, len(k), 1):
            # the value has to be between the previous and this value
            _df_out['_k_index'] = np.where((_df_out[sortby] < k[_k_index]) & (_df_out[sortby] >= k[_k_index - 1]),
                                           _k_index+1, _df_out['_k_index'])
    elif isinstance(k, (str, datetime)):
        # check for sortby

        if sortby is None:
            raise ValueError(f"k={k} (string, datetime) requires sortby")
        _df_out = df.copy()
        _df_out['_k_index'] = np.where(_df_out[sortby] < k, 1, 0)
        k = 1
    else:
        _df_out = []
        # - split each group
        for _index, _df_i in df.groupby(groupby):

            # sort (randomly or by given value)
            if sortby is None:
                _df_i = _df_i.sample(frac=1, random_state=random_state)
            else:
                if sortby == 'index':
                    _df_i = _df_i.sort_index()
                else:
                    _df_i = _df_i.sort_values(by=sortby)
            # get row numbers in INVERSE order so that key ordering will be inverse
            # (in case of sorted: new data has k = 0)
            _df_i[ROW_DUMMY] = range(_df_i.shape[0])[::-1]
            # assign k index based on row number
            _row_split = int(np.ceil(_df_i.shape[0] / k))
            _df_i['_k_index'] = _df_i[ROW_DUMMY] // _row_split
            # append to list
            _df_out.append(_df_i)
        # - merge
        _df_out = pd.concat(_df_out).sort_index()
        # drop row dummy
        _df_out = _df_out.drop(ROW_DUMMY, axis=1)
        # drop groupby dummy
        if GROUPBY_DUMMY in _df_out.columns:
            _df_out = _df_out.drop(GROUPBY_DUMMY, axis=1)
    # tprint
    if do_print:
        tprint('k_split done')
    # -- return
    if return_type in range(k):
        _df_train = _df_out[_df_out['_k_index'] != return_type].drop('_k_index', axis=1)
        _df_test = _df_out[_df_out['_k_index'] == return_type].drop('_k_index', axis=1)
        return _df_train, _df_test
    else:
        return _df_out['_k_index']


@docstr
@export
def remove_unused_categories(df: pd.DataFrame, inplace: bool = False) -> Optional[pd.DataFrame]:
    """
    Remove unused categories from all categorical columns in the DataFrame

    :param df: %(df)s
    :param inplace: %(inplace)s
    :return: pandas DataFrame with the unused categories removed
    """

    if not inplace:
        df = assert_df(df)

    for _col in df.select_dtypes('category'):
        df[_col] = df[_col].cat.remove_unused_categories()

    if not inplace:
        return df


@export
def read_csv(path: str, nrows: int = None, encoding: str = None, errors: str = 'replace', kws_open: Mapping = None,
             **kwargs):
    """
    wrapper for pandas.read_csv that reads the file into an IOString first. This enables one to use the error handling
    params of open which is very useful when opening a file with an uncertain encoding or illegal characters
    that would trigger an encoding error in pandas.read_csv

    :param path: path to file
    :param nrows: how many rows to read, defaults to all [optional]
    :param encoding: encoding to pass to open [optional]
    :param errors: how to handle errors, see open [optional]
    :param kws_open: other keyword arguments passed to open [optional]
    :param kwargs: other keyword arguments passed to pandas.read_csv [optional]
    :return:
    """
    # -- init
    # - defaults
    if kws_open is None:
        kws_open = {}

    # -- main
    with open(path.encode('utf-8'), 'r', encoding=encoding, errors=errors, **kws_open) as _f:
        if nrows:
            _csv = StringIO('\n'.join([next(_f) for _ in range(nrows + 1)]))
        else:
            _csv = StringIO(_f.read())

    # -- return
    return pd.read_csv(deepcopy(_csv), nrows=nrows, **kwargs)


@docstr
@export
def get_columns(df: pd.DataFrame, dtype: Union[SequenceOrScalar, np.number] = None,
                to_list: bool = False) -> Union[list, pd.Index]:
    """
    A quick way to get the columns of a certain dtype. I added this because in pandas 1.0.0
    pandas.DataFrame.select_dtypes('string') sometimes throws an error when the column does not contain correctly
    formated data.

    :param df: %(df)s
    :param dtype: dtype to filter for, mimics behaviour of pandas.DataFrame.select_dtypes
    :param to_list: Whether to return a list instead of a pandas.Index
    :return: object containing the column names - if to_list: list, else pandas.Index
    """

    # -- init
    _columns = []
    # -- main
    # - dtype filter
    for _index, _value in df.dtypes.iteritems():
        for _dtype in assert_list(dtype):
            # map int, float, boolean, np.number to their string representation
            if _dtype in [int, float, bool]:
                _dtype = 'int'
            elif _dtype == float:
                _dtype = 'float'
            elif _dtype == bool:
                _dtype = 'bool'
            elif _dtype == np.number:
                _dtype = 'number'
            # main comparison: check if given dtype string or type
            if isinstance(_dtype, str):
                # look for str representation -> enforce lower case
                _dtype = _dtype.lower()
                _value = str(_value).lower()
                if _dtype in ['number', 'numeric']:
                    # generic number
                    if ('float' in _value) or ('int' in _value):
                        _columns.append(_index)
                elif _dtype.lower() in _value:
                    # user specified type
                    _columns.append(_index)
            elif isinstance(_value, _dtype):
                # use an isinstance comparison
                _columns.append(_index)
    # - index to list
    if not to_list:
        _columns = pd.Index(_columns)
    # -- return
    return _columns


@docstr
@export
def reformat_columns(df: pd.DataFrame, printf: Callable = None, **kwargs) -> pd.DataFrame:
    """
    A quick way to clean the column names of a DataFrame

    :param df: %(df)s
    :param printf: Printing Function to use for steps [optional]
    :param kwargs: Additional keyword arguments passed to DFMapping [optional]
    :return: DataFrame with reformated column names
    """

    # -- assert
    df = assert_df(df)

    # -- main
    df = DFMapping(df, values=False, printf=printf, **kwargs).transform(df)

    return df
