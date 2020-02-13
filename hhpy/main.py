"""
hhpy.main.py
~~~~~~~~~~~~~~~~

Contains basic calculation functions that are used in the more specialized versions of the package but can also be used
on their own

"""
# ---- imports
# --- standard imports
import numpy as np
import pandas as pd
import warnings
import os
import sys
import datetime
import h5py
import re
# --- third party imports
from typing import Any, Callable, Union, Sequence, Mapping, List, Optional, Iterable
from docrep import DocstringProcessor
from collections import defaultdict
from copy import deepcopy
from time import sleep
from json import JSONDecodeError
# --- optional imports
try:
    # noinspection PyPackageRequirements
    from googletrans import Translator
except ImportError:
    Translator = None
try:
    # noinspection PyPackageRequirements
    import emoji
except ImportError:
    emoji = None

# ---- init
pd.plotting.register_matplotlib_converters()
pd.options.mode.chained_assignment = None

# ---- variables
# --- globals for functions
global_t = datetime.datetime.now()  # for times progress bar
global_tprint_len = 0  # for temporary printing
# --- typing classes
Scalar = Union[int, float, str, bytes]
ListOfScalars = Union[List[Scalar], Scalar]
SequenceOrScalar = Union[Sequence, Scalar]
SequenceOfScalars = Union[Sequence[Scalar], Scalar]
# --- rcParams
rcParams = {
    'tprint.r_loc': 'front',
}

# ---- constants
# --- true constants
STRING_NAN = ['nan', 'nat', 'NaN', 'NaT']
GROUPBY_DUMMY = '__groupby__'
# --- validations
validations = {
    'reformat_string__case': ['lower', 'upper'],
    'dict_inv__duplicates': ['adjust', 'drop']
}
# --- docstr
docstr = DocstringProcessor(
    df='Pandas DataFrame containing the data',
    x='Main variable, name of a column in the DataFrame or vector data',
    warn='Whether to show UserWarnings triggered by this function. Set to False to suppress, other warnings will still '
         'be triggered [optional]',
    **validations
)


# ---- decorators
def export(fn):
    # based on https://stackoverflow.com/questions/41895077/export-decorator-that-manages-all
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


# ---- classes
@export
class BaseClass:
    """
        Base class for various classes deriving from this. Implements __repr__, converting to dict as well as
        saving to pickle and restoring from pickle.
        Does NOT provide __init__ since it cannot be used by itself
    """

    # --- globals
    __name__ = 'BaseClass'
    __attributes__ = []

    # --- functions
    def __repr__(self):
        return _get_repr(self)

    def to_dict(self) -> dict:
        """
        Converts self to a dictionary

        :return: Dictionary
        """
        if len(self.__attributes__) == 0:
            warnings.warn('self.__attributes__ has length zero, did you declare it?')

        _dict = {}
        for _attr_name in list_merge('__name__', self.__attributes__):
            _attr = self.__getattribute__(_attr_name)
            if 'to_dict' in dir(_attr):
                _attr = _attr.to_dict()
            elif is_list_like(_attr):
                if isinstance(_attr, Mapping):
                    for _key, _value in _attr.items():
                        if 'to_dict' in dir(_value):
                            # noinspection PyUnresolvedReferences
                            _attr[_key] = _value.to_dict()
                else:
                    for _i in range(len(_attr)):
                        if 'to_dict' in dir(_attr[_i]):
                            _attr[_i] = _attr[_i].to_dict()
            _dict[_attr_name] = _attr

        return _dict

    def from_dict(self, dct: Mapping):
        """
        Restores self from a dictionary

        :param dct: Dictionary created from :meth:`~BaseClass.to_dict`
        :return: None
        """
        if len(self.__attributes__) == 0:
            warnings.warn('self.__attributes__ has length zero, did you declare it?')

        for _attr_name in list_merge('__name__', self.__attributes__):
            if _attr_name not in dct.keys():
                continue
            _attr = dct[_attr_name]
            if is_list_like(_attr):
                if isinstance(_attr, Mapping):

                    if '__name__' in _attr.keys():

                        _name = _attr['__name__']
                        if _name[:3] == 'cf.':
                            _name = _name[3:]
                        _attr_eval = eval(_name + '()')
                        if 'from_dict' in dir(_attr_eval):
                            _attr_eval.from_dict(_attr)
                        _attr = _attr_eval

                    else:

                        for _attr_key, _attr_value in _attr.items():

                            if isinstance(_attr_value, Mapping):

                                if '__name__' in _attr_value.keys():

                                    _name = _attr_value['__name__']
                                    if _name[:3] == 'cf.':
                                        _name = _name[3:]
                                    _attr_eval = eval(_name + '()')
                                    if 'from_dict' in dir(_attr_eval):
                                        _attr_eval.from_dict(_attr_value)
                                    # noinspection PyUnresolvedReferences
                                    _attr[_attr_key] = _attr_eval

                else:
                    for _i in range(len(_attr)):

                        _attr_value = _attr[_i]

                        if isinstance(_attr_value, Mapping):

                            if '__name__' in _attr_value.keys():

                                _name = _attr_value['__name__']
                                if _name[:3] == 'cf.':
                                    _name = _name[3:]
                                _attr_eval = eval(_name + '()')
                                if 'from_dict' in dir(_attr_eval):
                                    _attr_eval.from_dict(_attr_value)
                                _attr[_i] = _attr_eval

            self.__setattr__(_attr_name, _attr)

    def save(self, filename: str, f: Callable = pd.to_pickle):
        """
        Save self to file using an arbitrary function that supports saving dictionaries. Note that the object
        is implicitly converted to a dictionary before saving.

        :param filename: filename (path) to be used
        :param f: function to be used [optional]
        :return: None
        """
        _dict = self.copy().to_dict()
        f(_dict, filename)

    def load(self, filename: str, f: Callable = pd.read_pickle):
        """
        Load self from file saved with :meth:`~BaseClass.save` using an arbitrary function that supports loading
        dictionaries.

        :param filename: filename (path) of the file
        :param f: function to be used [optional]
        :return: None
        """
        self.from_dict(f(filename))

    def to_pickle(self, *args, **kwargs):
        """
        Wrapper for :meth:`~BaseClass.save` using f =
        `pandas.to_pickle <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_pickle.html>`_

        :param args: passed to save [optional]
        :param kwargs: passed to save [optional]
        :return: see save
        """
        self.save(*args, f=pd.to_pickle, **kwargs)

    def read_pickle(self, *args, **kwargs):
        """
        Wrapper for :meth:`BaseClass.load` using f =
        `pandas.read_pickle <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html>`_

        :param args: passed to load [optional]
        :param kwargs: passed to load [optional]
        :return: see load
        """
        self.load(*args, f=pd.read_pickle, **kwargs)

    def copy(self):
        """
        Uses `copy.deepcopy <https://docs.python.org/3/library/copy.html>`_ to return a copy of the object

        :return: Copy of self
        """
        return deepcopy(self)


# ---- functions
# --- internal functions
def _get_repr(obj: Any, rules: Mapping[type, Callable] = None, map_list: bool = True, map_dict: bool = True) -> str:
    """
    basic reuseable repr method for custom classes

    :param obj: Any instance of a custom class implementing .__name__ (str) and .__attributes__ (List[str])
    :param rules: Rules as dictionary of types and callables. Callable argument will be attribute value
    :param map_list: Whether to map the rules to list elements
    :return: str
    """
    def _get_repr_i(value: Any) -> str:

        __repr_i = repr(value)
        # case by case selector
        if isinstance(value, np.ndarray):
            __repr_i = f"numpy.ndarray{value.shape})"
        elif isinstance(value, pd.DataFrame):
            __repr_i = f"pandas.DataFrame{value.shape}"
        elif isinstance(value, pd.Series):
            __repr_i = f"pandas.Series{value.shape}"
        elif hasattr(value, '__code__'):
            if hasattr(value, '__name__'):
                __name = value.__name__
            else:
                __name = 'Callable'
            __repr_i = f"{__name}{value.__code__.co_varnames}"
        # eval custom rules
        for _type, _callable in rules.items():
            if isinstance(value, _type):  # or (value == _type)
                try:
                    __repr_i = _callable(value)
                except Exception as _e:
                    print(f"{_e.__class__.__name__}: {_e} handled for {value}")

        return __repr_i

    # -- assert

    # check if self.__name__ is still the same as Base (i.e. unset)
    # - name
    if hasattr(obj, '__name__'):
        _name = obj.__name__
    else:
        warnings.warn('Object has no __name__ attribute, did you declare it?')
        _name = '{Unnamed}'
    if obj.__name__ == 'BaseClass' and obj.__class__ != BaseClass:
        warnings.warn('__name__ is equal to BaseClass, did you declare it?')
    # - attributes
    if hasattr(obj, '__attributes__'):
        _attributes = obj.__attributes__
    else:
        warnings.warn('Object has no __attributes__ attribute, did you declare it?')
        _attributes = []
    if len(obj.__attributes__) == 0:
        warnings.warn('self.__attributes__ has length zero, did you declare it?')
    # - rules
    if rules is not None and not isinstance(rules, Mapping):
        raise ValueError('rules should be a dictionary of types and callables')

    # -- init
    _repr = f"{_name}("
    # iterator for separator handling
    _it = -1
    # -- main
    for _attribute in _attributes:
        # don't print __name__
        if _attribute == '__name__':
            continue
        # check if _attr exists
        if hasattr(obj, _attribute):
            # get value
            _value = obj.__getattribute__(_attribute)
            if _value is None:
                continue
            if map_list and isinstance(_value, list):
                _value = [_get_repr_i(_) for _ in _value]
            if map_dict and isinstance(_value, dict):
                for __key, __value in _value.items():
                    _value[__key] = _get_repr_i(__value)
            # get repr_i from value
            _repr_i = _get_repr_i(value=_value)
            # only iterate if you print
            _it += 1
            # add separator
            if _it > 0:
                _repr += ', '
            # add to repr string
            _repr += f"{_attribute}={_repr_i}"
        else:
            warnings.warn(f"{_attribute} is specified in self.__attributes__ but does not exist. Skipping...")
            continue
    # close brace
    _repr += ')'
    # -- return
    return _repr


# --- exported functions
@export
def today(date_format: str = '%Y_%m_%d') -> str:
    """
    Returns today's date as string

    :param date_format: The formating string for the date. Passed to strftime
    :return: Formated String

    **Examples**

    >>> today()
    '2020_01_14'

    """
    return datetime.datetime.today().strftime(date_format)


@export
def size(byte: int, unit: str = 'MB', dec: int = 2) -> str:
    """
    Formats bytes as human readable string

    :param byte: The byte amount to be formated
    :param unit: The unit to display the output in, supports 'KB', 'MB', 'GB' and 'TB'
    :param dec: The number of decimals to use
    :return: Formated bytes as string

    **Examples**

    >>> size(1024, unit='KB')
    '1.0 KB'

    >>> size(1024*1024*10, unit='MB')
    '10.0 MB'

    >>> size(10**10, unit='GB')
    '9.31 GB'

    """
    _power = {'KB': 1, 'MB': 2, 'GB': 3, 'TB': 4}[unit.upper()]

    return '{} {}'.format(np.round(byte / (1024 ** _power), dec), unit)


@export
def mem_usage(pandas_obj, *args, **kwargs) -> str:
    """
    Get memory usage of a pandas object

    :param pandas_obj: Pandas object to get the memory usage of
    :param args: passed to size()
    :param kwargs: passed to size()
    :return: memory usage of a pandas object formated as string

    **Examples**

    >>> import seaborn as sns
    >>> diamonds = sns.load_dataset('diamonds')
    >>> mem_usage(diamonds)
    '12.62 MB'

    """
    if isinstance(pandas_obj, pd.DataFrame):
        _usage_b = pandas_obj.memory_usage(deep=True).sum()

    else:  # we assume if not a df it's a series
        _usage_b = pandas_obj.memory_usage(deep=True)

    return size(_usage_b, *args, **kwargs)


@export
def tprint(*args, sep: str = ' ', r_loc: str = rcParams['tprint.r_loc'], **kwargs):
    """
    Wrapper for print() but with a carriage return at the end.
    This results in the text being overwritten by the next print call.
    Can be used for progress bars and the like.

    :param args: arguments to print
    :param sep: separator
    :param r_loc: where to put the carriage return, one of ['front', 'end']. Some interpreters (e.g. PyCharm)
        don't like end since they automatically clear the print area after each carriage return. When using front
        a regular print after a tprint will start at the end of the tprint. When using 'end' a regular
        print will overwrite the tprint output but will not clear the console so if it is . In either case a blank
        tprint() will clear the console and restore default print behaviour.
    :param kwargs: passed to print
    :return: None

    **Examples**

    >>> tprint('Hello World')
    'Hello World'

    >>> tprint(1)
    >>> tprint(2)
    2

    """
    global global_tprint_len

    _allowed_r_locs = ['front', 'end']

    if r_loc not in _allowed_r_locs:
        warnings.warn(f'r_loc not in {_allowed_r_locs}, defaulting to {rcParams["tprint.r_loc"]}')
        r_loc = rcParams['tprint.r_loc']

    _string = ''
    _arg_len = 0

    for _arg in args:

        if len(_string) > 0:
            _string += sep
            _arg_len += 1

        _string += str(_arg)
        _arg_len += len(str(_arg))

    # get whitespace len
    _whitespace_len = global_tprint_len - _arg_len

    if _whitespace_len > 0:
        _string += ' ' * _whitespace_len

    # print
    if r_loc == 'front':
        print('\r' + _string, end='', **kwargs)
        # reset tprint
        if len(args) == 0 or (len(args) == 1 and args[0] == ''):
            print('', end='\r', **kwargs)
    else:  # r_loc == 'end'
        print(_string, end='\r', **kwargs)

    # store len for next tprint use
    global_tprint_len = _arg_len


@export
def fprint(*args, file: str = '_fprint.txt', sep: str = ' ', mode: str = 'replace', append_sep: str = '\n',
           timestamp: bool = True, do_print: bool = False, do_tprint: bool = False):
    """
    Write the output of print to a file instead. Supports also writing to console.

    :param args: the arguments to print
    :param file: the name of the file to print to
    :param sep: separator
    :param mode: weather to append or replace the contents of the file
    :param append_sep: if mode=='append', use this separator
    :param timestamp: weather to include a timestamp in the print statement
    :param do_print: weather to also print to console
    :param do_tprint: weather to also print to console using tprint
    :return: None

    **Examples**

    The below output gets written to a file called 'fprint.txt'

    >>> fprint('Hello World', file='fprint.txt')

    The below output gets written both to a file and to console

    >>> fprint('Hello World', file='fprint.txt', do_print=True)
    'Hello World'

    """
    if file[-4:] != '.txt':
        file += '.txt'

    # if append -> get old content ; else start with empty string
    if (mode == 'append') and (os.path.exists(file)):
        with open(file, 'r') as _txt:
            _text = _txt.read()
        _string = _text + append_sep
    else:
        _string = ''

    if timestamp:
        _string += '[{:%Y-%m-%d %H:%M:%S}]: '.format(datetime.datetime.now())

    # args to string
    _print = ''
    _i = -1
    for _arg in args:
        _i += 1
        if _i > 0:
            _print += sep
        _print += str(_arg)

    _string += _print

    # can also print to console
    if do_tprint:
        tprint(_print)
    if do_print:
        print(_print)

    # write to file
    with open(file, 'w') as _txt:
        _txt.write(_string)


@export
def elapsed_time_init() -> None:
    """
    Resets reference time for elapsed_time()

    :return: None

    **Examples**

    see :func:`elapsed_time`
    """
    global global_t
    global_t = datetime.datetime.now()


@export
def elapsed_time(do_return: bool = True, ref_t: datetime.datetime = None) -> datetime.timedelta:
    """
    Get the elapsed time since reference time ref_time.

    :param do_return: Whether to return or print
    :param ref_t: Reference time. If None is provided the time elapsed_time_init() was last called is used.
    :return: In case of do_return: Datetime object containing the elapsed time. Else calls tprint and returns None.

    **Examples**

    >>> from time import sleep
    >>> elapsed_time_init()
    >>> sleep(1)
    >>> elapsed_time(do_return=False)
    '0:00:01.0'

    >>> from time import sleep
    >>> elapsed_time_init()
    >>> sleep(1)
    >>> elapsed_time(do_return=True)
    datetime.timedelta(0, 1, 1345)
    """
    global global_t
    if ref_t is None:
        ref_t = global_t

    _delta_t = datetime.datetime.now() - ref_t

    if do_return:
        return _delta_t
    else:
        tprint(str(_delta_t)[:-5])


@export
def total_time(i: int, i_max: int) -> datetime.timedelta:
    """
    Estimates total time of running operation by linear extrapolation using iteration counters.

    :param i: current iteration
    :param i_max: max iteration
    :return: datetime object representing estimated total time of operation
    """
    _perc_f = i / i_max * 100
    _elapsed_time = elapsed_time(do_return=True)
    _total_time = _elapsed_time * 100 / _perc_f
    return _total_time


@export
def remaining_time(i: int, i_max: int) -> datetime.timedelta:
    """
    Estimates remaining time of running operation by linear extrapolation using iteration counters.

    :param i: current iteration
    :param i_max: max iteration
    :return: datetime object representing estimated remaining time of operation
    """
    _elapsed_time = elapsed_time(do_return=True)
    _total_time = total_time(i, i_max)
    _remaining_time = _total_time - _elapsed_time
    return _remaining_time


@export
def progressbar(i: int = 1, i_max: int = 1, symbol: str = '=', empty_symbol: str = '_', mid: str = None,
                mode: str = 'perc', print_prefix: str = '', p_step: int = 1, printf: Callable = tprint,
                persist: bool = False, **kwargs):
    """
    Prints a progressbar for the currently running process based on iteration counters.

    :param i: current iteration
    :param i_max: max iteration
    :param symbol: symbol that represents reached progress blocks
    :param empty_symbol: symbol that represents not yet reached progress blocks
    :param mid: what to write in the middle of the progressbar, if mid is passed mode is ignored
    :param mode: {'perc', 'total', 'elapsed'}.
        If perc is passed writes percentage. If 'remaining' or 'elapsed' writes remaining or elapsed time respectively.
        [optional]
    :param print_prefix: what to write in front of the progressbar. Useful when calling progressbar multiple times
        from different functions.
    :param p_step: progressbar prints one symbol (progress block) per p_step
    :param printf: Using tprint by default. Use fprint to write to file instead.
    :param persist: Whether to persist the progressbar after reaching 100 percent.
    :param kwargs: Passed to print function
    :return:
    """
    # -- init
    _perc_f = i / i_max * 100
    _perc = int(np.floor(_perc_f))
    _rem = 100 - _perc
    if len(print_prefix) > 0 and (print_prefix[-2:] != ': ') and (print_prefix[-1:] not in [':', '\n']):
        print_prefix += ": "

    # -- main
    if _perc <= 50:

        _right = empty_symbol * (50 // p_step)
        _left = symbol * int(np.ceil(_perc / p_step)) + empty_symbol * ((50 - _perc) // p_step)

    else:

        _left = symbol * (50 // p_step)
        _right = symbol * int(np.ceil(((50 - (100 - _perc)) / p_step))) + empty_symbol * ((100 - _perc) // p_step)

    if mid is not None:
        _mid = mid
    elif mode in ['remaining', 'elapsed']:

        _elapsed_time = elapsed_time(do_return=True)

        # special case for i==0 since we cannot calculate remaining time
        if i == 0:
            _mid = '{}'.format(str(_elapsed_time)[:-5])
        else:

            _total_time = _elapsed_time * 100 / _perc_f
            _remaining_time = _total_time - _elapsed_time

            if i < i_max:

                if mode == 'remaining':
                    _mid = '-{}'.format(str(_remaining_time)[:-5])
                else:
                    _mid = '{} / {}'.format(str(_elapsed_time)[:-5], str(_total_time)[:-5])

            else:
                _mid = '{}'.format(str(_elapsed_time)[:-5])

    elif mode == 'perc':
        _mid = '{:6.2f}%'.format(_perc_f)
    else:
        _mid = ''

    _bar = f"{print_prefix}|{_left}{_mid}{_right}|"

    printf(_bar, **kwargs)

    if persist and i == i_max:
        print('')


@export
def time_to_str(t: datetime.datetime, time_format: str = '%Y-%m-%d') -> str:
    """
    Wrapper for strftime

    :param t: datetime object
    :param time_format: time format, passed to strftime
    :return: formated datetime as string
    """
    return pd.to_datetime(t).strftime(time_format)


@export
def cf_vec(x: Any, func: Callable, to_list: bool = True, *args, **kwargs) -> Any:
    """
    Pandas compatible vectorize function. In case a DataFrame is passed the function is applied to all columns.

    :param x: Any vector like object
    :param func: Any function that should be vectorized
    :param to_list: Whether to cast the output to a list
    :param args: passed to func
    :param kwargs: passed to func
    :return: Vector like object
    """
    # - case: pandas DataFrame
    if isinstance(x, pd.DataFrame):

        _df = x.copy()

        for _col in _df.columns:
            _df[_col] = func(_df[_col], *args, **kwargs)

        return _df

    # - case: numpy array
    _x = np.array(x)
    if _x.shape == ():
        _out = func(x, *args, **kwargs)
    elif (len(_x.shape) == 1) and to_list:
        _out = [func(_x_i, *args, **kwargs) for _x_i in _x]
    else:
        with np.nditer(_x, op_flags=['readwrite']) as _it:
            for _x_i in _it:
                _x_i[...] = func(_x_i, *args, **kwargs)
        _out = _x
    if to_list:
        _out = force_list(_out)
    return _out


@export
def round_signif_i(x: np.number, digits: int = 1) -> float:
    """
    Round to significant number of digits for a Scalar number

    :param x: any number
    :param digits: integer amount of significant digits
    :return: float rounded to significant digits
    """
    if not np.isfinite(x):
        return x
    elif x == 0:
        return 0
    else:
        _scale = -int(np.floor(np.log10(abs(x)))) + digits - 1
        return round(float(x), _scale)


@export
def round_signif(x: Any, *args, **kwargs) -> Any:
    """
    Round to significant number of digits

    :param x: any vector like object of numbers
    :param args: passed to cf_vec
    :param kwargs: passed to cf_vec
    :return: Vector like object of floats rounded to significant digits
    """
    return cf_vec(x, round_signif_i, *args, **kwargs)


@export
def floor_signif(x: Any, digits: int = 1) -> Any:
    """
    Floor to significant number of digits

    :param x: any vector like object of numbers
    :param digits: integer amount of significant digits
    :return: float floored to significant digits
    """
    if x == 0:
        return 0
    else:
        round_signif_x = round_signif(x, digits=digits)
        if round_signif_x <= x:
            return round_signif_x
        else:
            _scale = -int(np.floor(np.log10(abs(x)))) + digits - 1
            return round_signif_x - 1 / np.power(10., _scale)


@export
def ceil_signif(x: Any, digits: int = 1) -> Any:
    """
    Ceil to significant number of digits

    :param x: any vector like object of numbers
    :param digits: integer amount of significant digits
    :return: float ceiled to significant digits
    """
    if x == 0:
        return 0
    else:
        round_signif_x = round_signif(x, digits=digits)
        if round_signif_x >= x:
            return round_signif_x
        else:
            _scale = -int(np.floor(np.log10(abs(x)))) + digits - 1
            return round_signif_x + 1 / np.power(10., _scale)


@export
def concat_cols(df: pd.DataFrame, columns: list, sep: str = '_', to_int: bool = False) -> pd.Series:
    """
    Concat a number of columns of a pandas DataFrame

    :param df: Pandas DataFrame
    :param columns: Names of the columns to be concat
    :param sep: Separator
    :param to_int: If true: Converts columns to int before concatting
    :return: Pandas Series containing the concat columns
    """
    _df = df.copy()
    del df

    _df['_out'] = ''

    for _it, _column in enumerate(force_list(columns)):

        if _it > 0:
            _df['_out'] = _df['_out'] + sep

        _col = _df[_column]

        if to_int:
            _col = _col.round(0).astype(int)

        _df['_out'] = _df['_out'] + _col.astype(str)

    return _df['_out']


@export
def list_unique(lst: Any) -> list:
    """
    Returns unique elements from a list (dropping duplicates)

    :param lst: any list like object
    :return: list containing each element only once
    """
    return list(dict.fromkeys(force_list(lst)))


@export
def list_duplicate(lst: Any) -> list:
    """
    Returns only duplicate elements from a list

    :param lst: any list like object
    :return: list of duplicates values
    """

    _ind = pd.Index(lst)
    return list_unique(_ind[_ind.duplicated()].tolist())


@export
def list_flatten(lst: Any) -> list:
    """
    Flatten a list of lists

    :param lst: list of lists
    :return: flattened list
    """
    return list(np.array(force_list(lst)).flat)


@export
def list_merge(*args: Any, unique: bool = True, flatten: bool = False) -> list:
    """
    Merges n lists together

    :param args: The lists to be merged together
    :param unique: if True then duplicate elements will be dropped
    :param flatten: if True then the individual lists will be flatten before merging
    :return: The merged list
    """
    _list = []

    for _arg in args:

        if _arg is None:
            continue

        if flatten:
            _arg = list_flatten(_arg)
        else:
            _arg = force_list(_arg)

        _list += _arg

    if unique:
        _list = list_unique(_list)

    return _list


@export
def list_intersection(lst: SequenceOrScalar, *args: SequenceOrScalar) -> list:
    """
    Returns common elements of n lists

    :param lst: the first list
    :param args: the subsequent lists
    :return: the list of common elements
    """
    # more performant than list comprehension
    _list_out = force_list(lst)

    for _arg in args:
        _list = force_list(_arg)
        _list_out = list(set(_list_out).intersection(_list))

    return _list_out


@export
def list_exclude(lst: SequenceOrScalar, *args: SequenceOrScalar) -> list:
    """
    Returns a list that includes only those elements from the first list that are not in any subsequent list.
    Can also be called with non list args, then those elements are removed.

    :param lst: the list to exclude from
    :param args: the subsequent lists
    :return: the filtered list
    """
    # more performant than list comprehension
    _list_out = force_list(lst)

    for _arg in args:
        try:
            if _arg in _list_out:
                _list_out.remove(_arg)
        except Exception as _e:  # sometimes causes errors when comparing multi objects
            _ = _e
            pass
        for _el in force_list(_arg):
            if _el in _list_out:
                _list_out.remove(_el)

    return _list_out


@export
def rand(shape: tuple = None, lower: int = None, upper: int = None, step: int = None, seed: int = None) -> np.array:
    """
    A seedable wrapper for numpy.random.random_sample that allows for boundaries and steps

    :param shape: A tuple containing the shape of the desired output array
    :param lower: Lower bound of random numbers
    :param upper: Upper bound of random numbers
    :param step: Minimum step between random numbers
    :param seed: Random Seed
    :return: Numpy Array
    """
    # seed
    if seed is not None:
        np.random.seed(seed)

    # create base random numbers (between 0 and 1)
    _rand = np.random.random_sample(shape)

    # default values
    if lower is None:
        lower = 0
    if upper is None:
        upper = lower + 1

    _samples = _rand * (upper - lower) + lower

    # apply step
    if step is not None:

        _samples = np.round(_samples / step) * step

        # if step is integer: return integers
        if isinstance(step, int):
            _samples = _samples.astype(int)

    return _samples


@export
def dict_list(*args, dict_type: str = 'defaultdict') -> dict:
    """
    Creates a dictionary of empty named lists. Useful for iteratively creating a pandas DataFrame

    :param args: The names of the lists
    :param dict_type: Whether to use a 'regular' or 'defaultdict' (default to empty list) type dictionary
    :return: Dictionary of empty named lists
    """
    if dict_type == 'regular':
        _dict = {}
    else:
        _dict = defaultdict(list)

    for _arg in args:
        for _list in force_list(_arg):
            _dict[_list] = []

    return _dict


@export
def append_to_dict_list(dct: Union[dict, defaultdict], append: Union[dict, list],
                        inplace: bool = True) -> Optional[dict]:
    """
    Appends to a dictionary of named lists. Useful for iteratively creating a pandas DataFrame.

    :param dct: Dictionary to append to
    :param append: List or dictionary of values to append
    :param inplace: Modify inplace or return modified copy
    :return: None if inplace, else modified dictionary
    """
    if not inplace:
        dct = dct.copy()

    # allows lists and dicts
    if not isinstance(append, Mapping):

        if is_list_like(append):
            _append = list(append)
        else:
            _append = [append]

        if len(_append) > len(dct):
            warnings.warn('list is longer than dict, trailing entries will be lost')
        _append = dict(zip(dct.keys(), _append))

    else:
        _append = append

    for _key in _append.keys():
        dct[_key].append(_append[_key])

    if not inplace:
        return dct


@export
def is_scalar(obj: Any) -> bool:
    """
    Checks if a given python object is scalar, i.e. one of int, float, str, bytes

    :param obj: Any python object
    :return: True if scaler, else False
    """

    return isinstance(obj, Scalar.__args__)


@export
def is_list_like(obj: Any) -> bool:
    """
    Checks if a given python object is list like. The conditions must be satisfied:

        * not a string or bytes object

        * one of (Sequence, 1d-array like Iterable)


    :param obj: Any python object
    :return: True if list like, else False
    """
    # str, bytes
    if isinstance(obj, (str, bytes)):
        return False
    # Sequence and similar
    if isinstance(obj, (Sequence, {}.keys().__class__, {}.values().__class__, pd.Index)):
        return True
    # Iterable
    if isinstance(obj, Iterable):
        # check if the first element of the cast list is different from the object itself (object is castable to list)
        try:  # try is needed because pandas objects return a sequence for != operator
            if list(obj)[0] != obj:
                return True
        except (ValueError, IndexError):
            pass
        # check if the object is array like
        _shape = np.array(obj).shape
        # 1d arrays are list like
        if len(_shape) == 1:
            return True
        elif len(_shape) == 2:
            # 2d arrays are list like if the 2nd dimension contains only one entry (e.g. single column DataFrame)
            if _shape[1] == 1:
                return True
    # Other
    return False


@export
def force_list(*args: Any) -> list:
    """
    Takes any python objects and turns them into an iterable list.

    :param args: Any python object
    :return: List
    """
    args = list(args)

    # Empty case
    if len(args) == 0:
        return []

    # None case
    if len(args) == 1:
        if args[0] is None:
            return []

    # Regular case
    for _it, _arg in enumerate(args):
        if is_list_like(_arg):
            # require direct casts
            if isinstance(_arg, (Sequence, {}.keys().__class__, {}.values().__class__, pd.Index)):
                _arg = list(_arg)
            elif isinstance(_arg, Iterable):
                # not all iterables implement list() in the same way -> cast to np.array and flatten
                _arg = list(np.array(_arg).flatten())
            else:  # other cases: direct cast
                _arg = list(_arg)
        else:
            _arg = [_arg]
        args[_it] = _arg

    # depending on whether just one argument was passed or list of arguments we need to return differently
    if len(args) == 1:
        args = args[0]
    else:
        args = tuple(args)

    return args


@export
def force_scalar(obj: Any, warn: bool = True) -> Scalar:
    """
    Takes any python object and turns it into a scalar object.

    :param obj: Any python object
    :param warn: Whether to trigger a warning when objects are being truncated
    :return: List
    """
    _lst = force_list(obj)
    _len = len(_lst)
    if warn and _len > 1:
        warnings.warn(f"force scalar: object {obj} has length {_len}, retaining only first entry")

    return _lst[0]


@export
def qformat(value: Any, int_format: str = ',', float_format: str = ',.2f', datetime_format: str = '%Y-%m-%d',
            sep: str = ' - ', key_sep: str = ': ', print_key: bool = True) -> str:
    """
    Creates a human readable representation of a generic python object

    :param value: Any python object
    :param int_format: Format string for integer
    :param float_format: Format string for float
    :param datetime_format: Format string for datetime
    :param sep: Separator
    :param key_sep: Separator used between key and value if print_key is True
    :param print_key: Whether to print keys as well as values (if object has keys)
    :return: Formated string
    """

    def _qformat(_value_i: Any) -> str:

        if is_list_like(_value_i):
            _value_i = str(_value_i)
        if isinstance(_value_i, str):
            _value_i = _value_i  # do nothing
        elif isinstance(_value_i, datetime.datetime):
            _value_i = _value_i.strftime(datetime_format)
        elif isinstance(_value_i, int):
            _value_i = format(_value_i, int_format)
        elif isinstance(_value_i, float):
            if _value_i.is_integer():
                _value_i = format(int(_value_i), int_format)
            else:
                _value_i = format(_value_i, float_format)
        else:
            _value_i = str(_value_i)

        return _value_i

    _string = ''

    if isinstance(value, Mapping):

        for _key, _value in value.items():

            _formated_value = _qformat(_value)

            if len(_string) > 0:
                _string += sep

            if print_key:
                _string += '{}{}{}'.format(_key, key_sep, _formated_value)
            else:
                _string += _formated_value

    elif is_list_like(value):

        for _value in value:
            if len(_string) > 0:
                _string += sep
            _string += _qformat(_value)

    else:

        _string += _qformat(value)

    return _string


@export
def to_hdf(df: pd.DataFrame, file: str, groupby: Union[str, List[str]] = None, key: str = None, replace: bool = False,
           do_print=True, **kwargs) -> None:
    """
    saves a pandas DataFrame as h5 file, if groupby is supplied will save each group with a different key.
    Needs with groupby OR key to be supplied. Extends on pandas.DataFrame.to_hdf.

    :param df: DataFrame to save
    :param file: filename to save the DataFrame as
    :param groupby: if supplied will save each sub-DataFrame as a different key. [optional]
    :param key: The key to write as. Ignored if groupby is supplies.
    :param replace: Whether to replace or append to existing files. Defaults to append. [optional]
    :param do_print: Whether to print intermediate steps to console [optional]
    :param kwargs: Other keyword arguments passed to pd.DataFrame.to_hdf [optional]
    :return: None
    """
    assert (groupby is not None) or (key is not None), "You must supply either groupby or key"

    # -- init
    # - no inplace
    df = pd.DataFrame(df).copy()
    # - defaults
    # groupby
    if groupby is None:
        groupby = GROUPBY_DUMMY
        df[groupby] = 1

    # -- main
    # remove old file
    if replace and os.path.exists(file):
        os.remove(file)
        if do_print:
            tprint()
            print('removed old {}'.format(file))

    _i_max = df[groupby].drop_duplicates().shape[0]

    for _it, (_index, _df_i) in enumerate(df.groupby(groupby)):

        if key is None:
            _key = qformat(_index, as_string=True)
        else:
            _key = key

        if do_print:
            progressbar(_it, _i_max, print_prefix=f"writing key {_key:<30}: ")

        if GROUPBY_DUMMY in _df_i.columns:
            _df_i = _df_i.drop(GROUPBY_DUMMY, axis=1)

        pd.DataFrame.to_hdf(_df_i, file, key=_key, format='table', **kwargs)

    if do_print:
        tprint()
        print('{}saved to {}'.format('\n', file))


@export
def get_hdf_keys(file: str) -> List[str]:
    """
    Reads all keys from an hdf file and returns as list

    :param file: The path of the file to read the keys of
    :return: List of keys
    """

    with h5py.File(file, 'r') as _file:
        _keys = list(_file.keys())
        _file.close()

    return _keys


@export
def read_hdf(file: str, key: Union[str, List[str]] = None, sample: int = None, random_state: int = None,
             do_print: bool = True, catch_error: bool = True) -> pd.DataFrame:
    """
    read a DataFrame from hdf file

    :param file: The path to the file to read from
    :param key: The key(s) to read, if not specified all keys are read [optional]
    :param sample: If specified will read sample keys at random from the file, ignored if key is specified [optional]
    :param random_state: Random state for sample [optional]
    :param do_print: Whether to print intermediate steps [optional]
    :param catch_error: Whether to catch errors when reading [optional]
    :return: pandas DataFrame
    """

    if not os.path.exists(file): 
        raise ValueError('{} does not exist'.format(file))

    # if key was not specified: read all keys
    if key is None:
        _keys = get_hdf_keys(file)
        _read_keys = 'all'
        if sample is not None:
            np.random.seed(random_state)
            _keys = np.random.sample(_keys, sample)
            _read_keys = ','.join(_keys)
    else:
        if not isinstance(key, list):
            _keys = [key]
        else:
            _keys = key
        _read_keys = ','.join(_keys)

    _df = []

    _i = 0

    for _key in _keys:

        _i += 1

        if do_print:
            tprint('reading {} - key {} / {} : {}...'.format(file, _i, len(_keys), _key))
        if catch_error:
            try:
                _df.append(pd.read_hdf(file, key=_key))
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as _e:
                tprint('')
                print(f"{_e.__class__.__name__}: '{_e}' while reading key {_key}")
        else:
            _df.append(pd.read_hdf(file, key=_key))

    if do_print:
        tprint('concat...')
    try:
        _df = pd.concat(_df, ignore_index=True, sort=False)
    except Exception as _e:
        tprint('')
        print(f"{_e.__class__.__name__}: {_e} during pandas.concat")

    if do_print:
        tprint('read {} ; keys: {}'.format(file, _read_keys))
        print('')

    return _df


@export
def rounddown(x: Any, digits: int) -> Any:
    """
    convenience wrapper for np.floor with digits option

    :param x: any python object that supports np.floor
    :param digits: amount of digits
    :return: rounded x
    """

    return np.floor(x * 10**digits) / 10**digits


@export
def roundup(x: Any, digits: int) -> Any:
    """
    convenience wrapper for np.ceil with digits option

    :param x: any python object that supports np.ceil
    :param digits: amount of digits
    :return: rounded x
    """

    return np.ceil(x * 10**digits) / 10**digits


@docstr
@export
def reformat_string(string: str, case: Optional[str] = 'lower', replace: Optional[Mapping[str, str]] = None,
                    lstrip: Optional[str] = ' ', rstrip: Optional[str] = ' ', demojize: bool = True,
                    trans: bool = False, trans_dest: Optional[str] = 'en', trans_src: Optional[str] = 'auto',
                    trans_sleep: Union[float, bool] = .4, warn: bool = True) -> str:
    """
    Function to quickly reformat a string to a specific convention. The default convention is only lowercase,
    numbers and underscores. Also allows translation if optional dependency googletrans is installed.

    :param string: input string to be reformatted
    :param case: casts string to specified case, one of %(reformat_string__case)s [optional]
    :param replace: Dictionary containing the replacements to be made passed to
        `re.sub <https://docs.python.org/3/library/re.html>`_ . Defaults to replacing any non [a-zA-Z0-9]
        string with '_'. Note that this means that special characters from other languages get replaced. If you don't
        want that set replace to False or specify your own mapping. Is applied **last** so make sure your
        conventions match [optional]
    :param lstrip: The leading characters to be removed, passed to
        `string.lstrip <https://docs.python.org/3/library/stdtypes.html>`_ [optional]
    :param rstrip: The training characters to be removed, passed to
        `string.rstrip <https://docs.python.org/3/library/stdtypes.html>`_ [optional]
    :param demojize: Whether to remove emojis using `emoji.demojize <https://pypi.org/project/emoji/>`_ [optional]
    :param trans: Whether to translate the string using
        `googletrans.Translator.translate <https://py-googletrans.readthedocs.io/en/latest/#googletrans-translator>`_
        [optional]
    :param trans_dest: The language to translate from, passed to googletrans as dest=trans_dest [optional]
    :param trans_src: The language to translate to, passed to googletrans as src=trans_src [optional]
    :param trans_sleep: Amount of seconds to sleep before translating, should be at least .4 to avoid triggering
        google's rate limits. Set it to lower values / None / False for a speedup at your own risk [optional]
    :param warn: %(warn)s
    :return: reformatted string
    """
    # -- init
    if replace is None:
        replace = {'[^A-Za-z0-9]': '_'}

    # implicitly cast to string
    string = str(string)

    # -- demojize: (needs to come before trans)
    if demojize:
        if emoji:
            string = emoji.demojize(string)
        else:
            warnings.warn('Missing optional dependency emoji, skipping demojize')

    # -- trans: (needs to come after demojize but before the rest)
    if trans:
        assert Translator, 'Missing optional dependency googletrans'
        _translator = Translator()
        try:
            # avoid rate limits
            if trans_sleep:
                sleep(trans_sleep)
            # translate
            string = _translator.translate(string, dest=trans_dest, src=trans_src).text
        except JSONDecodeError:
            if warn:
                warnings.warn(f'handled JSONDecodeError at {string}, this probably means that you exceeded the '
                              f'googletrans rate limit and need to wait 24 hours.')
        except Exception as _exc:
            if warn:
                warnings.warn(f'handled exception " {type(_exc).__name__,}: {_exc}" when translating {string}, '
                              f'skipping translation')

    # -- case
    if case:
        if case == 'lower':
            string = string.lower()
        elif case == 'upper':
            string = string.upper()
        else:
            if warn:
                warnings.warn(f'ignoring unknown case {case}')

    # -- lstrip
    if lstrip:
        string = string.lstrip(lstrip)

    # -- rstrip
    if rstrip:
        string = string.rstrip(rstrip)

    # -- replace (comes last therefore replacement rules must be defined accordingly)
    for _exp, _replacement in replace.items():
        string = re.sub(_exp, _replacement, string)

    return string


@export
def dict_inv(dct: Mapping, key_as_str: bool = False, duplicates: str = 'keep') -> dict:
    """
    Returns an inverted copy of a given dictionary (if it is invertible)

    :param dct: Dictionary to be inverted
    :param key_as_str: Whether all keys of the inverted dictionary should be forced to string
    :param duplicates: Whether to 'adjust' or 'drop' duplicates. In case of 'adjust' duplicates are suffixed with '_'
    :return: Inverted dictionary
    """

    # -- assert
    if duplicates not in validations['dict_inv__duplicates']:
        raise ValueError(f"duplicates must be one of {validations['dict_inv__duplicates']}")
    # -- init
    _dct_inv = {}
    # -- main
    for _key, _value in dct.items():
        # assert scalar
        if not is_scalar(_value):
            raise ValueError(f'A non-scalar dictionary value is not invertible, found at key {_key}')
        # assert non-duplicate value
        if duplicates == 'adjust':
            _warn = True
            while _value in _dct_inv.keys():
                if _warn:
                    _warn = False
                    warnings.warn(f'duplicate value found at "{_key}: {_value}", appending _')
                _value = str(_value) + '_'
        elif (duplicates == 'drop') and (_value in _dct_inv.keys()):
            continue
        # if applicable: convert value to string
        if key_as_str:
            _value = str(_value)
        # assign
        _dct_inv[_value] = _key

    return _dct_inv
