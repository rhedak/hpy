"""
hpy.main.py
~~~~~~~~~~~~~~~~

Contains basic calculation functions that are used in the more specialized versions of the package but can also be used
on their own

"""

# standard library imports
import numpy as np
import pandas as pd
import warnings
import collections
import os
import datetime

# third party imports
from typing import Any, AnyStr, Callable, Union

# --- init
pd.plotting.register_matplotlib_converters()
pd.options.mode.chained_assignment = None

# --- classes
global_t = datetime.datetime.now()  # for times progress bar
global_tprint_len = 0  # for temporary printing


# ----------------- classes and operators ------------------------

class Infix:
    """Class for representing the pipe operator |__|
        The operator is based on the r %>% operator

    Parameters
    ----------

    Returns
    -------

    """
    def __init__(self, function):
        self.__name__ = 'Infix'
        self.function = function

    def __ror__(self, other):
        return Infix(lambda _x, _self=self, _other=other: self.function(_other, _x))

    def __or__(self, other):
        return self.function(other)

    def __rlshift__(self, other):
        return Infix(lambda _x, _self=self, _other=other: self.function(_other, _x))

    def __rshift__(self, other):
        return self.function(other)

    def __call__(self, value1, value2):
        return self.function(value1, value2)


# pipe operator
__ = Infix(lambda _o, _f: _f(_o))


# --- functions
def today(date_format: str = '%Y_%m_%d') -> str:
    """Returns today's date as string

    Parameters
    ----------
    date_format :
        The formating string for the date. Passed to strftime
    date_format: str :
         (Default value = '%Y_%m_%d')

    Returns
    -------
    type
        Formated String

    """
    return datetime.datetime.today().strftime(date_format)


# for pipe operator:
def round1(x: float) -> float:
    """Wrapper for np.round with default digits 1

    Parameters
    ----------
    x :
        number to be rounded
    x: float :
        

    Returns
    -------
    type
        float

    """
    return np.round(x, 1)


def round2(x: float) -> float:
    """Wrapper for np.round with default digits 2

    Parameters
    ----------
    x :
        number to be rounded
    x: float :
        

    Returns
    -------
    type
        float

    """
    return np.round(x, 2)


# format bytes
def size(byte: int, unit: str = 'MB', dec: int = 2) -> str:
    """Formats bytes as human readable string

    Parameters
    ----------
    byte :
        The byte amount to be formated
    unit :
        The unit to display the output in, supports 'KB', 'MB', 'GB' and 'TB'
    dec :
        The number of decimals to use
    byte: int :
        
    unit: str :
         (Default value = 'MB')
    dec: int :
         (Default value = 2)

    Returns
    -------
    type
        Formated bytes as string

    """
    _power = {'KB': 1, 'MB': 2, 'GB': 3, 'TB': 4}[unit]

    return '{} {}'.format(np.round(byte / (1024 ** _power), dec), unit)


# return memory usage more detailed than default
def mem_usage(pandas_obj, *args, **kwargs) -> str:
    """Get memory usage of a pandas object

    Parameters
    ----------
    pandas_obj :
        Pandas object to get the memory usage of
    args :
        passed to size()
    kwargs :
        passed to size()
    *args :
        
    **kwargs :
        

    Returns
    -------
    type
        memory usage of a pandas object formated as string

    """
    if isinstance(pandas_obj, pd.DataFrame):
        _usage_b = pandas_obj.memory_usage(deep=True).sum()

    else:  # we assume if not a df it's a series
        _usage_b = pandas_obj.memory_usage(deep=True)

    return size(_usage_b, *args, **kwargs)


# print temporary text that will be overwritten by the next print
def tprint(*args, sep=' ', **kwargs):
    """Wrapper for print() but with a carriage return at the end.
    This results in the text being overwritten by the next print call.
    Can be used for progress bars and the like.

    Parameters
    ----------
    args : arguments to print
        
    sep : separator
         (Default value = ' ')
    kwargs : passed to print
        
    *args :
        
    **kwargs :
        

    Returns
    -------
    None
        

    Examples
    --------
    >>>tprint('Hello World')
    'Hello World'
    """
    global global_tprint_len

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
    print(_string, end='\r', **kwargs)

    # store len for next tprint use
    global_tprint_len = _arg_len


# print to file
def fprint(*args, file: str = '_print.txt', sep: str = ' ', mode: str = 'replace', append_sep: str = '\n',
           timestamp: bool = True, do_print: bool = False, do_tprint: bool = False):
    """Write the output of print to a file instead. Supports also writing to console.

    Parameters
    ----------
    args :
        the arguments to print
    file :
        the name of the file to print to
    sep :
        separator
    mode :
        weather to append or replace the contents of the file
    append_sep :
        if mode=='append', use this separator
    timestamp :
        weather to include a timestamp in the print statement
    do_print :
        weather to also print to console
    do_tprint :
        weather to also print to console using tprint
    *args :
        
    file: str :
         (Default value = '_print.txt')
    sep: str :
         (Default value = ' ')
    mode: str :
         (Default value = 'replace')
    append_sep: str :
         (Default value = '\n')
    timestamp: bool :
         (Default value = True)
    do_print: bool :
         (Default value = False)
    do_tprint: bool :
         (Default value = False)

    Returns
    -------
    type
        None

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


def total_time(i: int, i_max: int) -> datetime.timedelta:
    """Estimates total time of running operation by linear extrapolation using iteration counters.

    Parameters
    ----------
    i :
        current iteration
    i_max :
        max iteration
    i: int :
        
    i_max: int :
        

    Returns
    -------
    type
        datetime object representing estimated total time of operation

    """
    _perc_f = i / i_max * 100
    _elapsed_time = elapsed_time(do_return=True)
    _total_time = _elapsed_time * 100 / _perc_f
    return _total_time


def remaining_time(i: int, i_max: int) -> datetime.timedelta:
    """Estimates remaining time of running operation by linear extrapolation using iteration counters.

    Parameters
    ----------
    i :
        current iteration
    i_max :
        max iteration
    i: int :
        
    i_max: int :
        

    Returns
    -------
    type
        datetime object representing estimated remaining time of operation

    """
    _elapsed_time = elapsed_time(do_return=True)
    _total_time = total_time(i, i_max)
    _remaining_time = _total_time - _elapsed_time
    return _remaining_time


def progressbar(i: int = 1, i_max: int = 1, symbol: str = '=', mid: str = None, mode: str = 'perc',
                print_prefix: str = '', p_step: int = 1, printf: Callable = tprint, persist: bool = False, **kwargs):
    """Prints a progressbar for the currently running process based on iteration counters.

    Parameters
    ----------
    i :
        current iteration
    i_max :
        max iteration
    symbol :
        symbol that represents progress percentage
    mid :
        what to write in the middle of the progressbar, if mid is passed mode is ignored
    mode :
        perc', 'total', 'elapsed'}.
        If perc is passed writes percentage. If 'remaining' or 'elapsed' writes remaining or elapsed time respectively.
        [optional]
    print_prefix :
        what to write in front of the progressbar. Useful when calling progressbar multiple times
        from different functions.
    p_step :
        progressbar prints one symbol per p_step
    printf :
        Using tprint by default. Use fprint to write to file instead.
    persist :
        Whether to persist the progressbar after reaching 100 percent.
    kwargs :
        Passed to print function
    i: int :
         (Default value = 1)
    i_max: int :
         (Default value = 1)
    symbol: str :
         (Default value = '=')
    mid: str :
         (Default value = None)
    mode: str :
         (Default value = 'perc')
    print_prefix: str :
         (Default value = '')
    p_step: int :
         (Default value = 1)
    printf: Callable :
         (Default value = tprint)
    persist: bool :
         (Default value = False)
    **kwargs :
        

    Returns
    -------

    """
    # uses tprint by default, pass fprint to write to file

    # if mid is passed mode is ignored

    # mode can be 'perc', 'remaining' or 'elapsed'
    # anything else, e.g. '', leads to an empty middle

    _perc_f = i / i_max * 100
    _perc = int(np.floor(_perc_f))
    _rem = 100 - _perc

    if _perc <= 50:

        _right = ' ' * (50 // p_step)
        _left = symbol * int(np.ceil(_perc / p_step)) + ' ' * ((50 - _perc) // p_step)

    else:

        _left = symbol * (50 // p_step)
        _right = symbol * int(np.ceil(((50 - (100 - _perc)) / p_step))) + ' ' * ((100 - _perc) // p_step)

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

    _bar = '|{}{}{}{}|'.format(_left, print_prefix, _mid, _right)

    printf(_bar, **kwargs)

    if persist and i == i_max:
        print('')


def elapsed_time_init():
    """Resets reference time for elapsed_time()
    
    :return: None

    Parameters
    ----------

    Returns
    -------

    """
    global global_t
    global_t = datetime.datetime.now()


def elapsed_time(do_return: bool = False, ref_t: datetime.datetime = None) -> datetime.timedelta:
    """Get the elapsed time since reference time ref_time.

    Parameters
    ----------
    do_return :
        Whether to return or print
    ref_t :
        Reference time. If None is provided the time elapsed_time_init() was last called is used.
    do_return: bool :
         (Default value = False)
    ref_t: datetime.datetime :
         (Default value = None)

    Returns
    -------
    type
        In case of do_return: Datetime object containing the elapsed time. Else None.

    """
    global global_t
    if ref_t is None:
        ref_t = global_t

    _delta_t = datetime.datetime.now() - ref_t

    if do_return:
        return _delta_t
    else:
        tprint(str(_delta_t)[:-5])


def time_to_str(t: datetime.datetime, time_format: str = '%Y-%m-%d') -> str:
    """Wrapper for strftime

    Parameters
    ----------
    t :
        datetime object
    time_format :
        time format, passed to strftime
    t: datetime.datetime :
        
    time_format: str :
         (Default value = '%Y-%m-%d')

    Returns
    -------
    type
        formated datetime as string

    """
    return pd.to_datetime(t).strftime(time_format)


# custom vectorize function (only works for 1 parameter)
def cf_vec(x: Any, func: Callable, *args, **kwargs) -> Any:
    """Pandas compatible vectorize function. In case a DataFrame is passed the function is applied to all columns.

    Parameters
    ----------
    x :
        Any vector like object
    func :
        Any function that should be vectorized
    args :
        passed to func
    kwargs :
        passed to func
    x: Any :
        
    func: Callable :
        
    *args :
        
    **kwargs :
        

    Returns
    -------
    type
        Vector like object

    """
    # df

    if isinstance(x, pd.DataFrame):

        _df = x.copy()

        for _col in _df.columns:
            _df[_col] = func(_df[_col], *args, **kwargs)

        return _df

    # generic

    _x = np.array(x)

    if _x.shape == ():
        _out = func(_x, *args, **kwargs)
    elif len(_x.shape) == 1:
        _out = np.array([func(_x_i, *args, **kwargs) for _x_i in _x])
    else:
        with np.nditer(_x, op_flags=['readwrite']) as _it:
            for _x_i in _it:
                _x_i[...] = func(_x_i, *args, **kwargs)

        _out = _x

    _out = force_list(_out)

    return _out


def round_signif_i(x: np.number, digits: int = 1) -> float:
    """Round to significant number of digits

    Parameters
    ----------
    x :
        any number
    digits :
        integer amount of significant digits
    x: np.number :
        
    digits: int :
         (Default value = 1)

    Returns
    -------
    type
        float rounded to significant digits

    """
    if not np.isfinite(x):
        return x
    elif x == 0:
        return 0
    else:
        _scale = -int(np.floor(np.log10(abs(x)))) + digits - 1
        return round(float(x), _scale)


def round_signif(x: Any, *args, **kwargs) -> Any:
    """Round to significant number of digits

    Parameters
    ----------
    x :
        any vector like object of numbers
    args :
        passed to cf_vec
    kwargs :
        passed to cf_vec
    x: Any :
        
    *args :
        
    **kwargs :
        

    Returns
    -------
    type
        Vector like object of floats rounded to significant digits

    """
    return cf_vec(x, round_signif_i, *args, **kwargs)


def floor_signif(x: Any, digits: int = 1) -> Any:
    """Floor to significant number of digits

    Parameters
    ----------
    x :
        any vector like object of numbers
    digits :
        integer amount of significant digits
    x: Any :
        
    digits: int :
         (Default value = 1)

    Returns
    -------
    type
        float floored to significant digits

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


def ceil_signif(x: Any, digits: int = 1) -> Any:
    """Ceil to significant number of digits

    Parameters
    ----------
    x :
        any vector like object of numbers
    digits :
        integer amount of significant digits
    x: Any :
        
    digits: int :
         (Default value = 1)

    Returns
    -------
    type
        float ceiled to significant digits

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


def concat_cols(df: pd.DataFrame, columns: list, sep: str = '_', to_int: bool = False) -> pd.Series:
    """Concat a number of columns of a pandas DataFrame

    Parameters
    ----------
    df :
        Pandas DataFrame
    columns :
        Names of the columns to be concat
    sep :
        Separator
    to_int :
        If true: Converts columns to int before concatting
    df: pd.DataFrame :
        
    columns: list :
        
    sep: str :
         (Default value = '_')
    to_int: bool :
         (Default value = False)

    Returns
    -------
    type
        Pandas Series containing the concat columns

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


def list_unique(lst: list) -> list:
    """Returns unique elements from a list

    Parameters
    ----------
    lst :
        any list like object
    lst: list :
        

    Returns
    -------
    type
        a list

    """
    return list(dict.fromkeys(force_list(lst)))


def list_flatten(lst: list) -> list:
    """Flatten a list of lists

    Parameters
    ----------
    lst :
        list of lists
    lst: list :
        

    Returns
    -------
    type
        flattened list

    """
    return np.array(force_list(lst)).flat


def list_merge(*args, unique=True, flatten=False) -> list:
    """Merges n lists together

    Parameters
    ----------
    args :
        The lists to be merged together
    unique :
        if True then duplicate elements will be dropped (Default value = True)
    flatten :
        if True then the individual lists will be flatten before merging (Default value = False)
    *args :
        

    Returns
    -------
    type
        The merged list

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


def list_intersection(lst: list, *args) -> list:
    """Returns common elements of n lists

    Parameters
    ----------
    lst :
        the first list
    args :
        the subsequent lists
    lst: list :
        
    *args :
        

    Returns
    -------
    type
        the list of common elements

    """
    # more performant than list comprehension
    _list_out = list(lst)

    for _arg in args:
        _list = list(_arg)
        _list_out = list(set(_list_out).intersection(_list))

    return _list_out


def rand(shape: tuple = None, lower: int = None, upper: int = None, step: int = None, seed: int = None) -> np.array:
    """A seedable wrapper for numpy.random.random_sample that allows for boundaries and steps

    Parameters
    ----------
    shape :
        A tuple containing the shape of the desired output array
    lower :
        Lower bound of random numbers
    upper :
        Upper bound of random numbers
    step :
        Minimum step between random numbers
    seed :
        Random Seed
    shape: tuple :
         (Default value = None)
    lower: int :
         (Default value = None)
    upper: int :
         (Default value = None)
    step: int :
         (Default value = None)
    seed: int :
         (Default value = None)

    Returns
    -------
    type
        Numpy Array

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


def dict_list(*args) -> dict:
    """Creates a dictionary of empty named lists. Useful for iteratively creating a pandas DataFrame

    Parameters
    ----------
    args :
        The names of the lists
    *args :
        

    Returns
    -------
    type
        Dictionary of empty named lists

    """
    _dict = {}

    for _arg in args:
        for _list in force_list(_arg):
            _dict[_list] = []

    return _dict


def append_to_dict_list(dct: dict, append: Union[dict, list], inplace: bool = True) -> Union[dict, None]:
    """Appends to a dictionary of named lists. Useful for iteratively creating a pandas DataFrame.

    Parameters
    ----------
    dct :
        dictionary to append to
    append :
        List of Dictionary of values to append
    inplace :
        Modify inplace or return modified copy
    dct: dict :
        
    append: Union[dict :
        
    list] :
        
    inplace: bool :
         (Default value = True)

    Returns
    -------
    type
        None if inplace, else modified dictionary

    """
    if inplace:
        _dic = dct
    else:
        _dic = dct.copy()

    # allows lists and dicts
    if not isinstance(append, collections.Mapping):

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
        _dic[_key].append(_append[_key])

    if not inplace:
        return _dic


def is_list_like(obj: Any) -> bool:
    """Checks any python object to see if it is list like

    Parameters
    ----------
    obj :
        Any python object
    obj: Any :
        

    Returns
    -------
    type
        Boolean

    """
    return isinstance(obj, collections.Iterable) and not isinstance(obj, (str, bytes))


def force_list(*args) -> list:
    """Takes any python object and turns it into an iterable list.

    Parameters
    ----------
    args :
        Any python object
    *args :
        

    Returns
    -------
    type
        List

    """
    args = list(args)

    # None case
    if len(args) == 1:
        if args[0] is None:
            return None

    _i = -1
    for _arg in args:

        _i += 1

        if is_list_like(_arg):
            _arg = list(_arg)
        else:
            _arg = [_arg]

        args[_i] = _arg

    # depending on weather just one argument was passed or list of arguments we need to return differently
    if len(args) == 1:
        args = args[0]
    else:
        args = tuple(args)

    return args


def qformat(value: Any, int_format: str = ',', float_format: str = ',.2f', datetime_format: str = '%Y-%m-%d',
            sep: str = ' - ', key_sep: str = ': ', print_key: bool = True) -> str:
    """Creates a human readable representation of a generic python object

    Parameters
    ----------
    value :
        Any python object
    int_format :
        Format string for integer
    float_format :
        Format string for float
    datetime_format :
        Format string for datetime
    sep :
        Separator
    key_sep :
        Separator used between key and value if print_key is True
    print_key :
        weather to print keys as well as values (if object has keys)
    value: Any :
        
    int_format: str :
         (Default value = ')
    ' :
        
    float_format: str :
         (Default value = ')
    .2f' :
        
    datetime_format: str :
         (Default value = '%Y-%m-%d')
    sep: str :
         (Default value = ' - ')
    key_sep: str :
         (Default value = ': ')
    print_key: bool :
         (Default value = True)

    Returns
    -------
    type
        Formated string

    """

    def _qformat(_value_i: Any) -> str:
        """

        Parameters
        ----------
        _value_i: Any :
            

        Returns
        -------

        """

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

    if isinstance(value, collections.Mapping):

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


def tdelta(*args, **kwargs) -> datetime.timedelta:
    """Wrapper for numpy.timedelta64

    Parameters
    ----------
    args :
        passed to numpy.timedelta64
    kwargs :
        passed to numpy.timedelta64
    *args :
        
    **kwargs :
        

    Returns
    -------
    type
        Timedelta object

    """
    return np.timedelta64(*args, **kwargs)
