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


# --- init
pd.plotting.register_matplotlib_converters()
pd.options.mode.chained_assignment = None

# --- classes
global_t = datetime.datetime.now()  # for times progress bar
global_tprint_len = 0  # for temporary printing


# ----------------- classes and operators ------------------------

class Infix:
    """
        Class for representing the pipe operator |__|
        The operator is based on the r %>% operator
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
def today(date_format='%Y_%m_%d'):
    return datetime.datetime.today().strftime(date_format)


# for pipe operator:
def round1(x: float) -> float:
    return np.round(x, 1)


def round2(x: float) -> float:
    return np.round(x, 2)


# format bytes
def size(byte, unit='MB', dec=2):
    _power = {'KB': 1, 'MB': 2, 'GB': 3, 'TB': 4}[unit]

    return '{} {}'.format(np.round(byte / (1024 ** _power), dec), unit)


# return memory usage more detailed than default
def mem_usage(pandas_obj, *args, **kwargs):
    if isinstance(pandas_obj, pd.DataFrame):
        _usage_b = pandas_obj.memory_usage(deep=True).sum()

    else:  # we assume if not a df it's a series
        _usage_b = pandas_obj.memory_usage(deep=True)

    return size(_usage_b, *args, **kwargs)


# print temporary text that will be overwritten by the next print
def tprint(*args, sep=' ', **kwargs):
    global global_tprint_len

    # print(_whitespace_len)
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
def fprint(*args, file='_print.txt', sep=' ', mode='replace', append_sep='\n', timestamp=True, do_print=False,
           do_tprint=False):
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


def total_time(i, i_max):
    _perc_f = i / i_max * 100
    _elapsed_time = elapsed_time(do_return=True)
    _total_time = _elapsed_time * 100 / _perc_f
    return _total_time


def remaining_time(i, i_max):
    _elapsed_time = elapsed_time(do_return=True)
    _total_time = total_time(i, i_max)
    _remaining_time = _total_time - _elapsed_time
    return _remaining_time


def progressbar(i=1, i_max=1, symbol='=', mid=None, mode='perc', print_prefix='', p_step=1, printf=tprint, **kwargs):
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


def elapsed_time_init():
    global global_t
    global_t = datetime.datetime.now()


def elapsed_time(do_return=False, ref_t=None):
    global global_t
    if ref_t is None:
        ref_t = global_t

    _delta_t = datetime.datetime.now() - ref_t

    if do_return:
        return _delta_t
    else:
        tprint(str(_delta_t)[:-5])


# formats a time as string, expects numpy datetime or similar
def time_to_str(t, time_format='%Y-%m-%d'):
    return pd.to_datetime(t).strftime(time_format)


# custom vectorize function (only works for 1 parameter)
def cf_vec(x, func, *args, **kwargs):
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


# signif round for one i
def round_signif_i(x, digits=1):
    if not np.isfinite(x):
        return x
    elif x == 0:
        return 0
    else:
        _scale = -int(np.floor(np.log10(abs(x)))) + digits - 1
        return round(float(x), _scale)


# signif round for many i
def round_signif(x, *args, **kwargs):
    return cf_vec(x, round_signif_i, *args, **kwargs)


# signif floor
def floor_signif(x, digits=1):
    if x == 0:
        return 0
    else:
        round_signif_x = round_signif(x, digits=digits)
        if round_signif_x <= x:
            return round_signif_x
        else:
            _scale = -int(np.floor(np.log10(abs(x)))) + digits - 1
            return round_signif_x - 1 / np.power(10., _scale)


# signif ceiling (for plotting)
def ceil_signif(x, digits=1):
    if x == 0:
        return 0
    else:
        round_signif_x = round_signif(x, digits=digits)
        if round_signif_x >= x:
            return round_signif_x
        else:
            _scale = -int(np.floor(np.log10(abs(x)))) + digits - 1
            return round_signif_x + 1 / np.power(10., _scale)


def concat_cols(df, columns, sep='_', to_int=False):
    _df = df.copy()
    del df

    _df['_out'] = ''

    for _i in range(len(columns)):

        if _i > 0:
            _df['_out'] = _df['_out'] + sep

        _col = _df[columns[_i]]

        if to_int:
            _col = _col.round(0).astype(int)

        _df['_out'] = _df['_out'] + _col.astype(str)

    return _df['_out']


# get unique elements from list
def list_unique(lst):
    return list(dict.fromkeys(lst))


# merge n lists (flatten????)
def list_merge(*args, unique=True):
    # more performant than list comprehension
    _list = []

    # args
    for _arg in force_list(args):

        if _arg is None:
            continue

        # argument as a list
        for _arg_i in force_list(_arg):

            if _arg_i is None:
                continue

            # argument as a list of lists
            for _arg_ii in force_list(_arg_i):
                _list.append(_arg_ii)

    if unique:
        _list = list_unique(_list)

    return _list


# common elements in n lists
def list_intersection(list_1, *args):
    # more performant than list comprehension
    _list_out = list(list_1)

    for _arg in args:
        _list = list(_arg)
        _list_out = list(set(_list_out).intersection(_list))

    return _list_out


def rand(shape=None, lower=None, upper=None, step=None, seed=None):
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


def dict_list(*args):
    _dict = {}

    for _arg in args:
        for _list in force_list(_arg):
            _dict[_list] = []

    return _dict


def append_to_dict_list(dic, append, inplace=True):
    if inplace:
        _dic = dic
    else:
        _dic = dic.copy()

    # allows lists and dicts
    if not isinstance(append, collections.Mapping):

        if is_list_like(append):
            _append = list(append)
        else:
            _append = [append]

        if len(_append) > len(dic):
            warnings.warn('list is longer than dict, trailing entries will be lost')
        _append = dict(zip(dic.keys(), _append))

    else:
        _append = append

    for _key in _append.keys():
        _dic[_key].append(_append[_key])

    if not inplace:
        return _dic


def is_list_like(obj):
    if isinstance(obj, collections.Iterable) and not isinstance(obj, (str, bytes)):
        return True
    else:
        return False


def force_list(*args):
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


def qformat(value, int_format=',', float_format=',.2f', datetime_format='%Y-%m-%d', sep=' - ', key_sep=': ',
            print_key=True):

    def _qformat(_value_i):

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


# because the compiler is annoying me about it
def tdelta(*args, **kwargs):
    return np.timedelta64(*args, **kwargs)
