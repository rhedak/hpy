import pickle
import warnings
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Dict

import pandas as pd

from hhpy.main.main import assert_list, get_repr, is_list_like, list_merge


class BaseClass:
    """
        Base class for various classes deriving from this. Implements __repr__, converting to dict as well as
        saving to pickle and restoring from pickle.
        Does NOT provide __init__ since it cannot be used by itself
    """

    # --- globals
    __name__ = 'BaseClass'
    __attributes__ = []
    __attributes_no_repr__ = []
    __dependent_classes__ = []

    # --- functions
    def __repr__(self):
        return get_repr(self)

    @property
    def __dict___(self) -> dict:
        return self.to_dict()

    def __getstate__(self):
        return self.__dict___

    def __setstate__(self, dct):
        return self.from_dict(dct=dct)

    def to_dict(self, recursive: bool = False):
        """
        Converts self to a dictionary

        :recursive: Whether to recursively propagate to_dict to children
        :return: Dictionary
        """

        if len(self.__attributes__) == 0:
            warnings.warn(
                'self.__attributes__ has length zero, did you declare it?')

        _dict = {}
        for _attr_name in list_merge('__name__', self.__attributes__):
            _attr = self.__getattribute__(_attr_name)

            # - call to children's to_dict
            if recursive:
                if not isinstance(_attr, pd.DataFrame) and hasattr(_attr, 'to_dict'):
                    _attr = _attr.to_dict()
                elif is_list_like(_attr):
                    if isinstance(_attr, Mapping):
                        for _key, _value in _attr.items():
                            if hasattr(_value, 'to_dict'):
                                # noinspection PyUnresolvedReferences
                                _attr[_key] = _value.to_dict()
                    else:
                        for _i in range(len(_attr)):
                            if hasattr(_attr[_i], 'to_dict'):
                                print(_attr[_i])
                                _attr[_i] = _attr[_i].to_dict()

            _dict[_attr_name] = _attr

        return _dict

    def from_dict(self, dct: Mapping, recursive: bool = False, classes: Any = None):
        """
        Restores self from a dictionary

        :param dct: Dictionary created from :meth:`~BaseClass.to_dict`
        :param recursive: Whether to recursively call from_dict of children
        :param classes: Classes required to load objects
        :return: None
        """
        # -- functions
        def _f_eval(attr, attr_value):

            # check for passed classes
            _attr_evaluated = None
            for _cla in _classes:
                if hasattr(_cla, '__name__') and attr == _cla.__name__:
                    _attr_evaluated = _cla()
            # if no passed class fits the name assume a default name
            if _attr_evaluated is None:
                _attr_evaluated = eval(attr + '()')
            # check if the evaluated object has a from dict function
            if hasattr(_attr_evaluated, 'from_dict'):
                # check if we can pass classes to the sub-function
                _varnames = _attr_evaluated.from_dict.__code__.co_varnames
                _kws = {}
                if 'classes' in _varnames:
                    _kws['classes'] = _classes
                # call sub-function
                _attr_evaluated.from_dict(attr_value, **_kws)

            return _attr_evaluated

        # -- init
        _classes = assert_list(self.__dependent_classes__, default=[
        ]) + assert_list(classes, default=[])

        # -- main
        if len(self.__attributes__) == 0:
            warnings.warn(
                'self.__attributes__ has length zero, did you declare it?')

        for _attr_name in list_merge('__name__', self.__attributes__):
            if _attr_name not in dct.keys():
                continue
            _attr = dct[_attr_name]

            # - call to children's from_dict
            if recursive:
                if is_list_like(_attr):
                    if isinstance(_attr, Dict):  # Dict
                        if '__name__' in _attr.keys():
                            _attr = _f_eval(_attr['__name__'], _attr)  # eval
                        else:
                            for _attr_key, _attr_value in _attr.items():
                                if isinstance(_attr_value, Dict) and '__name__' in _attr_value.keys():
                                    _attr[_attr_key] = _f_eval(
                                        _attr_value['__name__'], _attr_value)  # eval
                    else:  # List
                        for _i in range(len(_attr)):
                            _attr_value = _attr[_i]
                            if isinstance(_attr_value, Mapping) and '__name__' in _attr_value.keys():
                                _attr[_i] = _f_eval(
                                    _attr_value['__name__'], _attr_value)  # eval

            self.__setattr__(_attr_name, _attr)
        # -- return
        return self

    def to_pickle(self, filename: str):
        """
        Save self to pickle file

        :param filename: filename (path) to be used
        :return: None
        """
        pickle.dump(self, open(filename, 'wb'))

    def copy(self):
        """
        Uses `copy.deepcopy <https://docs.python.org/3/library/copy.html>`_ to return a copy of the object

        :return: Copy of self
        """
        return deepcopy(self)
