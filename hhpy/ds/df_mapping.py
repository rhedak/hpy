import pandas as pd

from hhpy.main import BaseClass, list_exclude


class DFMapping(BaseClass):
    """
        Mapping object bound to a pandas DataFrame that standardizes column names and values according to the chosen
        conventions. Also implements google translation. Can be used like a sklearn scalar object.
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
        # - if the function is called with only one argument attempt to parse its type and act accordingly
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
            but if you have too many you likely have a column of non-allowed character strings and the mapping
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
                warnings.warn(
                    f'Unknown return_type {return_type}, falling back to self')
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
                progressbar(_it, len(_columns), printf=printf,
                            print_prefix=f'{_column}: ')
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
                            warnings.warn(
                                'too many reformated duplicates in column names')
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
                                    warnings.warn(
                                        f'too many reformated duplicates in column {_column}')
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
        self.fit(df=df, col_names=col_names, values=values,
                 columns=columns, **kwargs_fit)
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
            raise ValueError(
                f"if_exists must be one of {validations['DFMapping__to_excel__if_exists']}")
        # - handle if_exists
        _sheet_names = []
        if os.path.exists(path):
            if if_exists == 'error':
                raise FileExistsError('file already exists, please specify if_exists as one of')
            elif if_exists == 'append':
                _sheet_names = pd.ExcelFile(path).sheet_names
        # -- main
        # pandas ExcelWriter object (saves on close)
        with pd.ExcelWriter(path) as _writer:
            # col mapping
            _write_excel_sheet(
                writer=_writer, mapping=self.col_mapping, sheet_name='__columns__')
            # value mappings
            for _key, _mapping in self.value_mapping.items():
                _write_excel_sheet(
                    writer=_writer, mapping=_mapping, sheet_name=_key)

    def from_excel(self, path: str) -> None:
        """
        Init the DFMapping object from an Excel file. For example, you could auto generate a DFMapping using googletrans
        and then adjust the translations you feel are inappropriate in the Excel file. Then regenerate the object
        from the edited Excel file.

        :param path: Path to the Excel file
        :return: None
        """

        def _read_excel(xls, sheet_name):
            return pd.read_excel(xls, sheet_name, index_col=0).T.to_dict(orient='records')[0]

        # open ExcelFile
        with pd.ExcelFile(path) as _xls:
            self.col_mapping = _read_excel(xls=_xls, sheet_name='__columns__')
            self.value_mapping = {}
            for _sheet_name in list_exclude(_xls.sheet_names, '__columns__'):
                self.value_mapping[_sheet_name] = _read_excel(
                    xls=_xls, sheet_name=_sheet_name)
