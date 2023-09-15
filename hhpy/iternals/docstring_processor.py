from typing import Callable


def make_docstring_decorator(*args, **kwargs) -> Callable:

    # - import
    try:
        # import here as is an optional dependency only needed
        # for building documentation
        # noinspection PyPackageRequirements
        from docrep import DocstringProcessor
    except ImportError:
        return lambda: None
    # - main
    return DocstringProcessor(
        # - args
        *args,
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
        warn='Whether to show UserWarnings triggered by this function. '
             'Set to False to suppress, other warnings will still be triggered [optional]',
        # - validations
        reformat_string__case=['lower', 'upper'],
        dict_inv__duplicates=['adjust', 'drop'],
        progressbar__mode=['perc', 'remaining', 'elapsed'],
        # - kwargs
        **kwargs
    )
