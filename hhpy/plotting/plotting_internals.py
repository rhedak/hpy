from typing import Union

import pandas as pd


def _get_ordered_levels(data: pd.DataFrame, level: str, order: Union[list, str, None], x: str = None) -> list:
    """
    internal function for getting the ordered levels of a categorical like column in a pandas DataFrame

    :param data: pandas DataFrame
    :param level: name of the column
    :param order: how to order it, details see below
    :param x: secondary column name, used to aggregate before sorting
    :return: list of ordered levels
    """
    if order is None or order == 'sorted':
        _hues = data[level].drop_duplicates().sort_values().tolist()
    elif order == 'inv':
        _hues = data[level].drop_duplicates().sort_values().tolist()[::-1]
    elif order == 'count':
        _hues = data[level].value_counts().reset_index().sort_values(
            by=[level, 'index'])['index'].tolist()
    elif order in ['mean', 'mean_descending']:
        _hues = data.groupby(level)[x].mean().reset_index().sort_values(by=[x, level], ascending=[False, True]
                                                                        )[level].tolist()
    elif order == 'mean_ascending':
        _hues = data.groupby(level)[x].mean().reset_index(
        ).sort_values(by=[x, level])[level].tolist()
    elif order in ['median', 'median_descending']:
        _hues = data.groupby(level)[x].median().reset_index().sort_values(by=[x, level], ascending=[False, True]
                                                                          )[level].tolist()
    elif order == 'median_ascending':
        _hues = data.groupby(level)[x].median().reset_index(
        ).sort_values(by=[x, level])[level].tolist()
    else:
        _hues = order

    return _hues
