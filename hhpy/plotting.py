"""
hhpy.plotting.py
~~~~~~~~~~~~~~~~

Contains plotting functions using matplotlib.pyplot

"""

# -- imports
# - standard imports
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
import itertools

# - third party imports
from matplotlib import patches, colors as mpl_colors
from matplotlib.animation import FuncAnimation
from matplotlib.legend import Legend
from colour import Color
from scipy import stats
from typing import Union, Sequence, Mapping, Callable, List

# local imports
from hhpy.main import export, concat_cols, is_list_like, floor_signif, ceil_signif, list_intersection, \
    assert_list, progressbar, DocstringProcessor, Scalar, SequenceOrScalar
from hhpy.ds import get_df_corr, lfit, kde, df_count, quantile_split, top_n_coding, df_rmsd, df_agg

# - optional imports
logger = logging.getLogger('hhpy.plotting')
try:
    from IPython.core.display import HTML
except ImportError:
    # noinspection PyPep8Naming
    def HTML(obj):
        logger.warning('Missing optional dependency IPython.core.display.HTML')
        return obj

try:
    # noinspection PyPackageRequirements
    from plotly import graph_objects as go
    go_figure = go.Figure
except ImportError:
    logger.warning('Missing optional dependency plotly')
    go = None
    go_figure = None

# --- constants
rcParams = {
    'palette': [
        'xkcd:blue', 'xkcd:red', 'xkcd:green', 'xkcd:cyan', 'xkcd:magenta',
        'xkcd:golden yellow', 'xkcd:dark cyan', 'xkcd:red orange', 'xkcd:dark yellow', 'xkcd:easter green',
        'xkcd:baby blue', 'xkcd:light brown', 'xkcd:strong pink', 'xkcd:light navy blue', 'xkcd:deep blue',
        'xkcd:deep red', 'xkcd:ultramarine blue', 'xkcd:sea green', 'xkcd:plum', 'xkcd:old pink',
        'xkcd:lawn green', 'xkcd:amber', 'xkcd:green blue', 'xkcd:yellow green', 'xkcd:dark mustard',
        'xkcd:bright lime', 'xkcd:aquamarine', 'xkcd:very light blue', 'xkcd:light grey blue', 'xkcd:dark sage',
        'xkcd:dark peach', 'xkcd:shocking pink'
    ],
    'hatches': ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'],
    'figsize_square': (7, 7),
    'fig_width': 7,
    'fig_height': 7,
    'float_format': '.2f',
    'int_format': ',.0f',
    'legend_outside.legend_space': .1,
    'distplot.label_style': 'mu_sigma',
    'distplot.legend_loc': None,
    'max_n': 10000,
    'max_n.random_state': None,
    'max_n.sample_warn': True,
    'return_fig_ax': True,
    'corr_cutoff': 0,
    'animplot.mode': 'jshtml',
}

validations = {
    'distplot__distfit': ['kde', 'gauss', 'False', 'None'],
    'cat_to_color__out_type': [None, 'hex', 'rgb', 'rgba', 'rgba_array']
}

docstr = DocstringProcessor(
    ax_in='The matplotlib.pyplot.Axes object to plot on, defaults to current axis [optional]',
    ax_out='The matplotlib.pyplot.Axes object with the plot on it',
    fig_ax_out='if return_fig_ax: figure and axes objects as tuple, else None',
    x='Name of the x variable in data or vector data',
    y='Name of the y variable in data or vector data',
    t='Name of the t variable in data or vector data',
    x_novec='Name of the x variable in data',
    y_novec='Name of the y variable in data',
    t_novec='Name of the t variable in data',
    data='Pandas DataFrame containing named data, optional if vector data is used',
    data_novec='Pandas DataFrame containing named data',
    hue='Further split the plot by the levels of this variable [optional]',
    order='Either a string describing how the (hue) levels or to be ordered or an explicit list of levels to be ' 
    'used for plotting. Accepted strings are: '
    '''
    
        * ``sorted``: following python standard sorting conventions (alphabetical for string, ascending for value)
        
        * ``inv``: following python standard sorting conventions but in inverse order
        
        * ``count``: sorted by value counts
        
        * ``mean``, ``mean_ascending``, ``mean_descending``: sorted by mean value, defaults to descending
        
        * ``median``, ``mean_ascending``, ``median_descending``: sorted by median value, defaults to descending
        
    ''',
    color='Color used for plotting, must be known to matplotlib.pyplot [optional]',
    palette='Collection of colors to be used for plotting. Can be a dictionary for with names for each level or '
            'a list of colors or an individual color name. Must be valid colors known to pyplot [optional]',
    cmap='Color map to use [optional]',
    annotations='Whether to display annotations [optional]',
    number_format='The format string used for annotations [optional]',
    float_format='The format string used for displaying floats [optional]',
    int_format='The format string used for displaying floats [optional]',
    corr_target='Target variable name, if specified only correlations with the target are shown [optional]',
    corr_cutoff='Filter all correlation whose absolute value is below the cutoff [optional]',
    col_wrap='After how many columns to create a new line of subplots [optional]',
    subplot_width='Width of each individual subplot [optional]',
    subplot_height='Height of each individual subplot [optional]',
    trendline='Whether to add a trendline [optional]',
    alpha='Alpha transparency level [optional]',
    max_n='''Maximum number of samples to be used for plotting, if this number is exceeded max_n samples are drawn '
          'at random from the data which triggers a warning unless sample_warn is set to False. '
          'Set to False or None to use all samples for plotting. [optional]''',
    max_n_random_state='Random state (seed) used for drawing the random samples [optional]',
    max_n_sample_warn='Whether to trigger a warning if the data has more samples than max_n [optional]',
    return_fig_ax='Whether to return the figure and axes objects as tuple to be captured as fig,ax = ..., '
                  'If False pyplot.show() is called and the plot returns None [optional]',
    legend='Whether to show a legend [optional]',
    legend_loc='Location of the legend, one of [bottom, right] or accepted value of pyplot.legend'
               'If in [bottom, right] legend_outside is used, else pyplot.legend [optional]',
    legend_ncol='Number of columns to use in legend [optional]',
    legend_space='Only valid if legend_loc is bottom. The space between the main plot and the legend [optional]',
    kde_steps='Nr of steps the range is split into for kde fitting [optional]',
    linestyle='Linestyle used, must a valid linestyle recognized by pyplot.plot [optional]',
    bins='Nr of bins of the histogram [optional]',
    sharex='Whether to share the x axis [optional]',
    sharey='Whether to share the y axis [optional]',
    row='Row index [optional]',
    col='Column index [optional]',
    legend_out='Whether to draw the legend outside of the axis, can also be a location string [optional]',
    legend_kws='Other keyword arguments passed to pyplot.legend [optional]',
    xlim='X limits for the axis as tuple, passed to ax.set_xlim() [optional]',
    ylim='Y limits for the axis as tuple, passed to ax.set_ylim() [optional]',
    grid='Whether to toggle ax.grid() [optional]',
    vline='A list of x positions to draw vlines at [optional]',
    to_abs='Whether to cast the values to absolute before proceeding [optional]',
    label='Label to use for the data [optional]',
    x_tick_rotation='Set x tick label rotation to this value [optional]',
    std_cutoff='Remove data outside of std_cutoff standard deviations, for a good visual experience try 3 [optional]',
    do_print='Whether to print intermediate steps to console [optional]',
    x_min='Lower limit for the x axis [optional]',
    x_max='Upper limit for the x axis [optional]',
    y_min='Lower limit for the y axis [optional]',
    y_max='Upper limit for the y axis [optional]',
    title_plotly='Figure title, passed to plotly.Figure.update_layout [optional]',
    xaxis_title='x axis title, passed to plotly.Figure.update_layout [optional]',
    yaxis_title='y axis title, passed to plotly.Figure.update_layout [optional]',
    fig_plotly='The plotly.Figure object to draw the plot on [optional]',
    **validations
)


# --- functions
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
        _hues = data[level].value_counts().reset_index().sort_values(by=[level, 'index'])['index'].tolist()
    elif order in ['mean', 'mean_descending']:
        _hues = data.groupby(level)[x].mean().reset_index().sort_values(by=[x, level], ascending=[False, True]
                                                                        )[level].tolist()
    elif order == 'mean_ascending':
        _hues = data.groupby(level)[x].mean().reset_index().sort_values(by=[x, level])[level].tolist()
    elif order in ['median', 'median_descending']:
        _hues = data.groupby(level)[x].median().reset_index().sort_values(by=[x, level], ascending=[False, True]
                                                                          )[level].tolist()
    elif order == 'median_ascending':
        _hues = data.groupby(level)[x].median().reset_index().sort_values(by=[x, level])[level].tolist()
    else:
        _hues = order

    return _hues


@docstr
@export
def heatmap(x: str, y: str, z: str, data: pd.DataFrame, ax: plt.Axes = None, cmap: object = None,
            agg_func: str = 'mean', invert_y: bool = True, **kwargs) -> plt.Axes:
    """
    Wrapper for seaborn heatmap in x-y-z format

    :param x: Variable name for x axis value
    :param y: Variable name for y axis value
    :param z: Variable name for z value, used to color the heatmap
    :param data: %(data)s
    :param ax: %(ax_in)s
    :param cmap: %(cmap)s
    :param agg_func: If more than one z value per x,y pair exists agg_func is used to aggregate the data.
        Must be a function name understood by pandas.DataFrame.agg
    :param invert_y: Whether to call ax.invert_yaxis (orders the heatmap as expected)
    :param kwargs: Other keyword arguments passed to seaborn heatmap
    :return: %(ax_out)s
    """
    if cmap is None:
        cmap = sns.diverging_palette(10, 220, as_cmap=True)

    _df = data.groupby([x, y]).agg({z: agg_func}).reset_index().pivot(x, y, z)

    if ax is None:
        ax = plt.gca()

    sns.heatmap(_df, ax=ax, cmap=cmap, **kwargs)
    ax.set_title(z)

    if invert_y:
        ax.invert_yaxis()

    return ax


@docstr
@export
def corrplot(data: pd.DataFrame, annotations: bool = True, number_format: str = rcParams['float_format'], ax=None):
    """
    function to create a correlation plot using a seaborn heatmap
    based on: https://www.linkedin.com/pulse/generating-correlation-heatmaps-seaborn-python-andrew-holt
        
    :param number_format: %(number_format)s
    :param data: %(data_novec)s
    :param annotations: %(annotations)s
    :param ax: %(ax_in)s
    :return: %(ax_out)s
    """
    # Create Correlation df
    _corr = data.corr()

    if ax is None:
        ax = plt.gca()
    # Generate Color Map
    _colormap = sns.diverging_palette(220, 10, as_cmap=True)

    # mask
    _mask = np.zeros_like(_corr)
    _mask[np.triu_indices_from(_mask)] = True

    # Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(_corr, cmap=_colormap, annot=annotations, fmt=number_format, mask=_mask, ax=ax)

    # Adjust tick labels
    ax.set_xticks(ax.get_xticks()[:-1])
    _yticklabels = ax.get_yticklabels()[1:]
    ax.set_yticks(ax.get_yticks()[1:])
    ax.set_yticklabels(_yticklabels)

    return ax


@docstr
@export
def corrplot_bar(data: pd.DataFrame, target: str = None, columns: List[str] = None,
                 corr_cutoff: float = rcParams['corr_cutoff'], corr_as_alpha: bool = False,
                 xlim: tuple = (-1, 1), ax: plt.Axes = None):
    """
    Correlation plot as barchart based on :func:`~hhpy.ds.get_df_corr`
    
    :param data: %(data)s
    :param target: %(corr_target)s
    :param columns: Columns for which to calculate the correlations, defaults to all numeric columns [optional]
    :param corr_cutoff: %(corr_cutoff)s
    :param corr_as_alpha: Whether to set alpha value of bars to scale with correlation [optional]
    :param xlim: xlim scale for plot, defaults to (-1, 1) to show the absolute scale of the correlations.
        set to None if you want the plot x limits to scale to the highest correlation values [optional]
    :param ax: %(ax_in)s
    :return: %(ax_out)s
    """
    _df_corr = get_df_corr(data, target=target)
    _df_corr = _df_corr[_df_corr['corr_abs'] >= corr_cutoff]

    if target is None:
        _df_corr['label'] = concat_cols(_df_corr, ['col_0', 'col_1'], sep=' X ')
    else:
        _df_corr['label'] = _df_corr['col_1']

    # filter columns (if applicable)
    if columns is not None:
        _columns = columns + []
        if target is not None and target not in _columns:
            _columns.append(target)
        _df_corr = _df_corr[(_df_corr['col_0'].isin(_columns)) & (_df_corr['col_1'].isin(_columns))]

    # get colors
    _rgba_colors = np.zeros((len(_df_corr), 4))

    # for red the first column needs to be one
    _rgba_colors[:, 0] = np.where(_df_corr['corr'] > 0., 0., 1.)
    # for blue the third column needs to be one
    _rgba_colors[:, 2] = np.where(_df_corr['corr'] > 0., 1., 0.)
    # the fourth column needs to be alphas
    if corr_as_alpha:
        _rgba_colors[:, 3] = _df_corr['corr_abs'].where(lambda _: _ > .1, .1)
    else:
        _rgba_colors[:, 3] = 1

    if ax is None:
        ax = plt.gca()

    _rgba_colors = np.round(_rgba_colors, 2)

    _plot = ax.barh(_df_corr['label'], _df_corr['corr'], color=_rgba_colors)
    ax.invert_yaxis()

    if xlim:
        # noinspection PyTypeChecker
        ax.set_xlim(xlim)

    if target is not None:
        ax.set_title('Correlations with {} by Absolute Value'.format(target))
        ax.set_xlabel('corr × {}'.format(target))
    else:
        ax.set_title('Correlations by Absolute Value')

    return ax


@docstr
@export
def pairwise_corrplot(data: pd.DataFrame, corr_cutoff: float = rcParams['corr_cutoff'], col_wrap: int = 4,
                      hue: str = None, hue_order: Union[list, str] = None,
                      width: float = rcParams['fig_width'], height: float = rcParams['fig_height'],
                      trendline: bool = True, alpha: float = .75, ax: plt.Axes = None,
                      target: str = None, palette: Union[Mapping, Sequence, str] = rcParams['palette'],
                      max_n: int = rcParams['max_n'], random_state: int = rcParams['max_n.random_state'],
                      sample_warn: bool = rcParams['max_n.sample_warn'],
                      return_fig_ax: bool = rcParams['return_fig_ax'], **kwargs) -> Union[tuple, None]:
    """
    print a pairwise_corrplot to for all variables in the df, by default only plots those with a correlation
    coefficient of >= corr_cutoff
    
    :param data: %(data_novec)s
    :param corr_cutoff: %(corr_cutoff)s
    :param col_wrap: %(col_wrap)s
    :param hue: %(hue)s
    :param hue_order: %(order)s
    :param width: %(subplot_width)s
    :param height: %(subplot_height)s
    :param trendline: %(trendline)s
    :param alpha: %(alpha)s
    :param ax: %(ax_in)s
    :param target: %(corr_target)s
    :param palette: %(palette)s
    :param max_n: %(max_n)s
    :param random_state: %(max_n_random_state)s
    :param sample_warn: %(max_n_sample_warn)s
    :param return_fig_ax: %(return_fig_ax)s
    :param kwargs: other keyword arguments passed to pyplot.subplots
    :return: %(fig_ax_out)s
    """

    # actual plot function
    def _f_plot(_f_x, _f_y, _f_data, _f_color, _f_color_trendline, _f_label, _f_ax):

        _data = _f_data.copy()

        # limit plot points
        if max_n is not None:
            if len(_data) > max_n:
                if sample_warn:
                    warnings.warn(
                        'Limiting Scatter Plot to {:,} randomly selected points. '
                        'Turn this off with max_n=None or suppress this warning with sample_warn=False.'.format(
                            max_n))
                _data = _data.sample(max_n, random_state=random_state)

        _f_ax.scatter(_f_x, _f_y, data=_data, alpha=alpha, color=_f_color, label=_f_label)

        if trendline:
            _f_ax.plot(_f_data[_f_x], lfit(_f_data[_f_x], _f_data[_f_y]), color=_f_color_trendline, linestyle=':')

        return _f_ax

    # avoid inplace operations
    _df = data.copy()
    _df_hues = pd.DataFrame()
    _df_corrs = pd.DataFrame()
    _hues = None

    if hue is not None:
        _hues = _get_ordered_levels(_df, hue, hue_order)
        _df_hues = {}
        _df_corrs = {}

        for _hue in _hues:
            _df_hue = _df[_df[hue] == _hue]
            _df_corr_hue = get_df_corr(_df_hue, target=target)

            _df_hues[_hue] = _df_hue.copy()
            _df_corrs[_hue] = _df_corr_hue.copy()

    # get df corr
    _df_corr = get_df_corr(_df, target=target)

    if corr_cutoff is not None:
        _df_corr = _df_corr[_df_corr['corr_abs'] >= corr_cutoff]

    # warning for empty df
    if len(_df_corr) == 0:
        warnings.warn('Correlation DataFrame is Empty. Do you need a lower corr_cutoff?')
        return None

    # edge case for less plots than ncols
    if len(_df_corr) < col_wrap:
        _ncols = len(_df_corr)
    else:
        _ncols = col_wrap

    # calculate nrows
    _nrows = int(np.ceil(len(_df_corr) / _ncols))

    _figsize = (width * col_wrap, height * _nrows)

    if ax is None:
        fig, ax = plt.subplots(nrows=_nrows, ncols=_ncols, figsize=_figsize, **kwargs)
    else:
        fig = plt.gcf()

    _row = None
    _col = None

    for _it in range(len(_df_corr)):

        _col = _it % _ncols
        _row = _it // _ncols

        _x = _df_corr.iloc[_it]['col_1']
        _y = _df_corr.iloc[_it]['col_0']  # so that target (if available) becomes y
        _corr = _df_corr.iloc[_it]['corr']

        if _ncols == 1:
            _rows_prio = True
        else:
            _rows_prio = False

        _ax = get_subax(ax, _row, _col, rows_prio=_rows_prio)

        _ax.set_xlabel(_x)
        _ax.set_ylabel(_y)
        _ax.set_title('corr = {:.3f}'.format(_corr))

        # hue if
        if hue is None:

            # actual plot
            _f_plot(_f_x=_x, _f_y=_y, _f_data=_df, _f_color=None, _f_color_trendline='k', _f_label=None, _f_ax=_ax)

        else:

            for _hue_it, _hue in enumerate(_hues):

                if isinstance(palette, Mapping):
                    _color = palette[_hue]
                elif is_list_like(palette):
                    _color = palette[_hue_it % len(palette)]
                else:
                    _color = palette

                _df_hue = _df_hues[_hue]
                _df_corr_hue = _df_corrs[_hue].copy()

                # sometimes it can happen that the correlation is not possible to calculate because
                # one of those values does not change in the hue level
                # i.e. use try except

                try:
                    _df_corr_hue = _df_corr_hue[_df_corr_hue['col_1'] == _x]
                    _df_corr_hue = _df_corr_hue[_df_corr_hue['col_0'] == _y]
                    _corr_hue = _df_corr_hue['corr'].iloc[0]
                except ValueError:
                    _corr_hue = 0

                # actual plot
                _f_plot(_f_x=_x, _f_y=_y, _f_data=_df_hue, _f_color=_color, _f_color_trendline=_color,
                        _f_label='{} corr: {:.3f}'.format(_hue, _corr_hue), _f_ax=_ax)

                _ax.legend()

    # hide unused axis
    for __col in range(_col + 1, _ncols):
        get_subax(ax, _row, __col, rows_prio=False).set_axis_off()

    if return_fig_ax:
        return fig, ax
    else:
        plt.show()


@docstr
@export
def distplot(x: Union[Sequence, str], data: pd.DataFrame = None, hue: str = None,
             hue_order: Union[Sequence, str] = 'sorted', palette: Union[Mapping, Sequence, str] = None,
             linecolor: str = 'black', edgecolor: str = 'black', alpha: float = None, bins: Union[Sequence, int] = 40,
             perc: bool = None, top_nr: int = None, other_name: str = 'other', title: bool = True,
             title_prefix: str = '', std_cutoff: float = None, hist: bool = None,
             distfit: Union[str, bool, None] = 'kde', fill: bool = True, legend: bool = True,
             legend_loc: str = rcParams['distplot.legend_loc'],
             legend_space: float = rcParams['legend_outside.legend_space'], legend_ncol: int = 1,
             agg_func: str = 'mean', number_format: str = rcParams['float_format'], kde_steps: int = 1000,
             max_n: int = 100000, random_state: int = None, sample_warn: bool = True, xlim: Sequence = None,
             linestyle: str = None, label_style: str = rcParams['distplot.label_style'], x_offset_perc: float = .025,
             ax: plt.Axes = None, **kwargs) -> plt.Axes:
    """
    Similar to seaborn.distplot but supports hues and some other things. Plots a combination of a histogram and
    a kernel density estimation.
    
    :param x: the name of the variable(s) in data or vector data, if data is provided and x is a list of columns
        the DataFrame is automatically melted and the newly generated column used as hue. i.e. you plot
        the distributions of multiple columns on the same axis
    :param data: %(data)s
    :param hue: %(hue)s
    :param hue_order: %(order)s
    :param palette: %(palette)s
    :param linecolor: Color of the kde fit line, overwritten with palette by hue level if hue is specified [optional]
    :param edgecolor: Color of the histogram edges [optional]
    :param alpha: %(alpha)s
    :param bins: %(bins)s
    :param perc: Whether to display the y-axes as percentage, if false count is displayed.
        Defaults if hue: True, else False [optional]
    :param top_nr: limit hue to top_nr levels using hhpy.ds.top_n, the rest will be cast to other [optional]
    :param other_name: name of the other group created by hhpy.ds.top_n [optional]
    :param title: whether to set the plot title equal to x's name [optional]
    :param title_prefix: prefix to be used in plot title [optional]
    :param std_cutoff: automatically cutoff data outside of the std_cutoff standard deviations range,
        by default this is off but a recommended value for a good visual experience without outliers is 3 [optional]
    :param hist: whether to show the histogram, default False if hue else True [optional]
    :param distfit: one of %(distplot__distfit)s. If 'kde' fits a kernel density distribution to the data.
        If gauss fits a gaussian distribution with the observed mean and std to the data. [optional]
    :param fill: whether to fill the area under the distfit curve, ignored if hist is True [optional]
    :param legend: %(legend)s
    :param legend_loc: %(legend_loc)s
    :param legend_space: %(legend_space)s
    :param legend_ncol: %(legend_ncol)s
    :param agg_func: one of ['mean', 'median']. The agg function used to find the center of the distribution [optional]
    :param number_format: %(number_format)s
    :param kde_steps: %(kde_steps)s
    :param max_n: %(max_n)s
    :param random_state: %(max_n_random_state)s
    :param sample_warn: %(max_n_sample_warn)s
    :param xlim: %(xlim)s
    :param linestyle: %(linestyle)s
    :param label_style: one of ['mu_sigma', 'plain']. If mu_sigma then the mean (or median) and std value are displayed
        inside the label [optional]
    :param x_offset_perc: the amount whitespace to display next to x_min and x_max in percent of x_range [optional]
    :param ax: %(ax_in)s
    :param kwargs: additional keyword arguments passed to pyplot.plot
    :return: %(ax_out)s
    """
    # -- asserts
    if distfit not in validations['distplot__distfit']:
        raise ValueError(f"distfit must be one of {validations['distplot__distfit']}")
    # -- defaults
    if palette is None:
        palette = rcParams['palette']
    if not top_nr:
        top_nr = None

    # case: vector data
    if data is None:

        if hasattr(x, 'name'):
            _x_name = x.name
        else:
            _x_name = 'x'

        _df = pd.DataFrame({_x_name: x})
        x = _x_name

    # data: DataFrame
    else:

        _df = data.copy()  # avoid inplace operations
        del data

        if is_list_like(x) and len(x) > 1:
            hue_order = assert_list(x)
            title = False
            hue = '__variable__'
            x = '__value__'
            _df = pd.melt(_df, value_vars=x, value_name=x, var_name=hue)

    # handle hue and default values
    if hue is None:
        if perc is None:
            perc = False
        if hist is None:
            hist = True
        if alpha is None:
            alpha = .75
    else:
        _df = _df[~_df[hue].isnull()]
        if perc is None:
            perc = True
        if hist is None:
            hist = False
        if alpha is None:
            alpha = .5

    # case more than max_n samples: take a random sample for calc speed
    if max_n and (len(_df) > max_n):
        if sample_warn:
            warnings.warn(
                f"Limiting samples to {max_n:,} for calc speed. Turn this off with max_n=None or suppress this warning "
                "with sample_warn=False.")
        _df = _df.sample(max_n, random_state=random_state)

    # the actual plot
    def _f_distplot(_f_x, _f_data, _f_x_label, _f_facecolor, _f_distfit_color, _f_bins,
                    _f_std_cutoff, _f_xlim, _f_distfit_line, _f_ax, _f_ax2, _f_fill):

        # make a copy to avoid inplace operations
        _df_i = _f_data.copy()

        # best fit of data
        _mu = _df_i.agg({_f_x: agg_func})[0]
        _sigma = _df_i.agg({_f_x: 'std'})[0]

        # apply sigma cutoff
        if (_f_std_cutoff is not None) or (_f_xlim is not None):

            if _f_xlim is not None:

                __x_min = _f_xlim[0]
                __x_max = _f_xlim[1]

            elif is_list_like(_f_std_cutoff):

                __x_min = _f_std_cutoff[0]
                __x_max = _f_std_cutoff[1]

            else:

                __x_min = _mu - _f_std_cutoff * _sigma
                __x_max = _mu + _f_std_cutoff * _sigma

            _df_i = _df_i[
                (_df_i[_f_x] >= __x_min) &
                (_df_i[_f_x] <= __x_max)
                ]

        # for plot trimming
        _x_mins.append(_df_i[x].min())
        _x_maxs.append(_df_i[x].max())

        # handle label
        try:
            _mu_label = format(_mu, number_format)
        except ValueError:
            _mu_label = '0'

        try:
            _sigma_label = format(_sigma, number_format)
        except ValueError:
            _sigma_label = '0'

        if agg_func == 'mean':
            _mu_symbol = r'\ \mu'
        else:
            _mu_symbol = r'\ \nu'

        if label_style == 'mu_sigma':
            _label = r'{}: $ {}={},\ \sigma={}$'.format(_f_x_label, _mu_symbol, _mu_label, _sigma_label)
        else:
            _label = _f_x_label

        # plot histogram
        if hist:
            _hist_n, _hist_bins = _f_ax.hist(_df_i[_f_x], _f_bins, density=perc, facecolor=_f_facecolor,
                                             edgecolor=edgecolor,
                                             alpha=alpha, label=_label)[:2]
            _label_2 = '__nolegend___'
            if _f_distfit_line is None:
                _f_distfit_line = '--'
        else:
            _hist_n = None
            _hist_bins = None
            _label_2 = _label + ''
            if _f_distfit_line is None:
                _f_distfit_line = '-'

        # plot distfit
        if distfit:

            # if a histogram was plot on the primary axis, the distfit goes on the secondary axis
            if hist:
                _ax = _f_ax2
            else:
                _ax = _f_ax

            if distfit == 'gauss':

                # add a 'best fit' line
                __x = _f_bins
                _y = stats.norm.pdf(_f_bins, _mu, _sigma)  # _hist_bins
                _ax.plot(__x, _y, linestyle=_f_distfit_line, color=_f_distfit_color, alpha=alpha, linewidth=2,
                         label=_label_2, **kwargs)

            elif distfit == 'kde':

                _kde = kde(x=_f_x, df=_df_i, x_steps=kde_steps)[0]
                __x = _kde[_f_x]
                _y = _kde['value']
                _ax.plot(__x, _y, linestyle=_f_distfit_line, color=_f_distfit_color, alpha=alpha, linewidth=2,
                         label=_label_2, **kwargs)

            if not hist:
                _ax.set_ylabel('pdf')
                if _f_fill:
                    # noinspection PyUnboundLocalVariable
                    _ax.fill_between(__x, _y, color=_f_facecolor, alpha=alpha)

        _f_ax2.get_yaxis().set_visible(False)

        if perc and hist:

            _y_max = np.max(_hist_n) / np.sum(_hist_n) * 100
            _y_ticklabels = list(_f_ax.get_yticks())
            _y_ticklabels = [float(_) for _ in _y_ticklabels]

            _factor = _y_max / np.nanmax(_y_ticklabels)
            if np.isnan(_factor):
                _factor = 1
            _y_ticklabels = [format(int(_ * _factor), ',') for _ in _y_ticklabels]
            _f_ax.set_yticklabels(_y_ticklabels)
            _f_ax.set_ylabel('%')

        elif hist:

            _f_ax.set_ylabel('count')

            # adjust xlims if necessary
            _xlim = list(_f_ax.get_xlim())

            # here _df is used to access the 'parent' DataFrame with all hue levels
            if _xlim[0] <= _plot_x_min:
                _xlim[0] = _plot_x_min
            if _xlim[1] >= _plot_x_max:
                _xlim[1] = _plot_x_max

            _f_ax.set_xlim(_xlim)

        return _f_ax, _f_ax2

    # -- preparing the data frame
    # drop nan values
    _df = _df[np.isfinite(_df[x])]

    # init plot
    if ax is None:
        ax = plt.gca()
    ax2 = ax.twinx()

    # for plot trimming
    _x_mins = []
    _x_maxs = []

    if hue is None:

        # handle x limits
        if xlim is not None:

            _x_min = xlim[0]
            _x_max = xlim[1]

        elif std_cutoff is not None:

            _x_min = _df[x].mean() - _df[x].std() * std_cutoff
            _x_max = _df[x].mean() + _df[x].std() * std_cutoff

        else:

            _x_min = _df[x].min()
            _x_max = _df[x].max()

        # edge case
        if _x_min == _x_max:
            warnings.warn('Distribution min and max are equal')
            _x_min -= 1
            _x_max += 1

        # handle bins
        if not is_list_like(bins):
            _x_step = (_x_max - _x_min) / bins
            _bins = np.arange(_x_min, _x_max + _x_step, _x_step)

            _plot_x_min = _df[x].min() - _x_step
            _plot_x_max = _df[x].max() + _x_step
        else:
            _bins = bins
            _plot_x_min = np.min(bins)
            _plot_x_max = np.max(bins)

        # handle palette / color
        if isinstance(palette, Mapping):
            _color = list(palette.values())[0]
        elif is_list_like(palette):
            _color = palette[0]
        else:
            _color = palette

        # plot
        ax, ax2 = _f_distplot(_f_x=x, _f_data=_df, _f_x_label=x, _f_facecolor=_color,
                              _f_distfit_color=linecolor,
                              _f_bins=_bins, _f_std_cutoff=std_cutoff,
                              _f_xlim=xlim, _f_distfit_line=linestyle, _f_ax=ax, _f_ax2=ax2, _f_fill=fill)

    else:  # no hue

        # group values outside of top_n to other_name
        if top_nr is not None:
            _hues = _df[hue].value_counts().reset_index().sort_values(by=[hue, 'index'])['index'].tolist()
            if (top_nr + 1) < len(_hues):  # the plus 1 is there to avoid the other group having exactly 1 entry
                _hues = pd.Series(_hues)[0:top_nr]
                _df[hue] = np.where(_df[hue].isin(_hues), _df[hue], other_name)
                _df[hue] = _df[hue].astype('str')
                _hues = list(_hues) + [other_name]
        # parse hue order
        else:
            _hues = _get_ordered_levels(_df, hue, hue_order, x)

        # find shared _x_min ; _x_max
        if xlim is not None:

            _std_cutoff_hues = None

            _x_min = xlim[0]
            _x_max = xlim[1]

        elif std_cutoff is None:

            _std_cutoff_hues = None

            _x_min = _df[x].min()
            _x_max = _df[x].max()

        else:
            _df_agg = _df.groupby(hue).agg({x: ['mean', 'std']}).reset_index()
            _df_agg.columns = [hue, 'mean', 'std']
            _df_agg['x_min'] = _df_agg['mean'] - _df_agg['std'] * std_cutoff
            _df_agg['x_max'] = _df_agg['mean'] + _df_agg['std'] * std_cutoff
            _df_agg['x_range'] = _df_agg['x_max'] - _df_agg['x_min']

            _x_min = _df_agg['x_min'].min()
            _x_max = _df_agg['x_max'].max()

            _std_cutoff_hues = [_x_min, _x_max]

        # handle bins
        _x_step = (_x_max - _x_min) / bins

        _plot_x_min = _df[x].min() - _x_step
        _plot_x_max = _df[x].max() + _x_step

        _bins = np.arange(_x_min, _x_max + _x_step, _x_step)

        # loop hues
        for _it, _hue in enumerate(_hues):

            if isinstance(palette, Mapping):
                _color = palette[_hue]
            elif is_list_like(palette):
                _color = palette[_it]
            else:
                _color = palette

            if isinstance(linestyle, Mapping):
                _linestyle = linestyle[_hue]
            elif is_list_like(linestyle):
                _linestyle = linestyle[_it]
            else:
                _linestyle = linestyle

            _df_hue = _df[_df[hue] == _hue]

            # one plot per hue
            ax, ax2 = _f_distplot(_f_x=x, _f_data=_df_hue, _f_x_label=_hue, _f_facecolor=_color,
                                  _f_distfit_color=_color, _f_bins=_bins,
                                  _f_std_cutoff=_std_cutoff_hues,
                                  _f_xlim=xlim, _f_distfit_line=_linestyle, _f_ax=ax, _f_ax2=ax2, _f_fill=fill)

    # -- postprocessing
    # handle legend
    if legend:
        if legend_loc in ['bottom', 'right']:
            legend_outside(ax, loc=legend_loc, legend_space=legend_space, ncol=legend_ncol)
            legend_outside(ax2, loc=legend_loc, legend_space=legend_space, ncol=legend_ncol)
        else:
            _, _labels = ax.get_legend_handles_labels()
            if len(_labels) > 0:
                ax.legend(loc=legend_loc, ncol=legend_ncol)

            _, _labels = ax2.get_legend_handles_labels()
            if len(_labels) > 0:
                ax2.legend(loc=legend_loc, ncol=legend_ncol)

    # handle title
    if title:
        _title = f"{title_prefix}{x}"
        if hue is not None:
            _title += f" by {hue}"
        ax.set_title(_title)

    # handle xlim
    if xlim is not None and xlim:
        # noinspection PyTypeChecker
        ax.set_xlim(xlim)
    else:
        _x_min = np.min(_x_mins)
        _x_max = np.max(_x_maxs)
        _x_offset = (_x_max - _x_min) * x_offset_perc
        # noinspection PyTypeChecker
        ax.set_xlim((_x_min - _x_offset, _x_max + _x_offset))

    return ax


@docstr
@export
def hist_2d(x: str, y: str, data: pd.DataFrame, bins: int = 100, std_cutoff: int = 3, cutoff_perc: float = .01,
            cutoff_abs: float = 0, cmap: str = 'rainbow', ax: plt.Axes = None, color_sigma: str = 'xkcd:red',
            draw_sigma: bool = True, **kwargs) -> plt.Axes:
    """
    generic 2d histogram created by splitting the 2d area into equal sized cells, counting data points in them and
    drawn using pyplot.pcolormesh
    
    :param x: %(x)s
    :param y: %(y)s
    :param data: %(data)s
    :param bins: %(bins)s
    :param std_cutoff: %(std_cutoff)s
    :param cutoff_perc: if less than this percentage of data points is in the cell then the data is ignored [optional]
    :param cutoff_abs: if less than this amount of data points is in the cell then the data is ignored [optional]
    :param cmap: %(cmap)s
    :param ax: %(ax_in)s
    :param color_sigma: color to highlight the sigma range in, must be a valid pyplot.plot color [optional] 
    :param draw_sigma: whether to highlight the sigma range [optional]
    :param kwargs: other keyword arguments passed to pyplot.pcolormesh [optional]
    :return: %(ax_out)s
    """
    _df = data.copy()
    del data

    if std_cutoff is not None:
        _x_min = _df[x].mean() - _df[x].std() * std_cutoff
        _x_max = _df[x].mean() + _df[x].std() * std_cutoff
        _y_min = _df[y].mean() - _df[y].std() * std_cutoff
        _y_max = _df[y].mean() + _df[y].std() * std_cutoff

        # x or y should be in std range
        _df = _df[
            ((_df[x] >= _x_min) & (_df[x] <= _x_max) &
             (_df[y] >= _y_min) & (_df[y] <= _y_max))
        ].reset_index(drop=True)

    _x = _df[x]
    _y = _df[y]

    # Estimate the 2D histogram
    _hist, _x_edges, _y_edges = np.histogram2d(_x, _y, bins=bins)

    # hist needs to be rotated and flipped
    _hist = np.rot90(_hist)
    _hist = np.flipud(_hist)

    # Mask too small counts
    if cutoff_abs is not None:
        _hist = np.ma.masked_where(_hist <= cutoff_abs, _hist)
    if cutoff_perc is not None:
        _hist = np.ma.masked_where(_hist <= _hist.max() * cutoff_perc, _hist)

    # Plot 2D histogram using pcolor
    if ax is None:
        ax = plt.gca()

    _mappable = ax.pcolormesh(_x_edges, _y_edges, _hist, cmap=cmap, **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    _cbar = plt.colorbar(mappable=_mappable, ax=ax)
    _cbar.ax.set_ylabel('count')

    # draw ellipse to mark 1 sigma area
    if draw_sigma:
        _ellipse = patches.Ellipse(xy=(_x.median(), _y.median()), width=_x.std(), height=_y.std(),
                                   edgecolor=color_sigma, fc='None', lw=2, ls=':')
        ax.add_patch(_ellipse)

    return ax


@docstr
@export
def paired_plot(data: pd.DataFrame, cols: Sequence, color: str = None, cmap: str = None, alpha: float = 1,
                **kwargs) -> sns.FacetGrid:
    """
    create a facet grid to analyze various aspects of correlation between two variables using seaborn.PairGrid
    
    :param data: %(data)s
    :param cols: list of exactly two variables to be compared
    :param color: %(color)s
    :param cmap: %(cmap)s
    :param alpha: %(alpha)s
    :param kwargs: other arguments passed to seaborn.PairGrid
    :return: seaborn FacetGrid object with the plots on it
    """
    def _f_corr(_f_x, _f_y, _f_s=10, **_f_kwargs):
        # Calculate the value
        _coef = np.corrcoef(_f_x, _f_y)[0][1]
        # Make the label
        _label = r'$\rho$ = ' + str(round(_coef, 2))

        # Add the label to the plot
        _ax = plt.gca()
        _ax.annotate(_label, xy=(0.2, 0.95 - (_f_s - 10.) / 10.), size=20, xycoords=_ax.transAxes, **_f_kwargs)

    # Create an instance of the PairGrid class.
    _grid = sns.PairGrid(data=data,
                         vars=cols,
                         **kwargs)

    # Map a scatter plot to the upper triangle
    _grid = _grid.map_upper(plt.scatter, alpha=alpha, color=color)
    # Map a corr coef
    _grid = _grid.map_upper(_f_corr)

    # density = True might not be supported in older versions of seaborn / matplotlib
    _grid = _grid.map_diag(plt.hist, bins=30, color=color, alpha=alpha, edgecolor='k', density=True)

    # Map a density plot to the lower triangle
    _grid = _grid.map_lower(sns.kdeplot, cmap=cmap, alpha=alpha)

    # add legend
    _grid.add_legend()

    return _grid


@export
def q_plim(s: pd.Series, q_min: float = .1, q_max: float = .9, offset_perc: float = .1, limit_min_max: bool = False,
           offset=True) -> tuple:
    """
    returns quick x limits for plotting (cut off data not in q_min to q_max quantile)
    
    :param s: pandas Series to truncate
    :param q_min: lower bound quantile [optional]
    :param q_max: upper bound quantile [optional]
    :param offset_perc: percentage of offset to the left and right of the quantile boundaries
    :param limit_min_max: whether to truncate the plot limits at the data limits
    :param offset: whether to apply the offset
    :return: a tuple containing the x limits
    """
    _lower_bound = floor_signif(s.quantile(q=q_min))
    _upper_bound = ceil_signif(s.quantile(q=q_max))

    if _upper_bound == _lower_bound:
        _upper_bound = s.max()
        _lower_bound = s.min()

    if limit_min_max:

        if _upper_bound > s.max():
            _upper_bound = s.max()
        if _lower_bound < s.min():
            _lower_bound = s.min()

    if offset:

        _offset = (_upper_bound - _lower_bound) * offset_perc

    else:

        _offset = 0

    return _lower_bound - _offset, _upper_bound + _offset


@docstr
@export
def levelplot(data: pd.DataFrame, level: str, cols: Union[list, str], hue: str = None, order: Union[list, str] = None,
              hue_order: Union[list, str] = None, func: Callable = distplot, summary_title: bool = True,
              level_title: bool = True, do_print: bool = False, width: int = None, height: int = None,
              return_fig_ax: bool = None, kwargs_subplots_adjust: Mapping = None, kwargs_summary: Mapping = None,
              **kwargs) -> Union[None, tuple]:
    """
    Plots a plot for each specified column for each level of a certain column plus a summary plot
    
    :param data: %(data)s
    :param level: the name of the column to split the plots by, must be in data
    :param cols: the columns to create plots for, defaults to all numeric columns [optional]  
    :param hue: %(hue)s
    :param order: %(order)s
    :param hue_order: %(order)s
    :param func: function to use for plotting, must support 1 positional argument, data, hue, ax and kwargs [optional]
    :param summary_title: whether to automatically set the summary plot title [optional]
    :param level_title: whether to automatically set the level plot title [optional]
    :param do_print: %(do_print)s
    :param width: %(subplot_width)s
    :param height: %(subplot_height)s
    :param return_fig_ax: %(return_fig_ax)s
    :param kwargs_subplots_adjust: other keyword arguments passed to pyplot.subplots_adjust [optional]
    :param kwargs_summary: other keyword arguments passed to summary distplot, if None uses kwargs [optional]
    :param kwargs: other keyword arguments passed to func [optional]
    :return: see return_fig_ax
    """
    # -- init
    # - defaults
    if kwargs_summary is None:
        kwargs_summary = kwargs
    if width is None:
        width = rcParams['fig_width']
    if height is None:
        height = rcParams['fig_height']
    if return_fig_ax is None:
        return_fig_ax = rcParams['return_fig_ax']

    # handle no inplace
    data = pd.DataFrame(data).copy()

    if cols is None:
        cols = data.select_dtypes(include=np.number)

    _levels = _get_ordered_levels(data=data, level=level, order=order)

    if hue is not None:
        _hues = _get_ordered_levels(data=data, level=hue, order=hue_order)
        _hue_str = ' by {}'.format(hue)
    else:
        _hue_str = ''

    _nrows = len(cols)
    _ncols = len(_levels) + 1

    _it_max = _nrows * _ncols

    fig, ax = plt.subplots(nrows=_nrows, ncols=_ncols, figsize=(_ncols * width, _nrows * height))

    _it = -1

    for _col_i, _col in enumerate(cols):

        _ax_summary = get_subax(ax, _col_i, 0, rows_prio=False)  # always plot to col 0 of current row
        # summary plot
        func(_col, data=data, hue=level, ax=_ax_summary, **kwargs_summary)
        if summary_title:
            _ax_summary.set_title('{} by {}'.format(_col, level))

        for _level_i, _level in enumerate(_levels):

            _it += 1

            if do_print:
                progressbar(_it, _it_max, print_prefix='{}_{}'.format(_col, _level))

            _df_level = data[data[level] == _level]

            _ax = get_subax(ax, _col_i, _level_i + 1)

            # level plot
            func(_col, data=_df_level, hue=hue, ax=_ax, **kwargs)

            if level_title:
                _ax.set_title('{}{} - {}={}'.format(_col, _hue_str, level, _level))

    if kwargs_subplots_adjust is not None:
        plt.subplots_adjust(**kwargs_subplots_adjust)

    if do_print:
        progressbar()

    if return_fig_ax:
        return fig, ax
    else:
        plt.show()


@docstr
@export
def get_legends(ax: plt.Axes = None) -> list:
    """
    returns all legends on a given axis, useful if you have a secaxis
    
    :param ax: %(ax_in)s
    :return: list of legends
    """
    if ax is None:
        ax = plt.gca()
    return [_ for _ in ax.get_children() if isinstance(_, Legend)]


# a plot to compare four components of a DataFrame
def four_comp_plot(data, x_1, y_1, x_2, y_2, hue_1=None, hue_2=None, lim=None, return_fig_ax=None, **kwargs):
    # you can pass the hues to use or if none are given the default ones (std,plus/minus) are used
    # you can pass xlim and ylim or assume default (4 std)

    # four components, ie 2 x 2
    if return_fig_ax is None:
        return_fig_ax = rcParams['return_fig_ax']
    if lim is None:
        lim = {'x_1': 'default', 'x_2': 'default', 'y_1': 'default', 'y_2': 'default'}
    _nrows = 2
    _ncols = 2

    # init plot
    fig, ax = plt.subplots(ncols=_ncols, nrows=_nrows)

    # make a copy yo avoid inplace operations
    _df_plot = data.copy()

    _x_std = _df_plot[x_1].std()
    _y_std = _df_plot[y_1].std()

    # type 1: split by size in relation to std
    if hue_1 is None:
        _df_plot['std'] = np.where((np.abs(_df_plot[x_1]) <= 1 * _x_std) & (np.abs(_df_plot[y_1]) <= 1 * _y_std),
                                   '0_std', 'Null')
        _df_plot['std'] = np.where((np.abs(_df_plot[x_1]) > 1 * _x_std) | (np.abs(_df_plot[y_1]) > 1 * _y_std), '1_std',
                                   _df_plot['std'])
        _df_plot['std'] = np.where((np.abs(_df_plot[x_1]) > 2 * _x_std) | (np.abs(_df_plot[y_1]) > 2 * _y_std), '2_std',
                                   _df_plot['std'])
        _df_plot['std'] = np.where((np.abs(_df_plot[x_1]) > 3 * _x_std) | (np.abs(_df_plot[y_1]) > 3 * _y_std), '3_std',
                                   _df_plot['std'])
        _df_plot['std'] = _df_plot['std'].astype('category')

        hue_1 = 'std'

    # type 2: split by plus minus
    if hue_2 is None:
        _df_plot['plus_minus'] = np.where((_df_plot[x_1] <= 0) & (_df_plot[y_1] <= 0), '- -', 'Null')
        _df_plot['plus_minus'] = np.where((_df_plot[x_1] <= 0) & (_df_plot[y_1] > 0), '- +', _df_plot['plus_minus'])
        _df_plot['plus_minus'] = np.where((_df_plot[x_1] > 0) & (_df_plot[y_1] <= 0), '+ -', _df_plot['plus_minus'])
        _df_plot['plus_minus'] = np.where((_df_plot[x_1] > 0) & (_df_plot[y_1] > 0), '+ +', _df_plot['plus_minus'])
        _df_plot['plus_minus'] = _df_plot['plus_minus'].astype('category')

        hue_2 = 'plus_minus'

    _xs = [x_1, x_2]
    _ys = [y_1, y_2]
    _hues = [hue_1, hue_2]

    _xlims = [lim['x_1'], lim['x_2']]
    _ylims = [lim['y_1'], lim['y_2']]

    for _row in range(_nrows):

        for _col in range(_ncols):

            # init
            _ax = get_subax(ax, _row, _col)

            _x_name = _xs[_col]
            _y_name = _ys[_col]
            _hue = _hues[_row]

            _x = _df_plot[_x_name]
            _y = _df_plot[_y_name]

            # scatterplot
            _ax = sns.scatterplot(data=_df_plot, x=_x_name, y=_y_name, hue=_hue, marker='.', ax=_ax, **kwargs)

            # grid 0 line
            _ax.axvline(0, color='k', alpha=.5, linestyle=':')
            _ax.axhline(0, color='k', alpha=.5, linestyle=':')

            # title
            _ax.set_title('%s vs %s, hue: %s' % (_x_name, _y_name, _hue))

            # labels
            _ax.set_xlabel(_x_name)
            _ax.set_ylabel(_y_name)

            # set limits to be 4 std range
            if _xlims[_col] == 'default':

                _x_low = -_x.std() * 4
                if _x.min() > _x_low:
                    _x_low = _x.min()

                _x_high = _x.std() * 4
                if _x.max() < _x_high:
                    _x_high = _x.max()

                _ax.set_xlim([_x_low, _x_high])

            if _ylims[_col] == 'default':

                _y_low = -_y.std() * 4
                if _y.min() > _y_low:
                    _y_low = _y.min()

                _y_high = _y.std() * 4
                if _y.max() < _y_high:
                    _y_high = _y.max()

                _ax.set_ylim([_y_low, _y_high])

    if return_fig_ax:
        return fig, ax
    else:
        plt.tight_layout()
        plt.show()


@docstr
@export
def facet_wrap(func: Callable, data: pd.DataFrame, facet: Union[list, str], *args, facet_type: str = None,
               col_wrap: int = 4, width: int = None, height: int = None,
               catch_error: bool = True, return_fig_ax: bool = None, sharex: bool = False,
               sharey: bool = False, show_xlabel: bool = True, x_tick_rotation: int = None, y_tick_rotation: int = None,
               ax_title: str = 'set', order: Union[list, str] = None, subplots_kws: Mapping = None, **kwargs):
    """
    modeled after r's facet_wrap function. Wraps a number of subplots onto a 2d grid of subplots while creating
    a new line after col_wrap columns. Uses a given plot function and creates a new plot for each facet level.

    :param func: Any plot function. Must support keyword arguments data and ax
    :param data: %(data)s
    :param facet: The column / list of columns to facet over.
    :param args: passed to func
    :param facet_type: one of ['group', 'cols', None]. 
        If group facet is treated as the column creating the facet levels and a subplot is created for each level. 
        If cols each facet is in turn passed as the first positional argument to the plot function func. 
        If None then the facet_type is inferred: a single facet value will be treated as group and multiple
        facet values will be treated as cols. 
    :param col_wrap: %(col_wrap)s
    :param width: %(subplot_width)s
    :param height: %(subplot_height)s
    :param catch_error: whether to keep going in case of an error being encountered in the plot function [optional] 
    :param return_fig_ax: %(return_fig_ax)s
    :param sharex: %(sharex)s
    :param sharey: %(sharey)s
    :param show_xlabel: whether to show the x label for each subplot
    :param x_tick_rotation: x tick rotation for each subplot
    :param y_tick_rotation: y tick rotation for each subplot
    :param ax_title: one of ['set','hide'], if set sets axis title to facet name, if hide forcefully hids axis title
    :param order: %(order)s
    :param subplots_kws: other keyword arguments passed to pyplot.subplots
    :param kwargs: other keyword arguments passed to func
    :return: %(fig_ax_out)s

    **Examples**

    Check out the `example notebook <https://colab.research.google.com/drive/1bAEFRoWJgwPzkEqOoPBHVX849qQjxLYC>`_
    """
    # -- init
    # - defaults
    if width is None:
        width = rcParams['fig_width']
    if height is None:
        height = rcParams['fig_height']
    if return_fig_ax is None:
        return_fig_ax = rcParams['return_fig_ax']
    if subplots_kws is None:
        subplots_kws = {}

    # - handle no inplace
    _df = data.copy()
    del data
    _facet = None
    _row = None
    _col = None

    # if it is a list of column names we will melt the df together
    if facet_type is None:
        if is_list_like(facet):
            facet_type = 'cols'
        else:
            facet_type = 'group'

    # process the facets
    if facet_type == 'cols':
        _facets = facet
    else:
        _df['_facet'] = concat_cols(_df, facet)
        facet = '_facet'
        _facets = _get_ordered_levels(_df, facet, order)

    # init a grid
    if len(_facets) > col_wrap:
        _ncols = col_wrap
        _nrows = int(np.ceil(len(_facets) / _ncols))
    else:
        _ncols = len(_facets)
        _nrows = 1

    fig, ax = plt.subplots(ncols=_ncols, nrows=_nrows, figsize=(width * _ncols, height * _nrows), **subplots_kws)
    _ax_list = ax_as_list(ax)

    # loop facets
    for _it, _facet in enumerate(_facets):

        _col = _it % _ncols
        _row = _it // _ncols
        _ax = _ax_list[_it]

        # get df facet
        _facet = _facets[_it]

        # for list set target to be in line with facet to ensure proper naming
        if facet_type == 'cols':
            _df_facet = _df.copy()
            _args = assert_list(_facet) + list(args)
        else:
            _df_facet = _df[_df[facet] == _facet]
            _args = args

        # apply function on target (try catch)
        if catch_error:
            try:
                func(*_args, data=_df_facet, ax=_ax, **kwargs)
            except Exception as _exc:
                warnings.warn('could not plot facet {} with exception {}, skipping. '
                              'For details use catch_error=False'.format(_exc, _facet))
                _ax.set_axis_off()
                continue
        else:
            func(*_args, data=_df_facet, ax=_ax, **kwargs)

        # set axis title to facet or hide it or do nothing (depending on preference)
        if ax_title == 'set':
            _ax.set_title(_facet)
        elif ax_title == 'hide':
            _ax.set_title('')

        # tick rotation
        if x_tick_rotation is not None:
            _ax.xaxis.set_tick_params(rotation=x_tick_rotation)
        if y_tick_rotation is not None:
            _ax.yaxis.set_tick_params(rotation=y_tick_rotation)

        # hide x label (if appropriate)
        if not show_xlabel:
            _ax.set_xlabel('')

    # hide unused axes
    for __col in range(_col + 1, _ncols):
        ax[_row, __col].set_axis_off()
    # share xy
    if sharex or sharey:
        share_xy(ax, x=sharex, y=sharey)

    if return_fig_ax:
        return fig, ax
    else:
        plt.show()


@docstr
@export
def get_subax(ax: Union[plt.Axes, np.ndarray], row: int = None, col: int = None, rows_prio: bool = True) -> plt.Axes:
    """
    shorthand to get around the fact that ax can be a 1D array or a 2D array (for subplots that can be 1x1,1xn,nx1)
    
    :param ax: %(ax_in)s
    :param row: %(row)s
    :param col: %(col)s
    :param rows_prio: decides if to use row or col in case of a 1xn / nx1 shape (False means cols get priority)
    :return: %(ax_out)s
    """

    if isinstance(ax, np.ndarray):
        _dims = len(ax.shape)
    else:
        _dims = 0

    if _dims == 0:
        _ax = ax
    elif _dims == 1:
        if rows_prio:
            _ax = ax[row]
        else:
            _ax = ax[col]
    else:
        _ax = ax[row, col]

    return _ax


@docstr
@export
def ax_as_list(ax: Union[plt.Axes, np.ndarray]) -> list:
    """
    takes any Axes and turns them into a list
    
    :param ax: %(ax_in)s
    :return: List containing the subaxes
    """
    
    if isinstance(ax, np.ndarray):
        _dims = len(ax.shape)
    else:
        _dims = 0

    if _dims == 0:
        _ax_list = [ax]
    elif _dims == 1:
        _ax_list = list(ax)
    else:
        _ax_list = list(ax.flatten())

    return _ax_list


@docstr
@export
def ax_as_array(ax: Union[plt.Axes, np.ndarray]) -> np.ndarray:
    """
    takes any Axes and turns them into a numpy 2D array

    :param ax: %(ax_in)s
    :return: Numpy 2D array containing the subaxes
    """
    if isinstance(ax, np.ndarray):
        if len(ax.shape) == 2:
            return ax
        else:
            return ax.reshape(-1, 1)
    else:
        return np.array([ax]).reshape(-1, 1)


# bubble plot
def bubbleplot(x, y, hue, s, text=None, text_as_label=False, data=None, s_factor=250, palette=None,
               hue_order=None, x_range_factor=5, y_range_factor=5, show_std=False, ax=None,
               legend_loc='right', text_kws=None):
    if palette is None:
        palette = rcParams['palette']
    if text_kws is None:
        text_kws = {}
    if ax is None:
        ax = plt.gca()

    _df = data.copy()
    _df = _df[~((_df[x].isnull()) | (_df[y].isnull()) | (_df[s].isnull()))].reset_index(drop=True)

    if hue_order is not None:
        _df['_sort'] = _df[hue].apply(lambda _: hue_order.index(_))
        _df = _df.sort_values(by=['_sort'])

    _df = _df.reset_index(drop=True)

    _x = _df[x]
    _y = _df[y]
    _s = _df[s] * s_factor

    if text is not None:
        _text = _df[text]
    else:
        _text = pd.Series()

    if isinstance(palette, Mapping):
        _df['_color'] = _df[hue].apply(lambda _: palette[_])
    elif is_list_like(palette):
        _df['_color'] = palette[:_df.index.max() + 1]
    else:
        _df['color'] = palette

    # draw ellipse to mark 1 sigma area
    if show_std:

        _x_min = None
        _x_max = None
        _y_min = None
        _y_max = None

        for _index, _row in _df.iterrows():

            _ellipse = patches.Ellipse(xy=(_row[x], _row[y]), width=_row[x + '_std'] * 2, height=_row[y + '_std'] * 2,
                                       edgecolor=_row['_color'], fc='None', lw=2, ls=':')
            ax.add_patch(_ellipse)

            _x_min_i = _row[x] - _row[x + '_std'] * 1.05
            _x_max_i = _row[x] + _row[x + '_std'] * 1.05
            _y_min_i = _row[y] - _row[y + '_std'] * 1.05
            _y_max_i = _row[y] + _row[y + '_std'] * 1.05

            if _x_min is None:
                _x_min = _x_min_i
            elif _x_min_i < _x_min:
                _x_min = _x_min_i
            if _x_max is None:
                _x_max = _x_max_i
            elif _x_max_i > _x_max:
                _x_max = _x_max_i
            if _y_min is None:
                _y_min = _y_min_i
            elif _y_min_i < _y_min:
                _y_min = _y_min_i
            if _y_max is None:
                _y_max = _y_max_i
            elif _y_max_i > _y_max:
                _y_max = _y_max_i

    else:
        # scatter for bubbles
        ax.scatter(x=_x, y=_y, s=_s, label='__nolegend__', facecolor=_df['_color'], edgecolor='black', alpha=.75)

        _x_range = _x.max() - _x.min()
        _x_min = _x.min() - _x_range / x_range_factor
        _x_max = _x.max() + _x_range / x_range_factor

        _y_range = _y.max() - _y.min()
        _y_min = _y.min() - _y_range / y_range_factor
        _y_max = _y.max() + _y_range / y_range_factor

    # plot fake data for legend (a little hacky)
    if text_as_label:

        _xlim_before = ax.get_xlim()

        for _it in range(len(_x)):
            _label = _text[_it]
            # fake data
            ax.scatter(x=-9999, y=_y[_it], label=_label, facecolor=_df['_color'].loc[_it], s=200, edgecolor='black',
                       alpha=.75)

        ax.set_xlim(_xlim_before)

    if (text is not None) and (not text_as_label):
        for _it in range(len(_text)):

            _ = ''

            if (not np.isnan(_x.iloc[_it])) and (not np.isnan(_y.iloc[_it])):
                ax.text(x=_x.iloc[_it], y=_y.iloc[_it], s=_text.iloc[_it], horizontalalignment='center',
                        verticalalignment='center', **text_kws)

    # print(_x_min,_x_max)

    ax.set_xlim(_x_min, _x_max)
    ax.set_ylim(_y_min, _y_max)

    ax.set_xlabel(_x.name)
    ax.set_ylabel(_y.name)

    if text_as_label and (legend_loc in ['bottom', 'right']):
        legend_outside(ax, loc=legend_loc)
    else:
        ax.legend(loc=legend_loc)

    # title
    ax.set_title(hue)

    return ax


def bubblecountplot(x, y, hue, data, agg_function='median', show_std=True, top_nr=None, n_quantiles=10,
                    other_name='other', dropna=True, float_format='.2f', text_end='', **kwargs):
    _df = data.copy()

    if dropna:
        _df = _df[~_df[hue].isnull()]

    if hue in _df.select_dtypes(include=np.number):

        _n = n_quantiles

        if top_nr is not None:
            if top_nr < n_quantiles:
                _n = top_nr

        _df[hue] = quantile_split(_df[hue], _n)

    if top_nr is not None:
        _df[hue] = top_n_coding(_df[hue], n=top_nr, other_name=other_name)

    # handle na
    _df[x] = _df[x].fillna(_df[x].dropna().agg(agg_function))
    _df[y] = _df[y].fillna(_df[y].dropna().agg(agg_function))

    # build agg dict
    _df['_count'] = 1
    _df = _df.groupby([hue]).agg({x: [agg_function, 'std'], y: [agg_function, 'std'], '_count': 'count'}).reset_index()
    if x != y:
        _columns = [hue, x, x + '_std', y, y + '_std', '_count']
    else:
        _columns = [hue, x, x + '_std', '_count']
    _df.columns = _columns
    _df['_perc'] = _df['_count'] / _df['_count'].sum() * 100
    _df['_count_text'] = _df.apply(lambda _: "{:,}".format(_['_count']), axis=1)
    _df['_perc_text'] = np.round(_df['_perc'], 2)
    _df['_perc_text'] = _df['_perc_text'].astype(str) + '%'

    if show_std:
        _df['_text'] = _df[hue].astype(str) + '(' + _df['_count_text'] + ')' + '\n' \
                       + 'x:' + _df[x].apply(lambda _: format(_, float_format)) + r'$\pm$' + _df[x + '_std'].apply(
            lambda _: format(_, float_format)) + '\n' \
                       + 'y:' + _df[y].apply(lambda _: format(_, float_format)) + r'$\pm$' + _df[y + '_std'].apply(
            lambda _: format(_, float_format))
    else:
        _df['_text'] = _df[hue].astype(str) + '\n' + _df['_count_text'] + '\n' + _df['_perc_text']

    _df['_text'] += text_end

    bubbleplot(x=x, y=y, hue=hue, s='_perc', text='_text', data=_df, show_std=show_std, **kwargs)


@docstr
@export
def rmsdplot(x: str, data: pd.DataFrame, groups: Union[Sequence, str] = None, hue: str = None,
             hue_order: Union[Sequence, str] = None, cutoff: float = 0, ax: plt.Axes = None,
             color_as_balance: bool = False, balance_cutoff: float = None, rmsd_as_alpha: bool = False,
             sort_by_hue: bool = False, palette=None, barh_kws=None, **kwargs):
    """
    creates a seaborn.barplot showing the rmsd calculating :func:`~hhpy.ds.df_rmsd`
    
    :param x: %(x)s
    :param data: %(data)s
    :param groups: the columns to calculate the rmsd for, defaults to all columns [optional]
    :param hue: %(hue)s
    :param hue_order: %(order)s 
    :param cutoff: drop rmsd values smaller than cutoff [optional]
    :param ax: %(ax_in)s
    :param color_as_balance: Whether to color the bars based on how balanced (based on maxperc values) the levels are
        [optional]
    :param balance_cutoff: If specified: all bars with worse balance (based on maxperc values) than cutoff are shown
        in red [optional]
    :param rmsd_as_alpha: Whether to use set the alpha values of the columns based on the rmsd value [optional]
    :param sort_by_hue: Whether to sort the plot by hue value [optional]
    :param palette: %(palette)s
    :param barh_kws: other keyword arguments passed to seaborn.barplot [optional]
    :param kwargs: other keyword arguments passed to :func:`hhpy.ds.rf_rmsd` [optional]
    :return: %(ax_out)s

    **Examples**

    Check out the `example notebook <https://colab.research.google.com/drive/1wvkYK80if0okXJGf1j2Kl-SxXZdl-97k>`_
    """
    if palette is None:
        palette = rcParams['palette']
    if barh_kws is None:
        barh_kws = {}
    _data = data.copy()
    del data

    if hue is not None and hue_order is not None:
        _data = _data.query('{} in @hue_order'.format(hue))

    _df_rmsd = df_rmsd(x=x, df=_data, groups=groups, hue=hue, sort_by_hue=sort_by_hue, **kwargs)
    _df_rmsd = _df_rmsd[_df_rmsd['rmsd'] >= cutoff]

    if hue is not None:
        _df_rmsd_no_hue = df_rmsd(x=x, df=_data, groups=groups, include_rmsd=False, **kwargs)
    else:
        _df_rmsd_no_hue = pd.DataFrame()

    if isinstance(x, list):
        if hue is None:
            _df_rmsd['label'] = concat_cols(_df_rmsd, ['x', 'group'], sep=' X ')
        else:
            _df_rmsd['label'] = concat_cols(_df_rmsd, ['x', 'group', hue], sep=' X ')
    else:
        if hue is None:
            _df_rmsd['label'] = _df_rmsd['group']
        else:
            _df_rmsd['label'] = concat_cols(_df_rmsd, ['group', hue], sep=' X ')

    _df_rmsd['rmsd_scaled'] = _df_rmsd['rmsd'] / _df_rmsd['rmsd'].max()

    # get colors
    _rgba_colors = np.zeros((len(_df_rmsd), 4))
    _hues = []

    if hue is not None:

        _hues = _get_ordered_levels(data=_df_rmsd, level=hue, order=hue_order, x=x)

        if isinstance(palette, Mapping):
            _df_rmsd['_color'] = _df_rmsd[hue].apply(lambda _: palette[_])
        elif is_list_like(palette):
            _df_rmsd['_color'] = _df_rmsd[hue].apply(lambda _: palette[list(_hues).index(_)])
        else:
            _df_rmsd['_color'] = palette

        _rgba_colors[:, 0] = _df_rmsd['_color'].apply(lambda _: Color(_).red)
        _rgba_colors[:, 1] = _df_rmsd['_color'].apply(lambda _: Color(_).green)
        _rgba_colors[:, 2] = _df_rmsd['_color'].apply(lambda _: Color(_).blue)

    elif color_as_balance:

        if balance_cutoff is None:

            _rgba_colors[:, 0] = _df_rmsd['maxperc']  # for red the first column needs to be one
            _rgba_colors[:, 2] = 1 - _df_rmsd['maxperc']  # for blue the third column needs to be one

        else:

            _rgba_colors[:, 0] = np.where(_df_rmsd['maxperc'] >= balance_cutoff, 1, 0)
            _rgba_colors[:, 2] = np.where(_df_rmsd['maxperc'] < balance_cutoff, 1, 0)

    else:
        _rgba_colors[:, 2] = 1  # for blue the third column needs to be one

    # the fourth column needs to be alphas
    if rmsd_as_alpha:
        _rgba_colors[:, 3] = _df_rmsd['rmsd_scaled']
    else:
        _rgba_colors[:, 3] = 1

    if ax is None:
        ax = plt.gca()

    # make positions from labels

    if hue is not None:
        _pos_factor = .8
    else:
        _pos_factor = 1

    _df_rmsd['pos'] = _df_rmsd.index * _pos_factor

    if (hue is not None) and (not sort_by_hue):
        # iterate over rows and add to pos if label changes
        for _row in range(1, len(_df_rmsd)):
            if _df_rmsd['group'].iloc[_row] != _df_rmsd['group'].iloc[_row - 1]:
                _df_rmsd['pos'][_row:] = _df_rmsd['pos'][_row:] + _pos_factor

        # make a df of the average positions for each group
        _df_ticks = _df_rmsd.groupby('group').agg({'pos': 'mean'}).reset_index()  # 'maxperc':'max'
        _df_ticks = pd.merge(_df_ticks, _df_rmsd_no_hue[['group', 'maxperc']])  # get maxperc from global value
    else:
        _df_ticks = pd.DataFrame()

    ax.barh(_df_rmsd['pos'], _df_rmsd['rmsd'], color=_rgba_colors, **barh_kws)

    _y_colors = None

    if (hue is not None) and (not sort_by_hue):

        _y_pos = _df_ticks['pos']
        _y_lab = _df_ticks['group']
        # color
        if balance_cutoff is not None:
            _y_colors = np.where(_df_ticks['maxperc'] > balance_cutoff, sns.xkcd_rgb['red'], 'k')

    else:

        _y_pos = _df_rmsd['pos']

        if not is_list_like(x):
            _y_lab = _df_rmsd['group']
        elif not is_list_like(groups):
            _y_lab = _df_rmsd['x']
        else:
            _y_lab = concat_cols(_df_rmsd, ['x', 'group'], sep=' X ')

    ax.set_yticks(_y_pos)
    ax.set_yticklabels(_y_lab)

    if _y_colors is not None:
        for _y_tick, _color in zip(ax.get_yticklabels(), _y_colors):
            _y_tick.set_color(_color)

    if hue is None:
        _offset = _pos_factor
    else:
        _offset = _pos_factor * len(_hues)

    # noinspection PyTypeChecker
    ax.set_ylim([_y_pos.min() - _offset, _y_pos.max() + _offset])
    ax.invert_yaxis()

    # create legend for hues
    if hue is not None:

        _patches = []
        for _hue, _color, _count in _df_rmsd[[hue, '_color', 'count']].drop_duplicates().values:
            _patches.append(patches.Patch(color=_color, label='{} (n={:,})'.format(_hue, _count)))
        ax.legend(handles=_patches)

    # check if standardized
    _x_label_suffix = ''

    if 'standardize' in kwargs.keys():
        if kwargs['standardize']:
            _x_label_suffix += ' [std]'

    if not is_list_like(x):
        ax.set_title('Root Mean Square Difference for {}'.format(x))
        ax.set_xlabel('RMSD: {}{}'.format(x, _x_label_suffix))
    elif not is_list_like(groups):
        ax.set_title('Root Mean Square Difference for {}'.format(groups))
        ax.set_xlabel('RMSD: {}{}'.format(groups, _x_label_suffix))
    else:
        ax.set_title('Root Mean Square Difference')

    return ax


# plot agg
def aggplot(x, data, group, hue=None, hue_order=None, width=16, height=9 / 2,
            p_1_0=True, palette=None, sort_by_hue=False, return_fig_ax=None, agg=None, p=False,
            legend_loc='upper right', aggkws=None, subplots_kws=None, subplots_adjust_kws=None, **kwargs):
    if return_fig_ax is None:
        return_fig_ax = rcParams['return_fig_ax']
    if palette is None:
        palette = rcParams['palette']
    if agg is None:
        agg = ['mean', 'median', 'std']
    if aggkws is None:
        aggkws = {}
    if subplots_kws is None:
        subplots_kws = {}
    if subplots_adjust_kws is None:
        subplots_adjust_kws = {'top': .95, 'hspace': .25, 'wspace': .35}

    # avoid inplace operations
    data = pd.DataFrame(data).copy()
    _len = len(agg) + 1 + p

    _x = x
    _group = group

    # EITHER x OR group can be a list (hue cannot be a lists)
    if is_list_like(x) and is_list_like(group):

        warnings.warn('both x and group cannot be a list, setting group = {}'.format(group[0]))
        _x_is_list = True
        _group_is_list = False
        _group = group[0]
        _ncols = len(x)
        _nrows = _len

    elif isinstance(x, list):

        _x_is_list = True
        _group_is_list = False
        _group = group
        _ncols = len(x)
        _nrows = _len

    elif isinstance(group, list):

        _x_is_list = False
        _group_is_list = True
        _ncols = len(group)
        _nrows = _len

    else:

        _x_is_list = False
        _group_is_list = False
        _ncols = int(np.floor(_len / 2))
        _nrows = int(np.ceil(_len / 2))

    fig, ax = plt.subplots(figsize=(width * _ncols, height * _nrows), nrows=_nrows, ncols=_ncols, **subplots_kws)

    _it = -1

    for _col in range(_ncols):

        if _x_is_list:
            _x = x[_col]
        if _group_is_list:
            _group = group[_col]

        _df_agg = df_agg(x=_x, group=_group, hue=hue, df=data, agg=agg, p=p, **aggkws)

        if hue is not None:
            if sort_by_hue:
                _sort_by = [hue, _group]
            else:
                _sort_by = [_group, hue]
            _df_agg = _df_agg.sort_values(by=_sort_by).reset_index(drop=True)
            _label = '_label'
            _df_agg[_label] = concat_cols(_df_agg, [_group, hue], sep='_').astype('category')
            _hues = _get_ordered_levels(data=data, level=hue, order=hue_order, x=x)

            if isinstance(palette, Mapping):
                _df_agg['_color'] = _df_agg[hue].apply(lambda _: palette[_])
            elif is_list_like(palette):
                _df_agg['_color'] = _df_agg[hue].apply(lambda _: palette[list(_hues).index(_)])
            else:
                _df_agg['_color'] = palette

        else:
            _label = _group

        for _row in range(_nrows):

            _it += 1

            if _x_is_list or _group_is_list:
                _index = _row
            else:
                _index = _it

            _ax = get_subax(ax, _row, _col)

            if _index >= _len:
                _ax.set_axis_off()
                continue

            _agg = list(_df_agg)[1:][_index]

            # one color per graph (if no hue)
            if hue is None:
                _df_agg['_color'] = palette[_index]

            # handle hue grouping
            if hue is not None:
                _pos_factor = .8
            else:
                _pos_factor = 1

            _df_agg['pos'] = _df_agg.index

            if (hue is not None) and (not sort_by_hue):
                # iterate over rows and add to pos if label changes
                for _row_2 in range(1, len(_df_agg)):
                    if _df_agg[_group].iloc[_row_2] != _df_agg[_group].iloc[_row_2 - 1]:
                        _df_agg['pos'][_row_2:] = _df_agg['pos'][_row_2:] + _pos_factor

                # make a df of the average positions for each group
                _df_ticks = _df_agg.groupby(_group).agg({'pos': 'mean'}).reset_index()
            else:
                _df_ticks = pd.DataFrame()

            _ax.barh('pos', _agg, color='_color', label=_agg, data=_df_agg, **kwargs)

            if (hue is not None) and (not sort_by_hue):
                _ax.set_yticks(_df_ticks['pos'])
                _ax.set_yticklabels(_df_ticks[_group])
            else:
                _ax.set_yticks(_df_agg['pos'])
                _ax.set_yticklabels(_df_agg[_group])

            _ax.invert_yaxis()
            _ax.set_xlabel(_x + '_' + _agg)
            _ax.set_ylabel(_group)

            # create legend for hues
            if hue is not None:
                _patches = []
                for _hue, _color in _df_agg[[hue, '_color']].drop_duplicates().values:
                    _patches.append(patches.Patch(color=_color, label=_hue))

                _ax.legend(handles=_patches)
            else:
                _ax.legend(loc=legend_loc)

            # range of p is between 0 and 1
            if _agg == 'p' and p_1_0:
                # noinspection PyTypeChecker
                _ax.set_xlim([0, 1])

    if _x_is_list:
        _x_title = ','.join(x)
    else:
        _x_title = _x

    if _group_is_list:
        _group_title = ','.join(group)
    else:
        _group_title = _group

    _title = _x_title + ' by ' + _group_title
    if hue is not None:
        _title = _title + ' per ' + hue

    plt.suptitle(_title, size=16)
    plt.subplots_adjust(**subplots_adjust_kws)

    if return_fig_ax:
        return fig, ax
    else:
        plt.show()


def aggplot2d(x, y, data, aggfunc='mean', ax=None, x_int=None, time_int=None,
              color=None, as_abs=False):
    if color is None:
        color = rcParams['palette'][0]
    # time int should be something like '<M8[D]'
    # D can be any datetime unit from numpy https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.datetime.html

    _y_agg = '{}_{}'.format(y, aggfunc)
    _y_std = '{}_std'.format(y)

    # preprocessing
    data = pd.DataFrame(data).copy()

    if as_abs:
        data[y] = np.abs(data[y])
    if x_int is not None:
        data[x] = np.round(data[x] / x_int) * x_int
    if time_int is not None:
        data[x] = data[x].astype('<M8[{}]'.format(time_int))

    # agg

    data = data.groupby([x]).agg({y: [aggfunc, 'std']}).set_axis([_y_agg, _y_std], axis=1, inplace=False).reset_index()

    if ax is None:
        ax = plt.gca()

    ax.plot(data[x], data[_y_agg], color=color, label=_y_agg)
    ax.fill_between(data[x], data[_y_agg] + data[_y_std], data[_y_agg] - data[_y_std], color='xkcd:cyan', label=_y_std)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()

    return ax


@export
def insert_linebreak(s: str, pos: int = None, frac: float = None, max_breaks: int = None) -> str:
    """
    used to insert linebreaks in strings, useful for formatting axes labels

    :param s: string to insert linebreaks into
    :param pos: inserts a linebreak every pos characters [optional]
    :param frac: inserts a linebreak after frac percent of characters [optional]
    :param max_breaks: maximum number of linebreaks to insert [optional]
    :return: string with the linebreaks inserted
    """
    _s = s + ''

    if pos is not None:
        _pos = pos
        _frac = int(np.ceil(len(_s) / _pos))
    elif frac is not None:
        _pos = int(np.ceil(len(_s) / frac))
        _frac = frac
    else:
        _pos = None
        _frac = None

    _pos_i = 0

    if max_breaks is not None:
        _max = np.min([max_breaks, _frac - 1])
    else:
        _max = _frac - 1

    for _it in range(_max):

        _pos_i += _pos
        if _it > 0:
            _pos_i += 1  # needed because of from 0 indexing
        _s = _s[:_pos_i] + '\n' + _s[_pos_i:]

    # remove trailing newlines
    if _s[-1:] == '\n':
        _s = _s[:-1]

    return _s


@docstr
@export
def ax_tick_linebreaks(ax: plt.Axes = None, x: bool = True, y: bool = True, **kwargs) -> None:
    """
    uses insert_linebreaks to insert linebreaks into the axes ticklabels
    
    :param ax: %(ax_in)s
    :param x: whether to insert linebreaks into the x axis label [optional]
    :param y: whether to insert linebreaks into the y axis label [optional]
    :param kwargs: other keyword arguments passed to insert_linebreaks
    :return: None
    """
    if ax is None:
        ax = plt.gca()

    if x:
        ax.set_xticklabels([insert_linebreak(_item.get_text(), **kwargs) for _item in ax.get_xticklabels()])
    if y:
        ax.set_yticklabels([insert_linebreak(_item.get_text(), **kwargs) for _item in ax.get_yticklabels()])


@docstr
@export
def annotate_barplot(ax: plt.Axes = None, x: Sequence = None, y: Sequence = None, ci: bool = True, 
                     ci_newline: bool = True, adj_ylim: float = .05, nr_format: str = None,
                     ha: str = 'center', va: str = 'center', offset: int = None,
                     **kwargs) -> plt.Axes:
    """
    automatically annotates a barplot with bar values and error bars (if present). Currently does not work with ticks!
    
    :param ax: %(ax_in)s 
    :param x: %(x)s
    :param y: %(y)s
    :param ci: whether to annotate error bars [optional]
    :param ci_newline: whether to add a newline between values and error bar values [optional] 
    :param adj_ylim: whether to automatically adjust the plot y limits to fit the annotations [optional]
    :param nr_format: %(number_format)s
    :param ha: horizontal alignment [optional]
    :param va: vertical alignment [optional]
    :param offset: offset between bar top and annotation center, defaults to rcParams[font.size] [optional]
    :param kwargs: other keyword arguments passed to pyplot.annotate
    :return: %(ax_out)s
    """
    # -- init
    # - defaults
    if nr_format is None:
        nr_format = rcParams['float_format']
    if offset is None:
        offset = plt.rcParams['font.size']
    if ax is None:
        ax = plt.gca()
    # catch font warnings
    logging.getLogger().setLevel(logging.CRITICAL)

    _adj_plus = False
    _adj_minus = False

    if ci_newline:
        _ci_sep = '\n'
        _offset = offset + 5
    else:
        _ci_sep = ''
        _offset = offset

    for _it, _patch in enumerate(ax.patches):

        try:

            if x is None:
                _x = _patch.get_x() + _patch.get_width() / 2.
            elif is_list_like(x):
                _x = x[_it]
            else:
                _x = x

            if y is None:
                _y = _patch.get_height()
            elif is_list_like(y):
                _y = y[_it]
            else:
                _y = y

            _val = _patch.get_height()

            if _val > 0:
                _adj_plus = True
            if _val < 0:
                _adj_minus = True

            if np.isnan(_val):
                continue

            _val_text = format(_val, nr_format)

            _annotate = r'${}$'.format(_val_text)

            # TODO: HANDLE CAPS

            if ci and ax.lines.__len__() > _it:
                _line = ax.lines[_it]

                _line_y = _line.get_xydata()[:, 1]
                _ci = (_line_y[1] - _line_y[0]) / 2

                if not np.isnan(_ci):
                    _ci_text = format(_ci, nr_format)
                    _annotate = r'${}$'.format(_val_text) + _ci_sep + r'$\pm{}$'.format(_ci_text)

            ax.annotate(_annotate, (_x, _y), ha=ha, va=va, xytext=(0, np.sign(_val) * _offset),
                        textcoords='offset points', **kwargs)

        except Exception as exc:
            print(exc)

    if adj_ylim:

        _ylim = list(ax.get_ylim())
        _y_adj = (_ylim[1] - _ylim[0]) * adj_ylim
        if _adj_minus:
            _ylim[0] = _ylim[0] - _y_adj
        if _adj_plus:
            _ylim[1] = _ylim[1] + _y_adj
        # noinspection PyTypeChecker
        ax.set_ylim(_ylim)

    logging.getLogger().setLevel(logging.DEBUG)

    return ax


@docstr
@export
def animplot(data: pd.DataFrame = None, x: str = 'x', y: str = 'y', t: str = 't', lines: Mapping = None,
             max_interval: int = None, time_per_frame: int = 200, mode: str = None,
             title: bool = True, title_prefix: str = '', t_format: str = None, fig: plt.Figure = None,
             ax: plt.Axes = None, color: str = None, label: str = None, legend: bool = False, legend_out: bool = False,
             legend_kws: Mapping = None, xlim: tuple = None, ylim: tuple = None,
             ax_facecolor: Union[str, Mapping] = None, grid: bool = False, vline: Union[Sequence, float] = None,
             **kwargs) -> Union[HTML, FuncAnimation]:
    """
    wrapper for FuncAnimation to be used with pandas DataFrames. Assumes that you have a DataFrame containing
    one data point for each x-y-t combination.

    If mode is set to jshtml the function is optimized for use with Jupyter Notebook and returns an
    Interactive JavaScript Widget.

    :param data: %(data)s 
    :param x: %(x_novec)s
    :param y: %(y_novec)s
    :param t: %(t_novec)s
    :param lines: you can also pass lines that you want to animate. Details to follow [optional]
    :param max_interval: max interval at which to abort the animation [optional]
    :param time_per_frame: time per frame [optional]
    :param mode: one of the below [optional]

        * ``matplotlib``: Return the matplotlib FuncAnimation object

        * ``html``: Returns an HTML5 movie (You need to install ffmpeg for this to work)

        * ``jshtml``: Returns an interactive Javascript Widget

    :param title: whether to set the time as plot title [optional]
    :param title_prefix: title prefix to be put in front of the time if title is true [optional]
    :param t_format: format string used to format the time variable in the title [optional]
    :param fig: figure to plot on [optional]
    :param ax: axes to plot on [optional]
    :param color: %(color)s
    :param label: %(label)s
    :param legend: %(legend)s
    :param legend_out: %(legend_out)s
    :param legend_kws: %(legend_kws)s
    :param xlim: %(xlim)s
    :param ylim: %(ylim)s
    :param ax_facecolor: passed to ax.set_facecolor, can also be a conditional mapping to change the facecolor at
        specific timepoints t [optional]
    :param grid: %(grid)s
    :param vline: %(vline)s
    :param kwargs: other keyword arguments passed to pyplot.plot
    :return: see mode

    **Examples**

    Check out the `example notebook <https://drive.google.com/open?id=1hJRfZn3Zwnc1n4cK7h2-UPSEj4BmsxhY>`_
    """
    # example for lines (a list of dicts)
    # lines = [{'line':line,'data':data,'x':'x','y':'y','t':'t'}]

    # -- init
    # - defaults
    if mode is None:
        mode = rcParams['animplot.mode']
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()
    if legend_kws is None:
        legend_kws = {}
    # - handle no inplace
    data = pd.DataFrame(data).copy()
    # - preprocessing
    # if t is the index: save to regular column
    if (t == 'index') and (t not in data.columns):
        data[t] = data.index

    _args = {'data': data, 'x': x, 'y': y, 't': t}

    _ax_list = ax_as_list(ax)

    # init lines
    if lines is None:

        _ax = _ax_list[0]

        lines = []

        _len = 1
        if is_list_like(x):
            _len = np.max([_len, len(x)])
        if is_list_like(y):
            _len = np.max([_len, len(y)])

        for _it in range(_len):

            if is_list_like(x):
                _x = x[_it]
            else:
                _x = x

            if is_list_like(y):
                _y = y[_it]
            else:
                _y = y

            if is_list_like(vline):
                _vline = vline[_it]
            else:
                _vline = vline

            if isinstance(color, Mapping):
                if _y in color.keys():
                    _color = color[_y]
                else:
                    _color = None
            elif is_list_like(color):
                _color = color[_it]
            else:
                _color = color

            _kwargs = deepcopy(kwargs)
            _kwargs_keys = list(_kwargs.keys())

            # defaults
            if len(list_intersection(['markerfacecolor', 'mfc'], _kwargs_keys)) == 0:
                _kwargs['markerfacecolor'] = _color
            if len(list_intersection(['markeredgecolor', 'mec'], _kwargs_keys)) == 0:
                _kwargs['markeredgecolor'] = _color
                if len(list_intersection(['markeredgewidth', 'mew'], _kwargs_keys)) == 0:
                    _kwargs['markeredgewidth'] = 1

            if label is None:
                _label = _y
            elif isinstance(label, Mapping):
                _label = label[_y]
            elif is_list_like(label):
                _label = label[_it]
            else:
                _label = label

            lines += [{
                'line': _ax.plot([], [], label=_label, color=_color, **_kwargs)[0],
                'ax': _ax,
                'data': data,
                'x': _x,
                'y': _y,
                't': t,
                'vline': _vline,
                'title': title,
                'title_prefix': title_prefix,
            }]

        _ts = pd.Series(data[t].unique()).sort_values()

    else:

        _ts = pd.Series()

        for _line in lines:

            _keys = list(_line.keys())

            # default: label = y
            if 'label' not in _keys:
                if 'y' in _keys:
                    _line['label'] = _line['y']
                elif y is not None:
                    _line['label'] = y

                # update keys
                _keys = list(_line.keys())

            # get kws
            _line_kws = {}
            _line_kw_keys = [_ for _ in _keys if _ not in ['ax', 'line', 'ts', 'data', 'x', 'y', 't']]
            _kw_keys = [_ for _ in list(kwargs.keys()) if _ not in _line_kw_keys]

            for _key in _line_kw_keys:
                _line_kws[_key] = _line[_key]
            for _kw_key in _kw_keys:
                _line_kws[_kw_key] = kwargs[_kw_key]

            if 'ax' not in _keys:
                _line['ax'] = _ax_list[0]
            if 'line' not in _keys:
                _line['line'] = _line['ax'].plot([], [], **_line_kws)[0],
            if is_list_like(_line['line']):
                _line['line'] = _line['line'][0]

            for _arg in list(_args.keys()):
                if _arg not in _keys:
                    _line[_arg] = _args[_arg]

            _line['ts'] = _line['data'][_line['t']].drop_duplicates().sort_values().reset_index(drop=True)

            _ts = _ts.append(_line['ts']).drop_duplicates().sort_values().reset_index(drop=True)

    # get max interval
    if max_interval is not None:
        if max_interval < _ts.shape[0]:
            _max_interval = max_interval
        else:
            _max_interval = _ts.shape[0]
    else:
        _max_interval = _ts.shape[0]

    # unchanging stuff goes here
    def init():

        for __ax in _ax_list:

            _xylim_set = False
            _x_min = None
            _x_max = None
            _y_min = None
            _y_max = None
            _legend = legend

            for __line in lines:

                # -- xy lims --

                if __ax == __line['ax']:

                    if not _xylim_set:

                        # init with limits of first line

                        _x_min = __line['data'][__line['x']].min()
                        _x_max = __line['data'][__line['x']].max()
                        _y_min = __line['data'][__line['y']].min()
                        _y_max = __line['data'][__line['y']].max()

                        _xylim_set = True

                    else:

                        # compare with x y lims of other lines

                        if __line['data'][__line['x']].min() < _x_min:
                            _x_min = __line['data'][__line['x']].min()
                        if __line['data'][__line['y']].min() < _y_min:
                            _y_min = __line['data'][__line['y']].min()
                        if __line['data'][__line['x']].max() > _x_max:
                            _x_max = __line['data'][__line['x']].max()
                        if __line['data'][__line['y']].max() > _y_max:
                            _y_max = __line['data'][__line['y']].max()

                    # -- legend --
                    if 'legend' in list(__line.keys()):
                        _legend = __line['legend']
                    if _legend:
                        if legend_out:
                            legend_outside(__ax, width=.995)
                        else:
                            __ax.legend(**legend_kws)

                    # -- vlines --
                    if 'vline' in __line.keys():
                        _vline_i = __line['vline']
                        if _vline_i is not None:
                            if not is_list_like(_vline_i):
                                _vline_i = [_vline_i]
                            for _vline_j in _vline_i:
                                __ax.axvline(_vline_j, color='k', linestyle=':')

            # -- lims --
            if xlim is not None:
                if xlim:
                    __ax.set_xlim(xlim)
            else:
                __ax.set_xlim([_x_min, _x_max])

            if ylim is not None:
                if ylim:
                    __ax.set_ylim(ylim)
            else:
                __ax.set_ylim([_y_min, _y_max])

            # -- grid --
            if grid:
                __ax.grid()

            # -- ax facecolor --
            if isinstance(ax_facecolor, str):
                __ax.set_facecolor(ax_facecolor)

        return ()

    def animate(_i):

        _t = _ts[_i]

        for _line_i in lines:

            _line_keys_i = list(_line_i.keys())

            _data = _line_i['data'].copy()
            _data = _data[_data[_line_i['t']] == _t]

            _line_i['line'].set_data(_data[_line_i['x']], _data[_line_i['y']])

            if 'ax' in _line_keys_i:
                _ax_i = _line_i['ax']
            else:
                _ax_i = plt.gca()

            # -- title --
            _title = title
            _title_prefix = title_prefix

            if 'title' in list(_line_i.keys()):
                _title = _line_i['title']
            if 'title_prefix' in list(_line_i.keys()):
                _title_prefix = _line_i['title_prefix']

            if t_format is not None:
                _t_str = pd.to_datetime(_t).strftime(t_format)
            else:
                _t_str = _t

            if _title:
                _ax_i.set_title('{}{}'.format(_title_prefix, _t_str))

            # -- facecolor --
            if isinstance(ax_facecolor, Mapping):

                for _key_i in list(ax_facecolor.keys()):

                    _ax_facecolor = ax_facecolor[_key_i]
                    if (_key_i is None) or (_key_i > _t):
                        _ax_i.set_facecolor(_ax_facecolor)

        return ()

    # - get correct ax for each line
    for _line in lines:
        if 'ax' in list(_line.keys()):
            _ax = _line['ax']
        else:
            _ax = plt.gca()

    # - create main FuncAnimation object
    _anim = FuncAnimation(fig, animate, init_func=init, frames=_max_interval, interval=time_per_frame, blit=True)
    # - close plots
    plt.close('all')

    # -- return
    # -handle return mode
    if mode == 'html':
        return HTML(_anim.to_html5_video())
    elif mode == 'jshtml':
        return HTML(_anim.to_jshtml())
    else:
        return _anim


@docstr
@export
def legend_outside(ax: plt.Axes = None, width: float = .85, loc: str = 'right',
                   legend_space: float = None, offset_x: float = 0,
                   offset_y: float = 0, loc_warn: bool = True, **kwargs):
    """
    draws a legend outside of the subplot
    
    :param ax: %(ax_in)s
    :param width: how far to shrink down the subplot if loc=='right'
    :param loc: one of ['right','bottom'], where to put the legend
    :param legend_space: how far below the subplot to put the legend if loc=='bottom'
    :param offset_x: x offset for the legend
    :param offset_y: y offset for the legend
    :param loc_warn: Whether to trigger a warning if legend loc is not recognized
    :param kwargs: other keyword arguments passed to pyplot.legend
    :return: None
    """
    # -- init
    # - defaults
    if legend_space is None:
        legend_space = rcParams['legend_outside.legend_space']
    if ax is None:
        ax = plt.gca()
    # - check if loc is legend_outside specific, if not treat as inside loc and call regular ax.legend
    if loc not in ['bottom', 'right']:
        if loc_warn:
            warnings.warn('legend_outside: legend loc not recognized, defaulting to plt.legend')
        ax.legend(loc=loc, **kwargs)
        return None

    # -- main
    # - get loc and bbox
    _loc = {'bottom': 'upper center', 'right': 'center left'}[loc]
    _bbox_to_anchor = {'bottom': (0.5 + offset_x, - .15 + offset_y), 'right': (1, 0.5)}[loc]
    # - loop axes
    for _ax in ax_as_list(ax):

        # -- shrink box
        _box = _ax.get_position()
        _pos = {
            'bottom': [_box.x0, _box.y0, _box.width, _box.height * (1 - legend_space)],
            # 'bottom':[_box.x0, _box.y0 + _box.height * legend_space,_box.width, _box.height * (1-legend_space)],
            'right': [_box.x0, _box.y0, _box.width * width, _box.height]
        }[loc]
        _ax.set_position(_pos)

        # -- legend
        logging.getLogger().setLevel(logging.CRITICAL)
        _, _labels = _ax.get_legend_handles_labels()
        if len(_labels) > 0:
            _ax.legend(loc=_loc, bbox_to_anchor=_bbox_to_anchor, **kwargs)
        logging.getLogger().setLevel(logging.DEBUG)


@docstr
@export
def set_ax_sym(ax: plt.Axes, x: bool = True, y: bool = True):
    """
    automatically sets the select axes to be symmetrical
    
    :param ax: %(ax_in)s
    :param x: whether to set x axis to be symmetrical
    :param y: whether to set y axis to be symmetrical
    :return: None
    """
    if x:
        _x_max = np.max(np.abs(np.array(ax.get_xlim())))
        # noinspection PyTypeChecker
        ax.set_xlim((-_x_max, _x_max))
    if y:
        _y_max = np.max(np.abs(np.array(ax.get_ylim())))
        # noinspection PyTypeChecker
        ax.set_ylim((-_y_max, _y_max))


@docstr
@export
def custom_legend(colors: Union[list, str], labels: Union[list, str], do_show=True) -> Union[list, None]:
    """
    uses patches to create a custom legend with the specified colors

    :param colors: list of matplotlib colors to use for the legend
    :param labels: list of labels to use for the legend
    :param do_show: whether to show the created legend
    :return: if do_show: None, else handles
    """
    _handles = []

    for _color, _label in zip(assert_list(colors), assert_list(labels)):
        _handles.append(patches.Patch(color=_color, label=_label))

    if do_show:
        plt.legend(handles=_handles)
    else:
        return _handles


def lcurveplot(train, test, labels=None, legend='upper right', ax=None):
    if labels is None:
        if 'name' in dir(train):
            _label_train = train.name
        else:
            _label_train = 'train'
        if 'name' in dir(test):
            _label_test = test.name
        else:
            _label_test = 'test'
    elif isinstance(labels, Mapping):
        _label_train = labels['train']
        _label_test = labels['test']
    elif is_list_like(labels):
        _label_train = labels[0]
        _label_test = labels[1]
    else:
        _label_train = labels
        _label_test = labels

    if ax is None:
        ax = plt.gca()

    ax.plot(train, color='xkcd:blue', label=_label_train)
    ax.plot(test, color='xkcd:red', label=_label_test)
    ax.plot(lfit(test), color='xkcd:red', ls='--', alpha=.75, label=_label_test + '_lfit')
    ax.axhline(np.min(test), color='xkcd:red', ls=':', alpha=.5)
    ax.axvline(np.argmin(test), color='xkcd:red', ls=':', alpha=.5)

    if legend:
        if isinstance(legend, str):
            _loc = legend
        else:
            _loc = None
        ax.legend(loc=_loc)

    return ax


def dic_to_lcurveplot(dic, width=16, height=9 / 2, **kwargs):
    if 'curves' not in dic.keys():
        warnings.warn('key curves not found, stopping')
        return None

    _targets = list(dic['curves'].keys())
    _nrows = len(_targets)

    _, ax = plt.subplots(nrows=_nrows, figsize=(width, height * _nrows))
    _ax_list = ax_as_list(ax)

    for _it, _target in enumerate(_targets):
        _ax = _ax_list[_it]
        lcurveplot(dic['curves'][_target]['train'], dic['curves'][_target]['test'],
                   labels=['{}_train'.format(_target), '{}_test'.format(_target)], ax=_ax, **kwargs)

    plt.show()


@docstr
@export
def stemplot(x, y, data=None, ax=None, color=rcParams['palette'][0], baseline=0, kwline=None, **kwargs):
    """
    modeled after pyplot.stemplot but more customizeable
    
    :param x: %(x)s
    :param y: %(y)s
    :param data: %(data)s
    :param ax: %(ax_in)s
    :param color: %(color)s
    :param baseline: where to draw the baseline for the stemplot
    :param kwline: other keyword arguments passed to pyplot.plot
    :param kwargs: other keyword arguments passed to pyplot.scatter
    :return: %(ax_out)s
    """
    if kwline is None:
        kwline = {}
    if data is None:
        if 'name' in dir(x):
            _x = x.name
        else:
            _x = 'x'
        if 'name' in dir(y):
            _y = y.name
        else:
            _y = 'x'
        _data = pd.DataFrame({_x: x, _y: y})
    else:
        _x = x
        _y = y
        _data = data.copy()

    if ax is None:
        ax = plt.gca()

    # baseline
    ax.axhline(baseline, color='k', ls='--', alpha=.5)

    # iterate over data so you can draw the lines
    for _it, _row in _data.iterrows():
        ax.plot([_row[_x], _row[_x]], [baseline, _row[_y]], color=color, label='__nolegend__', **kwline)

    # scatterplot for markers
    ax.scatter(x=_x, y=_y, data=_data, facecolor=color, **kwargs)

    return ax


def from_to_plot(data: pd.DataFrame, x_from='x_from', x_to='x_to', y_from=0, y_to=1, palette=None, label=None,
                 legend=True, legend_loc=None, ax=None, **kwargs):
    # defaults
    if ax is None:
        ax = plt.gca()
    if palette is None:
        palette = rcParams['palette']

    _labels = []

    for _, _row in data.itertuples():

        _label = '__nolabel__'

        _name = None

        if label is not None:

            _name = _row[label]

            if _name not in _labels:
                _label = _name + ''
                _labels.append(_label)

        if isinstance(palette, Mapping):
            _color = palette[_name]
        elif is_list_like(palette):
            _color = palette[_labels.index(_name) % len(palette)]
        else:
            _color = palette

        ax.fill_betweenx([y_from, y_to], _row[x_from], _row[x_to], label=_label, color=_color, **kwargs)

    if legend and label:
        ax.legend(loc=legend_loc)

    return ax


def vlineplot(data, palette=None, label=None, legend=True, legend_loc=None, ax=None, **kwargs):
    # defaults
    if ax is None:
        ax = plt.gca()
    if palette is None:
        palette = rcParams['palette']

    _labels = []

    _name = None

    for _, _row in data.iterrows():

        _label = '__nolabel__'

        if label is not None:

            _name = _row[label]

            if _name not in _labels:
                _label = _name + ''
                _labels.append(_label)

        if isinstance(palette, Mapping):
            _color = palette[_name]
        elif is_list_like(palette):
            _color = palette[_labels.index(_name) % len(palette)]
        else:
            _color = palette

        ax.axvline(_row['x'], label=_label, color=_color, **kwargs)

    if legend and label:
        ax.legend(loc=legend_loc)

    return ax


def show_ax_ticklabels(ax, x=None, y=None):
    _ax_list = ax_as_list(ax)

    for _ax in _ax_list:

        if x is not None:
            plt.setp(_ax.get_xticklabels(), visible=x)
        if y is not None:
            plt.setp(_ax.get_yticklabels(), visible=y)


@docstr
@export
def get_twin(ax: plt.Axes) -> Union[plt.Axes, None]:
    """
    get the twin axis from an Axes object
    
    :param ax: %(ax_in)s
    :return: the twin axis if it exists, else None
    """
    for _other_ax in ax.figure.axes:
        if _other_ax is ax:
            continue
        if _other_ax.bbox.bounds == ax.bbox.bounds:
            return _other_ax
    return None


@docstr
@export
def get_axlim(ax: plt.Axes, xy: Union[str, None] = None) -> Union[tuple, Mapping]:
    """
    Wrapper function to get x limits, y limits or both with one function call
    
    :param ax: %(ax_in)s
    :param xy: one of ['x', 'y', 'xy', None]
    :return: if xy is 'xy' or None returns a dictionary else returns x or y lims as tuple
    """
    if xy == 'x':
        return ax.get_xlim()
    elif xy == 'y':
        return ax.get_ylim()
    else:
        return {'x': ax.get_xlim(), 'y': ax.get_ylim()}


@docstr
@export
def set_axlim(ax: plt.Axes, lim: Union[Sequence, Mapping], xy: Union[str, None] = None):
    """
    Wrapper function to set both x and y limits with one call
    
    :param ax: %(ax_in)s
    :param lim: axes limits as tuple or Mapping
    :param xy: one of ['x', 'y', 'xy', None]
    :return: None
    """
    if xy == 'x':
        # noinspection PyTypeChecker
        ax.set_xlim(lim)
    elif xy == 'y':
        # noinspection PyTypeChecker
        ax.set_ylim(lim)
    else:
        if isinstance(lim, Mapping):
            ax.set_xlim(lim['x'])
            ax.set_xlim(lim['y'])
        else:
            raise ValueError('Specify xy parameter or pass a dictionary')


@docstr
@export
def share_xy(ax: plt.Axes, x: bool = True, y: bool = True, mode: str = 'all', adj_twin_ax: bool = True):
    """
    set the subplots on the Axes to share x and/or y limits WITHOUT sharing x and y legends.
    If you want that please use pyplot.subplots(share_x=True,share_y=True) when creating the plots.
    
    :param ax: %(ax_in)s
    :param x: whether to share x limits [optional]
    :param y: whether to share y limits [optional]
    :param mode: one of ['all', 'row', 'col'], if all shares across all subplots, else just across rows / columns
    :param adj_twin_ax: whether to also adjust twin axes
    :return: None
    """
    _xys = []
    if x:
        _xys.append('x')
    if y:
        _xys.append('y')

    if isinstance(ax, np.ndarray):
        _dims = len(ax.shape)
    else:
        _dims = 0

    # slice for mode row / col (only applicable if shape==2)
    _ax_lists = []

    if (_dims <= 1) or (mode == 'all'):
        _ax_lists += [ax_as_list(ax)]
    elif mode == 'row':
        for _row in range(ax.shape[0]):
            _ax_lists += [ax_as_list(ax[_row, :])]
    elif mode == 'col':
        for _col in range(ax.shape[1]):
            _ax_lists += [ax_as_list(ax[:, _col])]

    # we can have different subsets (by row or col) that share x / y min
    for _ax_list in _ax_lists:

        # init as None
        _xy_min = {'x': None, 'y': None}
        _xy_max = {'x': None, 'y': None}

        # get min max
        for _ax in _ax_list:

            _lims = get_axlim(_ax)

            for _xy in _xys:

                _xy_min_i = _lims[_xy][0]
                _xy_max_i = _lims[_xy][1]

                if _xy_min[_xy] is None:
                    _xy_min[_xy] = _xy_min_i
                elif _xy_min[_xy] > _xy_min_i:
                    _xy_min[_xy] = _xy_min_i

                if _xy_max[_xy] is None:
                    _xy_max[_xy] = _xy_max_i
                elif _xy_max[_xy] < _xy_max_i:
                    _xy_max[_xy] = _xy_max_i

        # set min max
        for _ax in _ax_list:

            if adj_twin_ax:
                _ax2 = get_twin(_ax)
            else:
                _ax2 = False

            # collect xy funcs
            for _xy in _xys:

                # save old lim
                _old_lim = list(get_axlim(_ax, xy=_xy))
                # set new lim
                _new_lim = [_xy_min[_xy], _xy_max[_xy]]
                set_axlim(_ax, lim=_new_lim, xy=_xy)

                # adjust twin axis
                if _ax2:
                    _old_lim_2 = list(get_axlim(_ax2, xy=_xy))
                    _new_lim_2 = [0 if _old == 0 else _new / _old * _old2 for _new, _old, _old2 in
                                  zip(_new_lim, _old_lim, _old_lim_2)]
                    set_axlim(_ax2, lim=_new_lim_2, xy=_xy)


@docstr
@export
def share_legend(ax: plt.Axes, keep_i: int = None):
    """
    removes all legends except for i from an Axes object

    :param ax: %(ax_in)s
    :param keep_i: index of the plot whose legend you want to keep
    :return: None
    """

    _ax_list = ax_as_list(ax)

    if keep_i is None:
        keep_i = len(_ax_list) // 2

    for _it, _ax in enumerate(_ax_list):

        _it += 1
        _legend = _ax.get_legend()
        if _it != keep_i and (_legend is not None):
            _legend.remove()


def replace_xticklabels(ax, mapping):
    _new_labels = []

    for _it, _label in enumerate(list(ax.get_xticklabels())):

        _text = _label.get_text()

        if isinstance(mapping, Mapping):
            if _text in mapping.keys():
                _new_label = mapping[_text]
            else:
                _new_label = _text
        else:
            _new_label = mapping[_it]

        _new_labels.append(_new_label)

    ax.set_xticklabels(_new_labels)


def replace_yticklabels(ax, mapping):
    _new_labels = []

    for _it, _label in enumerate(list(ax.get_yticklabels())):

        _text = _label.get_text()

        if isinstance(mapping, Mapping):
            if _text in mapping.keys():
                _new_label = mapping[_text]
            else:
                _new_label = _text
        else:
            _new_label = mapping[_it]

        _new_labels.append(_new_label)

    ax.set_yticklabels(_new_labels)


def kdeplot(x, data=None, *args, hue=None, hue_order=None, bins=40, adj_x_range=False, baseline=0, highlight_peaks=True,
            show_kde=True, hist=True, show_area=False, area_center='mean', ha='center', va='center',
            legend_loc='upper right', palette=None, text_offset=15, nr_format=',.2f',
            kwline=None, perc=False, facecolor=None, sigma_color='xkcd:blue',
            sigma_2_color='xkcd:cyan', kde_color='black', edgecolor='black', alpha=.5, ax=None, ax2=None, kwhist=None,
            **kwargs):
    # -- init
    if palette is None:
        palette = rcParams['palette']
    if kwline is None:
        kwline = {}
    if kwhist is None:
        kwhist = {}
    if data is not None:
        _df = data.copy()
        del data
        _x_name = x
    else:
        if 'name' in dir(x):
            _x_name = x.name
        else:
            _x_name = 'x'

        _df = pd.DataFrame({_x_name: x})

    _df = _df.dropna(subset=[_x_name])

    if hue is None:
        hue = '_dummy'
        _df[hue] = 1
    if hue_order is None:
        hue_order = sorted(_df[hue].unique())

    _x = _df[_x_name]

    if facecolor is None:
        if show_area:
            facecolor = 'None'
        else:
            facecolor = 'xkcd:cyan'

    if show_kde and show_area:
        _label_hist = '__nolabel__'
    else:
        _label_hist = _x_name

    # default
    if adj_x_range and isinstance(adj_x_range, bool):
        adj_x_range = 2

    # -- get kde
    _it = -1
    _twinx = False

    for _hue in hue_order:

        _it += 1
        _df_hue = _df.query('{}==@_hue'.format(hue))

        _df_kde, _df_kde_ex = kde(x=x, df=_df_hue, *args, **kwargs)

        if isinstance(palette, Mapping):
            _color = palette[_hue]
        elif is_list_like(palette):
            _color = palette[_it % len(palette)]
        else:
            _color = palette

        if hue == '_dummy':
            _kde_color = kde_color
            _edgecolor = edgecolor
            _facecolor = facecolor
        else:
            _kde_color = _color
            _edgecolor = _color
            _facecolor = 'None'
            _df_kde['value'] = _df_kde['value'] / _df_kde['value'].max()
            _df_kde_ex['value'] = _df_kde_ex['value'] / _df_kde['value'].max()

        if adj_x_range:
            _x_min = _df_kde_ex['range_min'].min()
            _x_max = _df_kde_ex['range_max'].max()

            _x_step = (_x_max - _x_min) / bins
            _x_range_min = _x_min - _x_step * adj_x_range * bins
            _x_range_max = _x_max + _x_step * adj_x_range * bins

            _df_hue = _df_hue.query('{}>=@_x_range_min & {}<=@_x_range_max'.format(_x_name, _x_name))
            _df_kde = _df_kde.query('{}>=@_x_range_min & {}<=@_x_range_max'.format(_x_name, _x_name))

        # -- plot

        if ax is None:
            ax = plt.gca()

        # hist
        if hist:
            ax.hist(_df_hue[_x_name], bins, density=perc, facecolor=_facecolor, edgecolor=_edgecolor,
                    label=_label_hist, **kwhist)
            _twinx = True
        else:
            _twinx = False

        if _twinx and (ax2 is None):
            ax2 = ax.twinx()
        else:
            ax2 = ax

        _kde_label = '{} ; '.format(_x_name) + r'${:,.2f}\pm{:,.2f}$'.format(_df[_x_name].mean(), _df[_x_name].std())

        # kde
        ax2.plot(_df_kde[_x_name], _df_kde['value'], ls='--', label=_kde_label, color=_kde_color, **kwargs)

        _ylim = list(ax2.get_ylim())
        _ylim[0] = 0
        _ylim[1] = _ylim[1] * (100 + text_offset) / 100.
        ax2.set_ylim(_ylim)

        # area
        if show_area:

            # get max
            if area_center == 'max':
                _area_center = _df_kde[_df_kde['value'] == _df_kde['value'].max()].index[0]
            else:
                if area_center == 'mean':
                    _ref = _df_hue[_x_name].mean()
                else:
                    _ref = area_center

                _df_area = _df_kde.copy()
                _df_area['diff'] = (_df_area[_x_name] - area_center).abs()
                _df_area = _df_area.sort_values(by='diff', ascending=True)
                _area_center = _df_area.index[0]

            _sigma = None
            _2_sigma = None

            for _it in range(1, _df_kde.shape[0]):

                _perc_data = \
                    _df_kde[np.max([0, _area_center - _it]):np.min([_df_kde.shape[0], _area_center + _it + 1])][
                        'value'].sum() / _df_kde['value'].sum()

                if (_perc_data >= .6826) and (_sigma is None):
                    _sigma = _it + 0
                if (_perc_data >= .9544) and (_2_sigma is None):
                    _2_sigma = _it + 0
                    break
                if _it == _df_kde.shape[0] - 1:
                    _2_sigma = _it + 0

            _df_sigma = _df_kde.loc[
                        np.max([0, _area_center - _sigma]):np.min([_df_kde.shape[0], _area_center + _sigma])]
            _df_2_sigma_left = _df_kde.loc[
                               np.max([0, _area_center - _2_sigma]):np.min([_df_kde.shape[0], _area_center - _sigma])]
            _df_2_sigma_right = _df_kde.loc[
                                np.max([0, _area_center + _sigma]):np.min([_df_kde.shape[0], _area_center + _2_sigma])]

            _2_sigma_min = _df_2_sigma_left[_x_name].min()
            _2_sigma_max = _df_2_sigma_right[_x_name].max()
            if np.isnan(_2_sigma_min):
                _2_sigma_min = _df[_x_name].min()
            if np.isnan(_2_sigma_max):
                _2_sigma_max = _df[_x_name].max()

            _sigma_range = ': {:,.2f} to {:,.2f}'.format(_df_sigma[_x_name].min(), _df_sigma[_x_name].max())
            _2_sigma_range = ': {:,.2f} to {:,.2f}'.format(_2_sigma_min, _2_sigma_max)

            ax2.fill_between(_x_name, 'value', data=_df_sigma, color=sigma_color,
                             label=r'$1\sigma(68\%)$' + _sigma_range, alpha=alpha)
            ax2.fill_between(_x_name, 'value', data=_df_2_sigma_left, color=sigma_2_color,
                             label=r'$2\sigma(95\%)$' + _2_sigma_range, alpha=alpha)
            ax2.fill_between(_x_name, 'value', data=_df_2_sigma_right, color=sigma_2_color, label='__nolegend__',
                             alpha=alpha)
            ax2.legend(loc=legend_loc)

        # iterate over data so you can draw the lines
        if highlight_peaks:

            for _it, _row in _df_kde_ex.iterrows():

                _mu = _row[_x_name]
                _value_std = np.min([_row['value_min'], _row['value_max']])

                # stem (max)
                ax2.plot([_mu, _mu], [baseline, _row['value']], color=kde_color, label='__nolegend__', ls=':', **kwline)
                # std
                if highlight_peaks != 'max':
                    ax2.plot([_row['range_min'], _row['range_max']], [_value_std, _value_std],
                             color=kde_color, label='__nolegend__', ls=':', **kwline)

                # scatterplot for markers
                ax2.scatter(x=_mu, y=_row['value'], facecolor=kde_color, **kwargs)

                _mean_str = format(_mu, nr_format)
                _std_str = format(_row['range'] / 2., nr_format)

                _annotate = r'${}$'.format(_mean_str)
                if highlight_peaks != 'max':
                    _annotate += '\n' + r'$\pm{}$'.format(_std_str)

                ax2.annotate(_annotate, (_mu, _row['value']), ha=ha, va=va, xytext=(0, text_offset),
                             textcoords='offset points')

    if _twinx:
        ax2.legend(loc=legend_loc)
        ax2.set_axis_off()
    else:
        ax.legend(loc=legend_loc)

    return ax


def draw_ellipse(ax, *args, **kwargs):
    _e = patches.Ellipse(*args, **kwargs)
    ax.add_artist(_e)


@docstr
@export
def barplot_err(x: str, y: str, xerr: str = None, yerr: str = None, data: pd.DataFrame = None, **kwargs) -> plt.Axes:
    """
    extension on `seaborn barplot <https://seaborn.pydata.org/generated/seaborn.barplot.html>`_ that allows
    for plotting errorbars with preprocessed data. The idea is based on this `StackOverflow question
    <https://datascience.stackexchange.com/questions/31736/unable-to-generate-error-bars-with-seaborn/64128>`_

    :param x: %(x_novec)s
    :param y: %(y_novec)s
    :param xerr: variable to use as x error bars [optional]
    :param yerr: variable to use as y error bars [optional]
    :param data: %(data_novec)s
    :param kwargs: other keyword arguments passed to `seaborn barplot
        <https://seaborn.pydata.org/generated/seaborn.barplot.html>`_
    :return: %(ax_out)s
    """
    _data = []
    for _it in data.index:

        _data_i = pd.concat([data.loc[_it:_it]] * 3, ignore_index=True, sort=False)
        _row = data.loc[_it]

        if xerr is not None:
            _data_i[x] = [_row[x] - _row[xerr], _row[x], _row[x] + _row[xerr]]
        if yerr is not None:
            _data_i[y] = [_row[y] - _row[yerr], _row[y], _row[y] + _row[yerr]]
        _data.append(_data_i)

    _data = pd.concat(_data, ignore_index=True, sort=False)

    _ax = sns.barplot(x=x, y=y, data=_data, ci='sd', **kwargs)

    return _ax


def q_barplot(pd_series, ax=None, sort=False, percentage=False, **kwargs):
    _name = pd_series.name

    if ax is None:
        ax = plt.gca()

    _df_plot = pd_series.value_counts().reset_index()

    if sort:
        _df_plot = _df_plot.sort_values(['index'])

    if percentage:

        _y_name = _name + '_perc'
        _df_plot[_y_name] = _df_plot[_name] / _df_plot[_name].sum() * 100
        _df_plot[_y_name] = _df_plot[_y_name].round(2)

    else:

        _y_name = _name

    sns.barplot(data=_df_plot, x='index', y=_y_name, ax=ax, **kwargs)

    return ax


def histplot(x=None, data=None, hue=None, hue_order=None, ax=None, bins=30, use_q_xlim=False,
             legend_kws=None, **kwargs):
    # long or short format
    if legend_kws is None:
        legend_kws = {}
    if data is not None:
        # avoid inplace operations
        _df_plot = data.copy()
        del data
        _x = x
    else:
        # create dummy df
        _df_plot = pd.DataFrame.from_dict({'x': x})
        _x = 'x'

    _xs = _df_plot[_x]

    # if applicable: filter data
    if use_q_xlim:
        _x_lim = q_plim(_xs)
        _df_plot = _df_plot[(_df_plot[_x] >= _x_lim[0]) & (_df_plot[_x] <= _x_lim[1])]
        _xs = _df_plot[_x]

    # create bins
    if not isinstance(bins, list):
        bins = np.linspace(_xs.min(), _xs.max(), bins)

    # if an axis has not been passed initialize one
    if ax is None:
        ax = plt.gca()

    # if a hue has been passed loop them
    if hue is not None:

        # if no hue order has been passed use default sorting
        if hue_order is None:
            hue_order = sorted(_df_plot[hue].unique())

        for _hue in hue_order:
            _xs = _df_plot[_df_plot[hue] == _hue][_x]

            ax.hist(_xs, label=_hue, alpha=.5, bins=bins, **kwargs)

        ax.legend(**legend_kws)

    else:

        ax.hist(_xs, bins=bins, **kwargs)

    return ax


@docstr
@export
def countplot(x: Union[Sequence, str] = None, data: pd.DataFrame = None, hue: str = None, ax: plt.Axes = None,
              order: Union[Sequence, str] = None, hue_order: Union[Sequence, str] = None, normalize_x: bool = False,
              normalize_hue: bool = False, palette: Union[Mapping, Sequence, str] = None,
              x_tick_rotation: int = None, count_twinx: bool = False, hide_legend: bool = False, annotate: bool = True,
              annotate_format: str = rcParams['int_format'], legend_loc: str = 'upper right',
              barplot_kws: Mapping = None, count_twinx_kws: Mapping = None, **kwargs):
    """
    Based on seaborn barplot but with a few more options, uses :func:`~hhpy.ds.df_count`
    
    :param x: %(x)s
    :param data: %(data)s
    :param hue: %(hue)s
    :param ax: %(ax_in)s
    :param order: %(order)s
    :param hue_order: %(order)s
    :param normalize_x: Whether to normalize x, causes the sum of each x group to be 100 percent [optional]
    :param normalize_hue: Whether to normalize hue, causes the sum of each hue group to be 100 percent [optional]
    :param palette: %(palette)s
    :param x_tick_rotation: %(x_tick_rotation)s
    :param count_twinx: Whether to plot the count values on the second axis (if using normalize) [optional]
    :param hide_legend: Whether to hide the legend [optional]
    :param annotate: Whether to use annotate_barplot [optional]
    :param annotate_format: %(number_format)s
    :param legend_loc: %(legend_loc)s
    :param barplot_kws: Additional keyword arguments passed to seaborn.barplot [optional]
    :param count_twinx_kws: Additional keyword arguments passed to pyplot.plot [optional]
    :param kwargs: Additional keyword arguments passed to :func:`~hhpy.ds.df_count` [optional]
    :return: %(ax_out)s
    """

    # -- init
    # defaults
    if barplot_kws is None:
        barplot_kws = {}
    if count_twinx_kws is None:
        count_twinx_kws = {}
    # long or short format
    if data is not None:
        # avoid inplace operations
        data = data.copy()
        # if x is not specified count each row
        if x is None:
            x = '_dummy'
            data = data.assign(_dummy=1)
    else:
        if isinstance(x, pd.DataFrame):
            # case: only a DataFrame is passed as first argument (count rows)
            data = x.copy().assign(_dummy=1)
        else:
            # assume passed object is a Sequence and create dummy df
            data = pd.DataFrame({'_dummy': x})
        x = '_dummy'

    _count_x = 'count_{}'.format(x)
    _count_hue = 'count_{}'.format(hue)

    # if an axis has not been passed initialize one
    if ax is None:
        ax = plt.gca()

    if normalize_x:
        _y = 'perc_{}'.format(x)
    elif normalize_hue:
        _y = 'perc_{}'.format(hue)
    else:
        _y = 'count'

    _df_count = df_count(x=x, df=data, hue=hue, **kwargs)

    if order is None or order == 'count':
        _order = _df_count[[x, _count_x]].drop_duplicates().sort_values(by=[_count_x], ascending=False)[x].tolist()
    elif order == 'sorted':
        _order = _df_count[x].drop_duplicates().sort_values().tolist()
    else:
        _order = order

    if hue is not None:
        _hues = _get_ordered_levels(data=data, level=hue, order=hue_order, x=x)

    if palette is None:
        palette = rcParams['palette'] * 5

    sns.barplot(data=_df_count, x=x, y=_y, hue=hue, order=_order, hue_order=hue_order, palette=palette, ax=ax,
                **barplot_kws)
    ax.set_xlabel('')

    # cleanup for x=None
    if x is None:
        ax.get_xaxis().set_visible(False)
        if normalize_x:
            ax.set_ylabel('perc')

    if hue is None and normalize_hue:
        ax.set_ylabel('perc')

    if annotate:
        # add annotation
        annotate_barplot(ax, nr_format=annotate_format)
        # enlarge ylims
        _ylim = list(ax.get_ylim())
        _ylim[1] = _ylim[1] * 1.1
        # noinspection PyTypeChecker
        ax.set_ylim(_ylim)

    # legend
    if hide_legend:
        ax.get_legend().remove()
    elif hue is not None:
        legend_outside(ax, loc=legend_loc, loc_warn=False)

    # tick rotation
    if x_tick_rotation is not None:
        ax.xaxis.set_tick_params(rotation=x_tick_rotation)

    # total count on secaxis
    if count_twinx:
        _ax = ax.twinx()
        _count_twinx_kws_keys = list(count_twinx_kws.keys())
        if 'marker' not in _count_twinx_kws_keys:
            count_twinx_kws['marker'] = '_'
        if 'color' not in _count_twinx_kws_keys:
            count_twinx_kws['color'] = 'k'
        if 'alpha' not in _count_twinx_kws_keys:
            count_twinx_kws['alpha'] = .5
        _ax.scatter(x, _count_x, data=_df_count[[x, _count_x]].drop_duplicates(), **count_twinx_kws)
        _ax.set_ylabel('count')

    return ax


@docstr
@export
def quantile_plot(x: Union[Sequence, str], data: pd.DataFrame = None, qs: Union[Sequence, float] = None, x2: str = None,
                  hue: str = None, hue_order: Union[Sequence, str] = None, to_abs: bool = False, ax: plt.Axes = None,
                  **kwargs) -> plt.Axes:
    """
    plots the specified quantiles of a Series using seaborn.barplot
    
    :param x: %(x)s
    :param data: %(data)s
    :param qs: Quantile levels [optional]
    :param x2: if specified: subtracts x2 from x before calculating quantiles [optional]
    :param hue: %(hue)s
    :param hue_order: %(order)s 
    :param to_abs: %(to_abs)s
    :param ax: %(ax_in)s
    :param kwargs: other keyword arguments passed to seaborn.barplot
    :return: %(ax_out)s
    """
    # long or short format
    if qs is None:
        qs = [.1, .25, .5, .75, .9]
    if data is not None:
        # avoid inplace operations
        _df = data.copy()
        if x2 is None:
            _x = x
        else:
            _x = '{} - {}'.format(x, x2)
            _df[_x] = _df[x] - _df[x2]

    else:
        # create dummy df
        if x2 is None:
            _df = pd.DataFrame({'x': x})
            _x = 'x'
        else:
            _df = pd.DataFrame({'x': x, 'x2': x2}).eval('x_delta=x2-x')
            _x = 'x_delta'

    if ax is None:
        ax = plt.gca()

    _label = _x

    if to_abs:
        _df[_x] = _df[_x].abs()
        _label = '|{}|'.format(_x)

    if hue is None:
        _df_q = _df[_x].quantile(qs).reset_index()
    else:
        _hues = _get_ordered_levels(data=_df, level=hue, order=hue_order, x=_x)

        _df_q = []

        for _hue in _hues:
            _df_i = _df[_df[hue] == _hue][_x].quantile(qs).reset_index()
            _df_i[hue] = _hue
            _df_q.append(_df_i)

        _df_q = pd.concat(_df_q, ignore_index=True, sort=False)

    sns.barplot(x='index', y=_x, data=_df_q, hue=hue, ax=ax, **kwargs)

    ax.set_xticklabels(['q{}'.format(int(_ * 100)) for _ in qs])
    ax.set_xlabel('')
    ax.set_ylabel(_label)

    return ax


@docstr
@export
def plotly_aggplot(data: pd.DataFrame, x: Scalar, y: Scalar, hue: Scalar = None, groupby: SequenceOrScalar = None,
                   sep: str = ';', agg: str = 'sum', hue_order: Union[list, str] = None, x_min: Scalar = None,
                   x_max: Scalar = None, y_min: Scalar = None, y_max: Scalar = None, mode: str = 'lines+markers',
                   title: str = None, xaxis_title: str = None, yaxis_title: str = None, label_maxchar: int = 15,
                   direction: str = 'up', showactive: bool = True, dropdown_x: float = 0, dropdown_y: float = -.1,
                   fig: go_figure = None, do_print: bool = True, kws_dropdown: Mapping = None, kws_fig: Mapping = None,
                   **kwargs) -> go_figure:
    """
    create a (grouped) plotly aggplot that let's you select the groupby categories

    :param data: %(data)s
    :param x: %(x_novec)s
    :param y: %(y_novec)s
    :param hue: %(hue)s
    :param groupby: Column name(s) to split the plot by [optional]
    :param sep: Separator used for groupby columns [optional]
    :param agg: Aggregate function to use [optional]
    :param hue_order: %(order)s
    :param x_min: %(x_min)s
    :param x_max: %(x_max)s
    :param y_min: %(y_min)s
    :param y_max: %(y_max)s
    :param mode: plotly mode [optional]
    :param title: %(title_plotly)s
    :param xaxis_title: %(xaxis_title)s
    :param yaxis_title: %(yaxis_title)s
    :param label_maxchar: Maximum allowed number of characters of the labels [optional]
    :param direction: One of ['up', 'down'] , direction of the dropdown [optional]
    :param showactive: Whether to show the active selection in the dropdown [optional]
    :param dropdown_x: x position of the first dropdown [optional]
    :param dropdown_y: y position of the first dropdown [optional]
    :param fig: %(fig_plotly)s
    :param do_print: %(do_print)s
    :param kws_dropdown: Other keyword arguments passed to the dropdown updatemenu [optional]
    :param kws_fig: other keyword arguments passed to plotly.graph_objects.Figure [optional]
    :param kwargs: other keyword arguments passed to plotly.graph_objects.scatter [optional]
    :return: plotly Figure with the plot on it
    """
    # -- assert
    if (y_min is not None and y_max is None) or (y_min is None and y_max is not None):
        raise ValueError('If you supply y_min or y_max you must also supply the other')

    # -- functions
    def _get_xy(fltr: tuple = None, hue_i: Scalar = None) -> tuple:
        _df = data.copy()
        if hue != '__dummy__':
            _df = _df[_df[hue] == hue_i]
        if fltr is not None:
            for __it, _value in enumerate(fltr):
                _key = groupby[__it]
                if _value != '<ALL>':
                    _df = _df[_df[_key] == _value]
        _df_agg = _df.groupby(x).agg({y: agg}).reset_index()
        return _df_agg[x], _df_agg[y]

    # -- init
    # - no inplace
    data = pd.DataFrame(data).copy()
    # - defaults
    if kws_dropdown is None:
        kws_dropdown = {}
    if kws_fig is None:
        kws_fig = {}
    if title is None:
        title = f"{agg} of '{y}' over '{x}'"
        if groupby is not None:
            title += f", filtered by '{groupby}'"
        if groupby is not None:
            title += f", split by '{hue}'"
    if xaxis_title is None:
        xaxis_title = x
    elif xaxis_title in [False, 'None']:
        xaxis_title = None
    if yaxis_title is None:
        yaxis_title = y
    elif yaxis_title in [False, 'None']:
        yaxis_title = None
    if hue is None:
        hue = '__dummy__'
        data[hue] = 1
        _hues = [1]
    else:
        _hues = _get_ordered_levels(data, hue, hue_order)
    if fig is None:
        fig = go.Figure(**kws_fig)
    # - force_list
    groupby = assert_list(groupby)
    # - x_min / x_max
    if x_min is not None:
        data = data[data[x] >= x_min]
    if x_max is not None:
        data = data[data[x] <= x_max]

    # -- main
    # - scatter
    for _hue in _hues:
        _x, _y = _get_xy(hue_i=_hue)
        fig.add_trace(go.Scatter(x=_x, y=_y, mode=mode, name=_hue, **kwargs))
    # - concat groupbys
    _groupby_dict = {}
    for _groupby in groupby:
        _groupby_dict[_groupby] = ['<ALL>'] + data[_groupby].drop_duplicates().sort_values().tolist()
    _groupby_values = list(itertools.product(*list(_groupby_dict.values())))
    _len_groupby_values = len(_groupby_values)
    # - updatemenus
    _updatemenus = []
    _buttons = []
    for _it_group, _category in enumerate(_groupby_values):
        # show progressbar
        if do_print:
            progressbar(_it_group, _len_groupby_values)
        # get x, y by hue
        _xs = []
        _ys = []
        for _hue in _hues:
            _x, _y = _get_xy(fltr=_category, hue_i=_hue)
            _xs.append(_x)
            _ys.append(_y)
        # get label
        _label = ''
        for _it_cat, _category_i in enumerate(assert_list(_category)):
            if _it_cat > 0:
                _label += sep
            _label_i = str(_category_i)
            if len(_label_i) > label_maxchar:
                _label_i = _label_i[:label_maxchar] + '...'
            _label += _label_i
        # create button
        _buttons.append({
            'method': 'restyle',
            'label': _label,
            'args': [{'x': _xs, 'y': _ys}]
        })
        # print(_buttons)
        _updatemenus.append({
            'buttons': _buttons,
            'direction': direction,
            'showactive': showactive,
            'x': dropdown_x,
            'y': dropdown_y,
            **kws_dropdown
        })

    # - fig
    # noinspection PyUnboundLocalVariable
    fig.update_layout(updatemenus=_updatemenus)
    # # - annotation (not properly aligned, therefore dropped for now)
    # _annotation = sep.join([str(_) for _ in force_list(groupby)])
    # _fig.update_layout(annotations=[
    #     go.layout.Annotation(text=_annotation, showarrow=False, x=dropdown_x, y=dropdown_y+.1, xref="paper",
    #                          yref="paper", align="left")
    # ])
    # - title / axis titles
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    # - y_min / y_max
    if y_min is not None:
        fig.update_yaxes(range=[y_min, y_max])
    # - final progressbar
    if do_print:
        progressbar()

    # -- return
    return fig


def cat_to_color(s: pd.Series, palette: SequenceOrScalar = None, out_type: str = None) -> pd.Series:
    """
    Encodes a categorical column as colors of a specified palette

    :param s: pandas Series
    :param palette: %(palette)s
    :param out_type: Color output type, one of %(cat_to_color__out_type); defaults to None (no conversion) [optional]
    :return: pandas Series of color names
    """

    # -- functions
    def _to_color(color_index: int):
        _color = palette[color_index % len(palette)]
        if out_type == 'hex':
            _color = mpl_colors.to_hex(_color)
        elif out_type == 'rgb':
            _color = mpl_colors.to_rgb(_color)
        elif out_type == 'rgba':
            _color = mpl_colors.to_rgba(_color)
        elif out_type == 'rgba_array':
            _color = mpl_colors.to_rgba_array(_color)
        return _color

    # -- assert
    # - no inplace
    s = pd.Series(s).copy()
    # - out_type
    if out_type not in validations['cat_to_color__out_type']:
        raise ValueError(f"out_type must be one of {validations['cat_to_color__out_type']}")

    # -- init
    # - defaults
    if palette is None:
        palette = rcParams['palette']
    palette = assert_list(palette)

    s = s.astype('category')
    if len(s.cat.categories) > len(palette):
        warnings.warn('Not enough colors in palette, colors will be reused')

    return s.cat.codes.apply(_to_color).astype('category')
