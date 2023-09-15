from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from IPython.core.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from .plotting_main import ax_as_list, docstr, legend_outside, rc_params
from ..main import is_list_like, list_intersection


@docstr
def animplot(
        data: pd.DataFrame = None,
        x: str = 'x',
        y: str = 'y',
        t: str = 't',
        lines: Mapping = None,
        max_interval: int = None,
        time_per_frame: int = 200,
        mode: str = None,
        title: bool = True,
        title_prefix: str = '',
        t_format: str = None,
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        color: str = None,
        label: str = None,
        legend: bool = False,
        legend_out: bool = False,
        legend_kws: Mapping = None,
        xlim: tuple = None,
        ylim: tuple = None,
        ax_facecolor: Union[str, Mapping] = None,
        grid: bool = False,
        vline: Union[Sequence, float] = None,
        hue: str = None,
        hue_order: Sequence = None,
        **kwargs
) -> Union[HTML, FuncAnimation]:
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
    :param hue: %(hue)s
    :param hue_order: %(order)s
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
        mode = rc_params['animplot.mode']
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

    if hue:
        if hue_order is None:
            hue_order = sorted(data[hue].unique())
    else:
        hue_order = [None]

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

            for _hue in hue_order:
                if hue:
                    _label = _hue
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
            _line_kw_keys = [_ for _ in _keys if _ not in [
                'ax', 'line', 'ts', 'data', 'x', 'y', 't']]
            _kw_keys = [_ for _ in list(
                kwargs.keys()) if _ not in _line_kw_keys]

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

            _line['ts'] = _line['data'][_line['t']].drop_duplicates(
            ).sort_values().reset_index(drop=True)

            _ts = _ts.append(_line['ts']).drop_duplicates(
            ).sort_values().reset_index(drop=True)

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
                                __ax.axvline(
                                    _vline_j, color='k', linestyle=':')

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

        for __it, _line_i in enumerate(lines):

            _line_keys_i = list(_line_i.keys())

            if 'ax' in _line_keys_i:
                _ax_i = _line_i['ax']
            else:
                _ax_i = plt.gca()

            _data = _line_i['data'].copy()
            _data = _data[_data[_line_i['t']] == _t]

            if hue:
                __hue = hue_order[__it]
                _data = _data[_data[hue] == __hue]
                _line_i['line'].set_markerfacecolor(_color)
            _line_i['line'].set_data(_data[_line_i['x']], _data[_line_i['y']])

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
    _anim = FuncAnimation(fig, animate, init_func=init,
                          frames=_max_interval, interval=time_per_frame, blit=True)
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
