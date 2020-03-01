"""
hhpy.ipython.py
~~~~~~~~~~~~~~~

Contains convenience wrappers for ipython

"""
# ---- imports
# --- standard imports
import pandas as pd
# --- third party imports
from IPython.display import display, HTML
# --- local imports
from hhpy.main import export, assert_list, list_exclude


# ---- functions
# --- export
@export
def wide_notebook(width: int = 90):
    """
    makes the jupyter notebook wider by appending html code to change the width,
     based on https://stackoverflow.com/questions/21971449/how-do-i-increase-the-cell-width-of-the-jupyter-
     ipython-notebook-in-my-browser

    :param: width in percent, default 90 [optional]
    :return: None
    """
    # noinspection PyTypeChecker
    display(HTML('<style>.container { width:{}% !important; }</style>'.format(width)))


@export
def hide_code():
    """
    hides the code and introduces a toggle button
     based on https://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized
     -with-nbviewer

    :return: None
    """

    # noinspection PyTypeChecker
    display(HTML('''
        <script>
        code_show=true; 
        function code_toggle() {
         if (code_show){
         $('div.input').hide();
         } else {
         $('div.input').show();
         }
         code_show = !code_show
        } 
        $( document ).ready(code_toggle);
        </script>
        The raw code for this IPython notebook is by default hidden for easier reading.
        To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.
    '''))


@export
def display_full(*args, rows=None, cols=None, **kwargs):
    """
    wrapper to display a pandas DataFrame with all rows and columns

    :param rows: number of rows to display, defaults to all
    :param cols: number of columns to display, defaults to all
    :param args: passed to display
    :param kwargs: passed to display
    :return: None
    """
    with pd.option_context('display.max_rows', rows, 'display.max_columns', cols):
        display(*args, **kwargs)


@export
def pd_display(*args, number_format='{:,.2f}', full=True, **kwargs):
    """
    wrapper to display a pandas DataFrame with a specified number format

    :param args: passed to display
    :param number_format: the number format to apply
    :param full: whether to use :func:`~display_full` (True) or standard display (False)
    :param kwargs: passed to display
    :return: None
    """
    pd.set_option('display.float_format', number_format.format)

    if full:
        display_full(*args, **kwargs)
    else:
        display(*args, **kwargs)

    pd.reset_option('display.float_format')


@export
def display_df(df, int_format=',', float_format=',.2f', exclude=None, full=True, **kwargs):
    """
    Wrapper to display a pandas DataFrame with separate options for int / float, also adds an option to exclude columns

    :param df: pandas DataFrame to display
    :param int_format: format for integer columns
    :param float_format: format for float columns
    :param exclude: columns to exclude
    :param full: whether to show all rows and columns or keep default behaviour
    :param kwargs: passed to display
    :return: None
    """

    if exclude is None:
        exclude = []
    else:
        exclude = assert_list(exclude)
    # avoid inplace
    df = pd.DataFrame(df).copy()

    _cols_int = list_exclude(df.select_dtypes(int).columns, exclude)
    _cols_float = list_exclude(df.select_dtypes(float).columns, exclude)

    for _col in _cols_int:
        df[_col] = df[_col].apply(lambda _: format(_, int_format) if not pd.isna(_) else '<NaN>')
    for _col in _cols_float:
        df[_col] = df[_col].apply(lambda _: format(_, float_format) if not pd.isna(_) else '<NaN>')

    if full:
        display_full(df, **kwargs)
    else:
        display(df, **kwargs)


# --- pandas styles
@export
def highlight_max(df: pd.DataFrame, color: str = 'xkcd:cyan') -> pd.DataFrame:
    """
    highlights the largest value in each column of a pandas DataFrame

    :param df: pandas DataFrame
    :param color: color used for highlighting
    :return: the pandas DataFrame with the style applied to it
    """
    def cond_max(s: pd.Series):
        return ['background-color: {}'.format(color) if v else '' for v in s == s.max()]

    return df.style.apply(cond_max)


@export
def highlight_min(df: pd.DataFrame, color: str = 'xkcd:light red') -> pd.DataFrame:
    """
    highlights the smallest value in each column of a pandas DataFrame

    :param df: pandas DataFrame
    :param color: color used for highlighting
    :return: the pandas DataFrame with the style applied to it
    """
    def cond_min(s: pd.Series):
        return ['background-color: {}'.format(color) if v else '' for v in s == s.min()]

    return df.style.apply(cond_min)


@export
def highlight_max_min(df: pd.DataFrame, max_color: str = 'xkcd:cyan', min_color: str = 'xkcd:light red'):
    """
    highlights the largest and smallest value in each column of a pandas DataFrame

    :param df: pandas DataFrame
    :param max_color: color used for highlighting largest value
    :param min_color: color used for highlighting smallest value
    :return: the pandas DataFrame with the style applied to it
    """
    def cond_max_min(s):

        _out = []

        for _i in s:

            if _i == s.max():
                _out.append('background-color: {}'.format(max_color))
            elif _i == s.min():
                _out.append('background-color: {}'.format(min_color))
            else:
                _out.append('')

        return _out

    return df.style.apply(cond_max_min)
