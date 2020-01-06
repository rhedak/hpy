"""
hhpy.ipython.py
~~~~~~~~~~~~~~~~

Contains convenience wrappers for ipython

"""
# standard imports
import pandas as pd

# third party imports
from IPython.core.display import display, HTML

# local imports
from hhpy.main import export


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
def display_full(*args, **kwargs):
    """
    wrapper to display a pandas DataFrame with all rows and columns

    :param args: passed to display
    :param kwargs: passed to display
    :return: None
    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(*args, **kwargs)


@export
def pd_display(*args, number_format='{:,.2f}', full=True, **kwargs):
    """
    wrapper to display a pandas DataFrame with a specified number format

    :param args: passed to display
    :param number_format: the number format to apply
    :param full: whether to show all rows and columns or keep default behaviour
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
    if not isinstance(df, pd.DataFrame):
        _df = pd.DataFrame(df)
    else:
        _df = df.copy()
    del df

    _cols_int = [_ for _ in _df.select_dtypes(int).columns if _ not in exclude]
    _cols_float = [_ for _ in _df.select_dtypes(float).columns if _ not in exclude]

    for _col in _cols_int: _df[_col] = _df[_col].apply(lambda _: format(_, int_format))
    for _col in _cols_float: _df[_col] = _df[_col].apply(lambda _: format(_, float_format))

    if full:
        display_full(_df, **kwargs)
    else:
        display(_df, **kwargs)
