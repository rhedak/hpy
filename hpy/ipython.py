"""
hpy.ipython.py
~~~~~~~~~~~~~~~~

Contains convenience wrappers for ipython

"""

import pandas as pd

# third party imports
from IPython.core.display import display, HTML

# some html hacks
# make the notebook wider
wide_notebook = HTML('<style>.container { width:90% !important; }</style>')

# hide code
hide_code = HTML('''
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
''')


def display_full(*args, **kwargs):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(*args, **kwargs)


def pd_display(*args, number_format='{:,.2f}', full=True, **kwargs):
    pd.set_option('display.float_format', number_format.format)

    if full:
        display_full(*args, **kwargs)
    else:
        display(*args, **kwargs)

    pd.reset_option('display.float_format')


def display_df(df, int_format=',', float_format=',.2f', exclude=None, full=True, **kwargs):
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