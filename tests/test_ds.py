"""
tests for hhpy.ds
"""

import pytest
import hhpy.main as hmn
import hhpy.ds as hds
import pandas as pd
import os


# --- fixtures
@pytest.fixture
def testdata_jp():
    _df = pd.read_excel('files/testdata_jp.xlsx')
    return _df


@pytest.fixture
def testdata_jp_dfmapping(testdata_jp):
    _mapping = hds.DFMapping()
    _mapping.from_df(testdata_jp, trans=True, trans_src='ja')
    return _mapping


# --- tests
def test_init_dfmapping(testdata_jp_dfmapping):
    assert testdata_jp_dfmapping.col_mapping and testdata_jp_dfmapping.value_mapping


def test_dfmapping_to_excel(testdata_jp_dfmapping: hds.DFMapping):
    _path = 'tmp/testdata_jp_dfmapping.xlsx'
    testdata_jp_dfmapping.to_excel(_path)
    assert os.path.exists(_path)
    os.remove(_path)


def test_dfmapping_from_excel():
    _path = 'files/testdata_jp_dfmapping.xlsx'
    _testdata_jp_dfmapping = hds.DFMapping(_path)
    # print(_testdata_jp_dfmapping)
    assert _testdata_jp_dfmapping.col_mapping and _testdata_jp_dfmapping.value_mapping
