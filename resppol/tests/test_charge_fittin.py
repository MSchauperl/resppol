"""
Test of the charge fitting process
"""

import pytest
import resppol
import os

ROOT_DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')


#######################################################################################
# The following tests use gaussian ESP files and mol2 files as input
######################################################################################

def test_add():
    result=resppol.add_func(2,3)
    assert result == 5

