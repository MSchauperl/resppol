# Test of teh ngesp functions

from resppol.esp_qalpha import esp, molecule
from resppol.helper import readngesp
import os
import pytest

# Read in ngesp
ROOT_DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')

ngesp, eext, base = readngesp(os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/test.ngesp'))


def test_espvalues():
    result = ngesp[0][0]
    assert result == 'test3.gesp'
