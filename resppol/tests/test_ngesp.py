# Test of teh ngesp functions

from resppol.esp_qalpha import esp, molecule
from resppol.helper import readngesp
import pytest 

#Read in ngesp
ngesp, eext, base = readngesp('resppol/data/test_data/test.ngesp')


def test_espvalues():
    result=ngesp[0][0]
    assert result == 'test3.gesp'

