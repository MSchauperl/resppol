"""
Test of the charge fitting process
"""

import pytest
import resppol
import resppol.resppol
import os
from openeye import oechem


ROOT_DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')


#######################################################################################
# The following tests use gaussian ESP files and mol2 files as input
######################################################################################


# @pytest.mark.slow
def test_charge_fitting_1_conformer():
    test = resppol.resppol.Molecule(os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/butanol_0.mol2'))
    test.add_conformer_from_mol2(os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/butanol_0.mol2'))
    test.conformers[0].add_baseESP(os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/butanol_0.gesp'))
    test.optimize_charges()
    charges = [-0.48886482, 0.24846824, 0.2413244, -0.70349397, 0.11170063, 0.11170063, 0.11170063,
               -0.00506223, -0.00506223, -0.02161905, -0.02161905, 0.42082684]
    for i in range(len(charges)):
        assert test.q[i].magnitude == pytest.approx(charges[i], 0.001)
    assert test.q[:len(test._atoms)].units == 'elementary_charge'


def test_find_eq_atoms():
   ifs= oechem.oemolistream(os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/butanol_0.mol2'))
   oemol = oechem.OEMol()
   oechem.OEReadMol2File(ifs, oemol)
   x=resppol.resppol.find_eq_atoms(oemol)
   assert x == [[4, 5], [5, 6], [7, 8], [9, 10]]



def test_find_eq_atoms2():
    ifs= oechem.oemolistream(os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/phenol.mol2'))
    oemol = oechem.OEMol()
    oechem.OEReadMol2File(ifs, oemol)
    x=resppol.resppol.find_eq_atoms(oemol)
    assert x == [[1, 2], [3, 4], [8, 9], [10, 11]]
