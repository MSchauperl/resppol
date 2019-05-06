"""
Test of the charge fitting process
"""

import pytest
import resppol.rpol as rpol
import os

ROOT_DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')


#######################################################################################
# The following tests use gaussian ESP files and mol2 files as input
######################################################################################


"""
@pytest.mark.slow
def test_charge_fitting_1_conformer():
    test = rpol.Molecule(os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf0/mp2_0.mol2'))
    test.add_conformer_from_mol2(os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf0/mp2_0.mol2'))
    test.conformers[0].add_baseESP(os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf0/molecule0.gesp'))
    test.optimize_charges()
    charges = [-0.48886482, 0.24846824, 0.2413244, -0.70349397, 0.11170063, 0.11170063, 0.11170063,
               -0.00506223, -0.00506223, -0.02161905, -0.02161905, 0.42082684]
    for i in range(len(charges)):
        assert test.q[i].magnitude == pytest.approx(charges[i], 0.001)
    assert test.q[:len(test._atoms)].units == 'elementary_charge'


@pytest.mark.slow
def test_charge_fitting_2_conformers():
    test = rpol.Molecule(os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf0/mp2_0.mol2'))
    test.add_conformer_from_mol2(os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf0/mp2_0.mol2'))
    test.add_conformer_from_mol2(os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf1/mp2_1.mol2'))
    test.conformers[0].add_baseESP(os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf0/molecule0.gesp'))
    test.conformers[1].add_baseESP(os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf1/molecule1.gesp'))
    test.optimize_charges()
    charges = [-0.28489849, 0.09338576, 0.1448086, -0.57195518, 0.06795839, 0.06795839, 0.06795839, 0.01309174,
               0.01309174,
               0.02570774, 0.02570774, 0.33718519]
    for i in range(len(charges)):
        assert test.q[i].magnitude == pytest.approx(charges[i], 0.001)
    assert test.q[:len(test._atoms)].units == 'elementary_charge'

"""
@pytest.mark.slow
def test_load_wrong_conformer():
    with pytest.raises(Exception):
        test = rpol.Molecule(os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/butanol_0.mol2'))
        test.add_conformer_from_mol2(os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/phenol_0.mol2'))

"""
@pytest.mark.slow
def test_load_wrong_esp():
    with pytest.raises(Exception):
        test = rpol.Molecule(os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf0/mp2_0.mol2'))
        test.add_conformer_from_mol2(os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf0/mp2_0.mol2'))
        test.conformers[0].add_baseESP(os.path.join(ROOT_DIR_PATH, 'resppol/tmp/phenol/conf0/molecule0.gesp'))
"""