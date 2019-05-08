"""
Test of the charge fitting process
"""

import pytest
import resppol
from openeye import oechem
import resppol.rpol
import os

ROOT_DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')


#######################################################################################
# The following tests use gaussian ESP files and mol2 files as input
######################################################################################



@pytest.mark.slow
def test_charge_fitting_1_conformer():
    test = resppol.rpol.Molecule(os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/butanol_0.mol2'))
    test.add_conformer_from_mol2(os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/butanol_0.mol2'))
    test.conformers[0].add_baseESP(os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/butanol_0.gesp'))
    test.optimize_charges()
    charges = [-0.48886482, 0.24846824, 0.2413244, -0.70349397, 0.11170063, 0.11170063, 0.11170063,
               -0.00506223, -0.00506223, -0.02161905, -0.02161905, 0.42082684]
    for i in range(len(charges)):
        assert test.q[i].magnitude == pytest.approx(charges[i], 0.001)
    assert test.q[:len(test._atoms)].units == 'elementary_charge'



def test_charge_fitting_1_conformer_resppol():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf0/mp2_0.mol2')
    test = resppol.rpol.Molecule(datei)
    test.add_conformer_from_mol2(datei)
    espfile = '/home/michael/resppol/resppol/tmp/butanol/conf0/grid.espf'
    test.conformers[0].add_baseESP(espfile)
    test.optimize_charges()
    charges = [-4.31454353e-01,2.01444159e-01,2.99586459e-01,-7.54034555e-01,9.84041761e-02,9.84041761e-02,
               9.84041761e-02,-5.64812072e-04,-5.64812072e-04,-3.05463079e-02,-3.05463079e-02,4.51468003e-01]
    for i in range(len(charges)):
        assert test.q[i].magnitude == pytest.approx(charges[i], 0.001)
    assert test.q[:len(test._atoms)].units == 'elementary_charge'

@pytest.mark.slow
def test_charge_fitting_1_conformer_psi4():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf0/mp2_0.mol2')
    test = resppol.rpol.Molecule(datei)
    test.add_conformer_from_mol2(datei)
    espfile = '/home/michael/resppol/resppol/tmp/butanol/conf0/grid_esp.dat'
    gridfile = '/home/michael/resppol/resppol/tmp/butanol/conf0/grid.dat'
    test.conformers[0].add_baseESP(espfile,gridfile)
    test.optimize_charges()
    charges = [-4.31454353e-01,2.01444159e-01,2.99586459e-01,-7.54034555e-01,9.84041761e-02,9.84041761e-02,
               9.84041761e-02,-5.64812072e-04,-5.64812072e-04,-3.05463079e-02,-3.05463079e-02,4.51468003e-01]
    for i in range(len(charges)):
        assert test.q[i].magnitude == pytest.approx(charges[i], 0.001)
    assert test.q[:len(test._atoms)].units == 'elementary_charge'


@pytest.mark.slow
def test_charge_fitting_2_conformers():
    test = resppol.rpol.Molecule(os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf0/mp2_0.mol2'))
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


#@pytest.mark.slow
def test_load_wrong_conformer():
    with pytest.raises(Exception):
        test = resppol.rpol.Molecule(os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/butanol_0.mol2'))
        test.add_conformer_from_mol2(os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/phenol_0.mol2'))


#@pytest.mark.slow
def test_load_wrong_esp():
    with pytest.raises(Exception):
        test = resppol.rpol.Molecule(os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf0/mp2_0.mol2'))
        test.add_conformer_from_mol2(os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf0/mp2_0.mol2'))
        test.conformers[0].add_baseESP(os.path.join(ROOT_DIR_PATH, 'resppol/tmp/phenol/conf0/molecule0.gesp'))
