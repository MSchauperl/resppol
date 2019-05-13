import pytest
import resppol
from openeye import oechem
import resppol.rpol
import os

ROOT_DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')


#######################################################################################
# The following tests are based on 2 or 3 atoms and can be calculate easily on a piece of paper as well.
######################################################################################


def test_simple_charges():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test2.mol2')
    test = resppol.rpol.TrainingSet()
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test1.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    test.optimize_charges()
    for charge in test.molecules[0].q:
        assert charge == pytest.approx(0.0, 0.001)


def test_simple_charges2():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test2.mol2')
    test = resppol.rpol.TrainingSet()
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    test.optimize_charges()
    test_charges = [8.633, -8.633]
    for i, charge in enumerate(test.q[:2]):
        assert charge.magnitude == pytest.approx(test_charges[i], 0.01)

def test_simple_rotatedx():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test2_rotatedx.mol2')
    test = resppol.rpol.TrainingSet()
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test5.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    test.optimize_charges()
    test_charges = [4.3, -4.3]
    for i, charge in enumerate(test.q[:2]):
        assert charge.magnitude == pytest.approx(test_charges[i], 0.01)


def test_simple_rotatedx2():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test2_rotatedx.mol2')
    test = resppol.rpol.TrainingSet()
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test4.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    test.optimize_charges()
    test_charges = [8.633, -8.633]
    for i, charge in enumerate(test.q[:2]):
        assert charge.magnitude == pytest.approx(test_charges[i], 0.01)

def test_simple_rotatedz():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test2_rotatedz.mol2')
    test = resppol.rpol.TrainingSet()
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test6.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    test.optimize_charges()
    test_charges = [8.633, -8.633]
    for i, charge in enumerate(test.q[:2]):
        assert charge.magnitude == pytest.approx(test_charges[i], 0.01)