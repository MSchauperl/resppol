import pytest
import resppol
from openeye import oechem
import resppol.rpol
import os
from pint import UnitRegistry

ROOT_DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')

ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.define('bohr = 0.52917721067 * angstrom')
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
        assert charge.magnitude == pytest.approx(0.0, 0.001)


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
    test.molecules[0].conformers[0].delete_distances()
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

def test_multiple_esps3():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test2.mol2')
    test = resppol.rpol.TrainingSet()
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3_Z+.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field = [0,0,1])
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3_Z-.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile,e_field = [0,0,-1])
    test.optimize_charges()
    test_charges = [8.633, -8.633]
    for i, charge in enumerate(test.q[:2]):
        assert charge.magnitude == pytest.approx(test_charges[i], 0.01)


def test_1_confomer_polarization():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test2.mol2')
    test = resppol.rpol.TrainingSet()
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3_Z+.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, 1.0], 'elementary_charge / bohr / bohr'))
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3_Z-.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, -1.0], 'elementary_charge / bohr / bohr') )
    test.optimize_charges_alpha()
    assert test.q_alpha[0] == pytest.approx(8.6330,0.01)
    assert test.q_alpha[1] == pytest.approx(-8.6330,0.01)
    for ele in test.q_alpha[3:9]:
        assert ele == pytest.approx(3.40, 0.01)
    # Check e field
    assert test.molecules[0].conformers[0].polESPs[0].e_field_at_atom[1][0] == pytest.approx(-0.0, 0.002)

def test_intermolecular_pol_rst():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test1.mol2')
    test = resppol.rpol.TrainingSet(mode='q_alpha')
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test1.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test1_Z+.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, 1.0], 'elementary_charge / bohr / bohr'))
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test1_Z-.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, -1.0], 'elementary_charge / bohr / bohr') )
    test.optimize_charges_alpha()
    assert test.q_alpha[0] == pytest.approx(0.0,0.01)
    assert test.q_alpha[1] == pytest.approx(0.0,0.01)
    for ele in test.q_alpha[4:10]:
        assert ele == pytest.approx(3.40, 0.01)
    # Check e field
    assert test.molecules[0].conformers[0].polESPs[0].e_field_at_atom[1][0] == pytest.approx(-0.0, 0.002)



def test_1_confomer_efield():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test2.mol2')
    test = resppol.rpol.TrainingSet(scaleparameters=[1,1,1])
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3_Z+.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, 1.0], 'elementary_charge / bohr / bohr'))
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3_Z-.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, -1.0], 'elementary_charge / bohr / bohr') )
    test.optimize_charges_alpha()
    # Check e field
    assert test.molecules[0].conformers[0].polESPs[0].e_field_at_atom[1][0] == pytest.approx(-0.319, 0.01)



def test_1_confomer_polarization_SCF():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test2.mol2')
    test = resppol.rpol.TrainingSet(mode='q_alpha',SCF= True, scf_scaleparameters=[1,1,1], scaleparameters=[1,1,1])
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3_Z+.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, 1.0], 'elementary_charge / bohr / bohr'))
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3_Z-.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, -1.0], 'elementary_charge / bohr / bohr') )
    test.optimize_charges_alpha()
    assert test.q_alpha[0] == pytest.approx(8.93,0.01)
    assert test.q_alpha[1] == pytest.approx(-8.93,0.01)
    for ele in test.q_alpha[3:9]:
        assert ele == pytest.approx(3.48, 0.01)

    # Check e field
    assert test.molecules[0].conformers[0].polESPs[0].e_field_at_atom[1][0] == pytest.approx(-0.335, 0.01)



def test_2_molecules_polarization_SCF():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test2.mol2')
    test = resppol.rpol.TrainingSet(mode='q_alpha',SCF= True, scf_scaleparameters=[1,1,1], scaleparameters=[1,1,1])
    test.add_molecule(datei)
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    test.molecules[1].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    test.molecules[1].conformers[0].add_baseESP(espfile)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3_Z+.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, 1.0], 'elementary_charge / bohr / bohr'))
    test.molecules[1].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, 1.0], 'elementary_charge / bohr / bohr'))
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3_Z-.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, -1.0], 'elementary_charge / bohr / bohr') )
    test.molecules[1].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, -1.0], 'elementary_charge / bohr / bohr') )
    test.optimize_charges_alpha()
    assert test.q_alpha[0] == pytest.approx(8.93,0.01)
    assert test.q_alpha[1] == pytest.approx(-8.93,0.01)
    assert test.molecules[1].q_alpha[0] == pytest.approx(8.93,0.01)
    assert test.molecules[1].q_alpha[1] == pytest.approx(-8.93,0.01)
    for ele in test.q_alpha[3:9]:
        assert ele == pytest.approx(3.48, 0.01)
    for ele in test.q_alpha[16:22]:
        assert ele == pytest.approx(3.48, 0.01)
    for ele in test.molecules[0].q_alpha[3:9]:
        assert ele == pytest.approx(3.48, 0.01)
    for ele in test.molecules[0].q_alpha[3:9]:
        assert ele == pytest.approx(3.48, 0.01)

    # Check e field
    assert test.molecules[0].conformers[0].polESPs[0].e_field_at_atom[1][0] == pytest.approx(-0.335, 0.01)




def test_1_confomer_polarization_SCF_rotx():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test2_rotatedz.mol2')
    test = resppol.rpol.TrainingSet(mode='q_alpha',SCF= True, scf_scaleparameters=[1,1,1], scaleparameters=[1,1,1])
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test6.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test6_Z+.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.7071, 0.7071], 'elementary_charge / bohr / bohr'))
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test6_Z-.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, -0.7071, -0.7071], 'elementary_charge / bohr / bohr') )
    test.optimize_charges_alpha()
    assert test.q_alpha[0] == pytest.approx(8.93,0.01)
    assert test.q_alpha[1] == pytest.approx(-8.93,0.01)
    for ele in test.q_alpha[3:9]:
        assert ele == pytest.approx(3.48, 0.01)



def test_1_confomer_polarization_SCF_rotxz():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test2_rotatedx.mol2')
    test = resppol.rpol.TrainingSet(mode='q_alpha',SCF= True, scf_scaleparameters=[1,1,1], scaleparameters=[1,1,1])
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test4.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test4_Z+.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.7071, 0.7071, 0.0], 'elementary_charge / bohr / bohr'))
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test4_Z-.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([-0.7071, -0.7071, 0.0], 'elementary_charge / bohr / bohr') )
    test.optimize_charges_alpha()
    assert test.q_alpha[0] == pytest.approx(8.93,0.01)
    assert test.q_alpha[1] == pytest.approx(-8.93,0.01)
    for ele in test.q_alpha[3:9]:
        assert ele == pytest.approx(3.48, 0.01)



