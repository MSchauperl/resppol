import pytest
import os
import resppol.utilities as util
from openeye import oechem

ROOT_DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')

#######################################################################################
# This module test the functions and classes defined in resppol.utilities.py
######################################################################################


def test_read_mulliken_from_g09():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/hfAM1Mulliken.log')
    element_charge_array = util.read_mulliken_from_g09(datei)
    assert len(element_charge_array) == 12
    assert element_charge_array[3][0] == 'O'
    assert element_charge_array[3][1] == '-0.333431'
    assert element_charge_array[2][1] != '-0.333431'

def test_load_g09_charges_to_oemol():
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/hfAM1Mulliken.log')
    mol_datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/butanol_0.mol2')
    oemol = oechem.OEGraphMol()
    ifs = oechem.oemolistream()
    ifs.SetFormat(oechem.OEFormat_MOL2)
    if ifs.open(mol_datei):
        oechem.OEReadMolecule(ifs, oemol)
    util.load_g09_charges_to_oemol(oemol,datei)
    for atom in oemol.GetAtoms(oechem.OEHasAtomicNum(oechem.OEElemNo_O)):
        assert atom.GetPartialCharge() == pytest.approx(-0.333431,0.001)

def test_load_g09_charges_to_oemol_wrong_molecule():
    with pytest.raises(LookupError):
        datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/hfAM1Mulliken.log')
        mol_datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/phenol_0.mol2')
        oemol = oechem.OEGraphMol()
        ifs = oechem.oemolistream()
        ifs.SetFormat(oechem.OEFormat_MOL2)
        if ifs.open(mol_datei):
            oechem.OEReadMolecule(ifs, oemol)
        util.load_g09_charges_to_oemol(oemol,datei)
