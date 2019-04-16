#!/usr/bin/env python

# =============================================================================================
# MODULE DOCSTRING
# =============================================================================================
"""
This module should convert a mol2 file int a Molecule object and expose all necessary information
.. todo::
   * Load in the molecule
   * Determine the equivalent atoms in the molecule
   * Determine the bonds based on a smirnoff input file
"""

import logging as log

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================
from openeye import oechem
import openforcefield.topology as openff
from openforcefield.typing.engines.smirnoff import ForceField

# =============================================================================================
# GLOBAL PARAMETERS
# =============================================================================================
biological_elements = [1, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 34, 35, 53]


# =============================================================================================
# PRIVATE SUBROUTINES
# =============================================================================================


# =============================================================================================
# Molecule
# =============================================================================================


class Molecule:
    """
    This class loads in a mol2 file and expose all relevant information. It combines the functionalty of
    an openeye molecule with the functionality of the OFF molecule. The OE part is only needed to determine
    the chemical equivalent atoms. When this feature is implemented in OFF toolkit the OE molecule is not
    necessary anymore

    :param datei: mol2 file

    :return:
    """

    def __init__(self, datei):

        # Get Molecle name from filename
        self._name = datei.split('/')[-1].strip(".mol2")

        # Initialize OE Molecule
        self.oemol = oechem.OEMol()

        # Open Input File Stream
        ifs = oechem.oemolistream(datei)

        # Read from IFS to molecule
        oechem.OEReadMol2File(ifs, self.oemol)

        # Check if molecule is a 3 dimensional object
        if self.oemol.GetDimension() != 3:
            log.error('Molecule either not found or does not have a 3D structure')
            raise Exception('Molecule either not found or does not have a 3D structure')

        # Check for strange atoms
        for atom in self.oemol.GetAtoms():
            if atom.GetAtomicNum() not in biological_elements:
                log.warning("""I detect an atom with atomic number: {}
                Are you sure you want to include such an atom in your dataset?
                Maybe you are using a mol2 file with amber atomtypes""".format(atom.GetAtomicNum()))

        # Generate the OFF molecule from the openeye molecule
        self.offmol = openff.Molecule.from_openeye(self.oemol)
        self.offtop = openff.Topology.from_molecules([self.offmol])

        # Labe the atoms and bonds using a offxml file
        forcefield = ForceField('../tmp/BCCPOL.offxml')

        # Run the parameter labeling
        molecule_parameter_list = forcefield.label_molecules(self.offtop)

        # Initialize the bonds, atoms
        self._bonds = list()
        self._atoms = list()

        # Define all BCC bonds
        for i, properties in enumerate(molecule_parameter_list[0]['Bonds'].items()):
            atom_indices, parameter = properties
            self.add_bond(i, atom_indices, parameter.id)

        # Define all atomtypes for polarization
        for i, properties in enumerate(molecule_parameter_list[0]['vdW'].items()):
            atom_index, parameter = properties
            self.add_atom(i, atom_index, parameter.id)

    def add_bond(self, index, atom_indices, parameter_id):
        atom_index1 = atom_indices[0]
        atom_index2 = atom_indices[1]
        self._bonds.append(Bond(index, atom_index1, atom_index2, parameter_id))

    def add_atom(self, index, atom_index, parameter_id):
        self._atoms.append(Atom(index, atom_index[0], parameter_id))


# =============================================================================================
# Atom
# =============================================================================================

class Atom:
    """

    """

    def __init__(self, index, atom_index, parameter_id):
        self._id = index
        self._atom = atom_index
        self._parameter_id = parameter_id


# =============================================================================================
# Bond
# =============================================================================================

class Bond:

    def __init__(self, index, atom_index1, atom_index2, parameter_id):
        self._id = index
        self._atom1 = atom_index1
        self._atom2 = atom_index2
        self._parameter_id = parameter_id
