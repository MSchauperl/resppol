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


# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================

import logging as log
import numpy as np
from openeye import oechem
import openforcefield.topology as openff
from openforcefield.typing.engines.smirnoff import ForceField
from scipy.spatial import distance

# Initialize units
from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.define('bohr = 0.52917721067 * angstrom')

# =============================================================================================
# GLOBAL PARAMETERS
# =============================================================================================
biological_elements = [1, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 34, 35, 53]


# =============================================================================================
# PRIVATE SUBROUTINES
# =============================================================================================
def find_eq_atoms(mol1):
    """
    Finds all equivalent atoms in a molecule.
    :return
    Array of pairs of equivalent atoms.

    :parameter
    mol1: Openeye molecule object

    TODO Include rdkit support for this function
    """

    qmol = oechem.OEQMol()

    # build OEQMol from OEGRAPHMOLECULE
    oechem.OEBuildMDLQueryExpressions(qmol, mol1)
    ss2 = oechem.OESubSearch(qmol)
    oechem.OEPrepareSearch(mol1, ss2)

    # Store the equivalent atoms
    eq_atoms = [[] for i in range(mol1.NumAtoms())]

    # Goes through all matches and compares the atom indeces.
    for count, match in enumerate(ss2.Match(mol1)):
        for ma in match.GetAtoms():
            # if 2 atoms are not the same atoms
            if ma.pattern.GetIdx() != ma.target.GetIdx():
                # and the pair is not stored yet
                if ma.target.GetIdx() not in eq_atoms[ma.pattern.GetIdx()]:
                    # save it to the array
                    eq_atoms[ma.pattern.GetIdx()].append(ma.target.GetIdx())

    # goes through the array and returns pairs of equivalent atoms
    sorted_eq_atoms = []
    for i, ele in enumerate(eq_atoms):
        for element in ele:
            # Making sure we have not already covered this pair of atoms
            if [element, i] not in sorted_eq_atoms:
                sorted_eq_atoms.append([i, element])

    return (sorted_eq_atoms)


# =============================================================================================
# TrainingSet
# =============================================================================================

class TrainingSet():

    def __init__(self, datei):
        self.molecules = list()
        f = open(datei)
        lines = f.readlines()
        f.close()
        for i, line in enumerate(lines):
            mol2file = line.split()[1]
            add_molecule(Molecule(mol2file))


def add_molecule(self, datei):
    self.molecules.append(Molecule(datei))


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
        forcefield = ForceField('./tmp/BCCPOL.offxml')

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

        # Initialize conformers
        self.conformers = list()

    def add_bond(self, index, atom_indices, parameter_id):
        atom_index1 = atom_indices[0]
        atom_index2 = atom_indices[1]
        self._bonds.append(Bond(index, atom_index1, atom_index2, parameter_id))

    def add_atom(self, index, atom_index, parameter_id):
        self._atoms.append(Atom(index, atom_index[0], parameter_id))

    def add_conformer_from_mol2(self, mol2file):

        conf = openff.Molecule.from_file(mol2file)

        # Check if molecule is a 3 dimensional object and has the correct dimensions
        # Checks if this conformer is from this molecule based on atom names
        if self.offmol.n_atoms != conf.n_atoms or \
                self.offmol.n_bonds != conf.n_bonds or \
                self.offmol.n_angles != conf.n_angles:
            log.error('Molecule either not found or does not have a 3D structure')
            raise Exception('Molecule either not found or does not have a 3D structure')
        else:
            self.conformers.append(Conformer(conf))
        # Checks if this conformer is from this moleule based on atom names

    @property
    def chemical_eq_atoms(self):
        """
        # Note: Something similar in compare_charges, which can be copied and slightly modified
        :return:
        """
        return find_eq_atoms(self.oemol)

    def build_matrix_A(self):
        for conformer in self.conformers():
            conformer.build_matrix_A()


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


# =============================================================================================
# Conformer
# =============================================================================================

class Conformer:
    """

    """

    def __init__(self, conf):
        """

        :param conf: openff molecule file
        """
        self.atom_positions = Q_(conf.conformers[0]._value, 'angstrom')
        self.natoms=len(self.atom_positions.magnitude)
        self.baseESP = None
        self.polESP = list()

    def get_grid_coord(self, grid_txt):
        self.grid_coord = self.baseESP.positions
        self.npoints = len(self.baseESP.positions.magnitude)

    def get_distances(self):
        pass

    def add_baseESP(self, *args):
        self.baseESP = BCCUnpolESP(*args)

    def add_polESP(self, *args):
        self.polESP.append(BCCPolESP(*args))

    def build_Matrix_A(self):
        self.get_distances()

    def get_distances(self):
        self.get_grid_coord()

        # Distances between atoms and ESP points
        self.dist = np.zeros((self.natoms, self.npoints))
        self.dist_3 = np.zeros((self.natoms, self.npoints))
        self.dist_x = np.zeros((self.natoms, self.npoints))
        self.dist_y = np.zeros((self.natoms, self.npoints))
        self.dist_z = np.zeros((self.natoms, self.npoints))

        self.dist = 1. / distance.cdist(self.atom_positions, self.grid_coord)
        self.dist_3 = np.power(self.dist, 3)  # maybe free afterwards
        self.dist_x = -np.multiply(np.subtract.outer(np.transpose(self.atom_positions)[0], np.transpose(self.grid_coord)[0]),
                                   self.dist_3)
        # self.dist_x2=np.multiply(np.transpose(np.subtract.outer(np.transpose(self.grid_coord)[0],np.transpose(self.atom_positions)[0])),self.dist_3)
        self.dist_y = -np.multiply(np.subtract.outer(np.transpose(self.atom_positions)[1], np.transpose(self.grid_coord)[1]),
                                   self.dist_3)
        self.dist_z = -np.multiply(np.subtract.outer(np.transpose(self.atom_positions)[2], np.transpose(self.grid_coord)[2]),
                                   self.dist_3)
        del self.dist_3

        # Distances between atoms and atoms
        self.adist = np.zeros((self.natoms, self.natoms))
        self.adist_3 = np.zeros((self.natoms, self.natoms))
        self.adist_5 = np.zeros((self.natoms, self.natoms))
        self.adist_x = np.zeros((self.natoms, self.natoms))
        self.adist_y = np.zeros((self.natoms, self.natoms))
        self.adist_z = np.zeros((self.natoms, self.natoms))
        self.adistb_x = np.zeros((self.natoms, self.natoms))
        self.adistb_y = np.zeros((self.natoms, self.natoms))
        self.adistb_z = np.zeros((self.natoms, self.natoms))

        self.adist = distance.cdist(self.atom_positions, self.atom_positions)
        di = np.diag_indices(self.natoms)
        self.adist[di] = 1.0E10
        # self.adist=np.fill_diagonal(self.adist,1.0)
        self.adist = 1. / self.adist
        self.adist_3 = np.power(self.adist, 3)
        self.adist_5 = np.power(self.adist, 5)
        self.adist[di] = 0.0
        self.adist_x = np.multiply(np.subtract.outer(np.transpose(self.atom_positions)[0], np.transpose(self.atom_positions)[0]),
                                   self.adist_3)  # X distance between two atoms divided by the dist^3
        self.adist_y = np.multiply(np.subtract.outer(np.transpose(self.atom_positions)[1], np.transpose(self.atom_positions)[1]),
                                   self.adist_3)
        self.adist_z = np.multiply(np.subtract.outer(np.transpose(self.atom_positions)[2], np.transpose(self.atom_positions)[2]),
                                   self.adist_3)
        self.adistb_x = np.subtract.outer(np.transpose(self.atom_positions)[0],
                                          np.transpose(self.atom_positions)[0])  # X distances between two atoms
        self.adistb_y = np.subtract.outer(np.transpose(self.atom_positions)[1], np.transpose(self.atom_positions)[1])
        self.adistb_z = np.subtract.outer(np.transpose(self.atom_positions)[2], np.transpose(self.atom_positions)[2])


# =============================================================================================
# ESPGRID
# =============================================================================================

class ESPGRID:
    """

    """

    def define_grid(self, *args, **kwargs):
        for ele in args:
            if 'gesp' in ele:
                self.gridtype = 'gesp'
            if 'grid.dat' in ele:
                self.gridtype = 'psi4'

    def set_ext_e_field(self, vector):
        self._ext_e_field = vector

    def load_grid(self, *args):
        if self.gridtype == 'gesp':
            f = open(args[0], 'r')
            lines = f.readlines()
            f.close()
            for i, line in enumerate(lines):
                if 'ATOMIC' in line and 'COORDINTES' in line:
                    self.natoms = int(line.strip('\n')[-1])
                if 'GRID' in line:
                    self.ngrid = int(line.strip('\n')[-1])
                    grid = lines[i + 1:i + 1 + self.ngrid]
                    break
            for i, line in enumerate(grid):
                grid[i] = line.replace('D', 'E').split()
            self.positions = Q_(np.array(grid)[:, 1], 'bohr')
            self.esp_values = Q_(np.array(grid)[:, 0], 'elementary_charge / bohr')
        elif self.gridtype == 'psi4':
            self.positions = Q_(np.loadtxt(args[0]), 'angstrom')
            self.esp_values = Q_(np.loadtxt(args[1]), 'elementary_charge / bohr')


# =============================================================================================
# BCCUnpolESP
# =============================================================================================

class BCCUnpolESP(ESPGRID):
    """

    """

    def __init__(self, *args):
        # Decide if we have a Gaussian grid or a psi4 grid
        self.gridtype = None
        self.natoms = -1
        self.define_grid(*args)
        self.esp_values = None
        self.positions = None

        # External e-field is 0 in all directions
        vector = Q_([0, 0, 0], 'elementary_charge / bohr / bohr')
        self.set_ext_e_field(vector)

        self.load_grid(*args)


# =============================================================================================
# BCCPolESP
# =============================================================================================

class BCCPolESP(ESPGRID):
    """

    """

    def __init__(self, ESPfile, ext_e_field):
        pass


if __name__ == '__main__':
    datei = '/home/michael/resppol/resppol/tmp/butanolsybyl.mol2'
    test = Molecule(datei)
    test.add_conformer_from_mol2(datei)
    test.conformers
    espfile = '/home/michael/resppol/resppol/tmp/butanol/conf0/molecule0.gesp'
    # test.conformers[0].add_polESP(espfile, "vector")
    test.conformers[0].add_baseESP(espfile, )
    print('FINISH')
