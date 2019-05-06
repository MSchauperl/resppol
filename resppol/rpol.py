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
import os
# Initialize units
from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.define('bohr = 0.52917721067 * angstrom')

# =============================================================================================
# GLOBAL PARAMETERS
# =============================================================================================
biological_elements = [1, 3, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 34, 35, 53]

ROOT_DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


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

    # Goes through all matches and compares the atom indices.
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
    tmp = []
    for k, ele1 in enumerate(sorted_eq_atoms):
        exclude = 0
        for ele2 in sorted_eq_atoms[:k]:
            if ele1[0] == ele2[0]:
                exclude = 1
        if exclude == 0:
            tmp.append(ele1)
    sorted_eq_atoms = tmp

    return sorted_eq_atoms


# =============================================================================================
# TrainingSet
# =============================================================================================

class TrainingSet:

    def __init__(self, datei):
        self.molecules = list()
        f = open(datei)
        lines = f.readlines()
        f.close()
        for i, line in enumerate(lines):
            mol2file = line.split()[1]
            # noinspection PyTypeChecker
            self.add_molecule(Molecule(mol2file))

    def add_molecule(self, datei):
        self.molecules.append(Molecule(datei))

    def build_matrix_A(self):
        for molecule in self.molecules:
            molecule.build_matrix_A()

    def build_vector_B(self):
        for molecule in self.molecules:
            molecule.build_vector_B()


# =============================================================================================
# Molecule
# =============================================================================================


class Molecule:
    """
        This class loads in a mol2 file and expose all relevant information. It combines the functionality of
        an openeye molecule with the functionality of the OFF molecule. The OE part is only needed to determine
        the chemical equivalent atoms. When this feature is implemented in OFF toolkit the OE molecule is not
        necessary anymore
        :param datei: mol2 file
        :return:
        """

    def __init__(self, datei):

        # Get Molecle name from filename
        self.B = 0.0
        self.A = 0.0
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

        # Label the atoms and bonds using a offxml file
        forcefield = ForceField(os.path.join(ROOT_DIR_PATH, 'resppol/tmp/BCCPOL.offxml'))

        # Run the parameter labeling
        molecule_parameter_list = forcefield.label_molecules(self.offtop)

        # set molecule charge
        # self._charge=openff.Molecule.total_charge
        self._charge = 0

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
            log.error('Conformer and Molecule does not match')
            raise Exception('Conformer and Molecule does not match')
        else:
            self.conformers.append(Conformer(conf, molecule=self))
        # Checks if this conformer is from this moleule based on atom names

    @property
    def chemical_eq_atoms(self):
        """
        # Note: Something similar in compare_charges, which can be copied and slightly modified
        :return:
        """
        return find_eq_atoms(self.oemol)

    def build_matrix_A(self):
        for conformer in self.conformers:
            conformer.build_matrix_A()
            self.A += conformer.A

    def build_vector_B(self):
        for conformer in self.conformers:
            conformer.build_vector_B()
            self.B += conformer.B

    def optimize_charges(self):
        self.build_matrix_A()
        self.build_vector_B()
        self.q = Q_(np.linalg.solve(self.A, self.B), 'elementary_charge')


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

    def __init__(self, conf, molecule=None):
        """
        :param conf: openff molecule file
        """
        # noinspection PyProtectedMember
        self.atom_positions = Q_(np.array(conf.conformers[0]._value), 'angstrom')
        self.atom_positions_angstrom = self.atom_positions.to('angstrom').magnitude
        self.natoms = len(self.atom_positions.magnitude)
        self.baseESP = None
        self.polESP = list()
        self._molecule = molecule

    def get_grid_coord(self):
        self.grid_coord_angstrom = self.baseESP.positions.to('angstrom').magnitude
        self.npoints = len(self.baseESP.positions.magnitude)

    def add_baseESP(self, *args, ):
        self.baseESP = BCCUnpolESP(*args, conformer=self)

        # Check if atomic coordinates match
        for atom1 in self.baseESP.atom_positions:
            atom_is_present = 0
            for atom2 in self.atom_positions:
                if np.linalg.norm(atom1.to('angstrom').magnitude - atom2.to('angstrom').magnitude) < 0.01:
                    atom_is_present = 1
            if atom_is_present == 0:
                raise Exception("ESP grid does not belong to the conformer")

    def add_polESP(self, *args):
        self.polESP.append(BCCPolESP(*args, conformer=self))

    # Build the matrix A for the charge optimization
    def build_matrix_A(self):
        self.get_distances()

        """
        Fast method for only optimizing charges.
        :return:
        """
        # Determine size of matrix for this molecule
        # every atom is one line
        # one line to restrain the overall charge of the molecule
        # one line for every pair of equivalent atoms
        self.Alines = self.natoms + 1 + len(self._molecule.chemical_eq_atoms)
        self.A = np.zeros((self.Alines, self.Alines))
        for j in range(self.natoms):
            for k in range(j + 1):
                self.A[j][k] = np.dot(self.dist[j], self.dist[k])

        # Symmetric matrix -> copy diagonal elements, add total charge restrain
        for j in range(self.natoms):
            for k in range(j):
                self.A[k][j] = self.A[j][k]
            self.A[self.natoms][j] = 1.0
            self.A[j][self.natoms] = 1.0

        # Add charge restraints for equivalent atoms
        for k, eq_atoms in enumerate(self._molecule.chemical_eq_atoms):
            if eq_atoms[1] > 0:
                self.A[self.natoms + 1 + k][eq_atoms[0]] = 1
                self.A[self.natoms + 1 + k][eq_atoms[1]] = -1
                self.A[eq_atoms[0]][self.natoms + 1 + k] = 1
                self.A[eq_atoms[1]][self.natoms + 1 + k] = -1

    # noinspection PyProtectedMember
    def build_vector_B(self):
        """
        Creates the Vector B for the charge fitting.
        :return:
        """
        # Determine size of the vector for this molecule
        # every atom is one line
        # one line to restrain the overall charge of the molecule
        # one line for every pair of equivalent atoms
        self.Alines = self.natoms + 1 + len(self._molecule.chemical_eq_atoms)
        self.B = np.zeros(self.Alines)
        self.esp_values = self.baseESP.esp_values.to('elementary_charge / angstrom').magnitude

        for k in range(self.natoms):
            self.B[k] = np.dot(self.esp_values, self.dist[k])
            self.B[self.natoms] = self._molecule._charge
        for k, eq_atoms in enumerate(self._molecule.chemical_eq_atoms):
            if eq_atoms[1] > 0:
                self.B[self.natoms + 1 + k] = 0.0

    def optimize_charges(self):
        self.qd = np.linalg.solve(self.A, self.B)

    def build_matrix_Abcc(self):
        self.get_distances()

    def get_distances(self):
        self.get_grid_coord()

        # Distances between atoms and ESP points
        self.dist = np.zeros((self.natoms, self.npoints))
        self.dist_3 = np.zeros((self.natoms, self.npoints))
        self.dist_x = np.zeros((self.natoms, self.npoints))
        self.dist_y = np.zeros((self.natoms, self.npoints))
        self.dist_z = np.zeros((self.natoms, self.npoints))

        self.dist = 1. / distance.cdist(self.atom_positions_angstrom, self.grid_coord_angstrom)
        self.dist_3 = np.power(self.dist, 3)  # maybe free afterwards
        self.dist_x = -np.multiply(
            np.subtract.outer(np.transpose(self.atom_positions_angstrom)[0], np.transpose(self.grid_coord_angstrom)[0]),
            self.dist_3)
        # self.dist_x2=np.multiply(np.transpose(np.subtract.outer(np.transpose(self.grid_coord)[0],np.transpose(self.atom_positions)[0])),self.dist_3)
        self.dist_y = -np.multiply(
            np.subtract.outer(np.transpose(self.atom_positions_angstrom)[1], np.transpose(self.grid_coord_angstrom)[1]),
            self.dist_3)
        self.dist_z = -np.multiply(
            np.subtract.outer(np.transpose(self.atom_positions_angstrom)[2], np.transpose(self.grid_coord_angstrom)[2]),
            self.dist_3)
        del self.dist_3

        # Distances between atoms and atoms
        self.diatomic_dist = np.zeros((self.natoms, self.natoms))
        self.diatomic_dist_3 = np.zeros((self.natoms, self.natoms))
        self.diatomic_dist_5 = np.zeros((self.natoms, self.natoms))
        self.diatomic_dist_x = np.zeros((self.natoms, self.natoms))
        self.diatomic_dist_y = np.zeros((self.natoms, self.natoms))
        self.diatomic_dist_z = np.zeros((self.natoms, self.natoms))
        self.diatomic_distb_x = np.zeros((self.natoms, self.natoms))
        self.diatomic_distb_y = np.zeros((self.natoms, self.natoms))
        self.diatomic_distb_z = np.zeros((self.natoms, self.natoms))

        self.diatomic_dist = distance.cdist(self.atom_positions_angstrom, self.atom_positions_angstrom)
        di = np.diag_indices(self.natoms)
        self.diatomic_dist[di] = 1.0E10
        # self.adist=np.fill_diagonal(self.adist,1.0)
        self.diatomic_dist = 1. / self.diatomic_dist
        self.diatomic_dist_3 = np.power(self.diatomic_dist, 3)
        self.diatomic_dist_5 = np.power(self.diatomic_dist, 5)
        self.diatomic_dist[di] = 0.0
        self.diatomic_dist_x = np.multiply(np.subtract.outer(np.transpose(self.atom_positions_angstrom)[0],
                                                             np.transpose(self.atom_positions_angstrom)[0]),
                                           self.diatomic_dist_3)  # X distance between two atoms divided by the dist^3
        self.diatomic_dist_y = np.multiply(np.subtract.outer(np.transpose(self.atom_positions_angstrom)[1],
                                                             np.transpose(self.atom_positions_angstrom)[1]),
                                           self.diatomic_dist_3)
        self.diatomic_dist_z = np.multiply(np.subtract.outer(np.transpose(self.atom_positions_angstrom)[2],
                                                             np.transpose(self.atom_positions_angstrom)[2]),
                                           self.diatomic_dist_3)
        self.diatomic_distb_x = np.subtract.outer(np.transpose(self.atom_positions_angstrom)[0],
                                                  np.transpose(self.atom_positions_angstrom)[
                                                      0])  # X distances between two atoms
        self.diatomic_distb_y = np.subtract.outer(np.transpose(self.atom_positions_angstrom)[1],
                                                  np.transpose(self.atom_positions_angstrom)[1])
        self.diatomic_distb_z = np.subtract.outer(np.transpose(self.atom_positions_angstrom)[2],
                                                  np.transpose(self.atom_positions_angstrom)[2])

    def delete_distances(self):
        """Deletes the all calculated distances to free memory."""
        del self.dist
        del self.dist_x
        del self.dist_y
        del self.dist_z

        del self.adist
        del self.adist_3
        del self.adist_5
        del self.adist_x
        del self.adist_y
        del self.adist_z
        del self.adistb_x
        del self.adistb_y
        del self.adistb_z


# =============================================================================================
# ESPGRID
# =============================================================================================

# noinspection PyTypeChecker
class ESPGRID:
    """
    """

    def define_grid(self, *args, **kwargs):
        for ele in args:
            if 'gesp' in ele:
                self.gridtype = 'gesp'
            if 'espf' in ele:
                self.gridtype = 'psi4'

    def set_ext_e_field(self, vector):
        self._ext_e_field = vector

    def load_grid(self, *args):
        if self.gridtype == 'gesp':
            f = open(args[0], 'r')
            lines = f.readlines()
            f.close()
            for i, line in enumerate(lines):
                if 'ATOMIC' in line and 'COORDINATES' in line:
                    self.natoms = int(line.strip('\n').split()[-1])
                    for j in range(self.natoms):
                        entry = lines[i + 1 + j].replace('D', 'E').split()
                        self.atoms.append(entry[0])
                        self.atom_positions.append(Q_([float(entry[k]) for k in range(1, 4, 1)], 'bohr'))
                if 'GRID' in line:
                    self.ngrid = int(line.strip('\n').split()[-1])
                    grid = lines[i + 1:i + 1 + self.ngrid]
                    break
            # noinspection PyUnboundLocalVariable
            for i, line in enumerate(grid):
                grid[i] = [float(ele) for ele in line.replace('D', 'E').split()]

            self.positions = Q_(np.array(grid)[:, 1:4], 'bohr')
            self.esp_values = Q_(np.array(grid)[:, 0], 'elementary_charge / bohr')
        elif self.gridtype == 'psi4':
            f = open(args[0], 'r')
            lines = f.readlines()
            f.close
            ndata = int(len(lines) / 2) if len(lines) % 2 == 0 else int((len(lines) - 1) / 2)
            grid = np.zeros((ndata, 4))
            for i in range(ndata):
                grid[i] = [float(ele) for ele in lines[2 * i].split()]
            self.positions = Q_(np.array(grid)[:, 0:3], 'angstrom')
            self.esp_values = Q_(np.array(grid)[:, 3], 'elementary_charge / bohr')


# =============================================================================================
# BCCUnpolESP
# =============================================================================================

class BCCUnpolESP(ESPGRID):
    """
    """

    def __init__(self, *args, conformer=None):
        # Decide if we have a Gaussian grid or a psi4 grid
        self.gridtype = None
        self.natoms = -1
        self.atoms = []
        self.atom_positions = []
        self.define_grid(*args)
        self.esp_values = None
        self.positions = None
        self.conformer = conformer

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
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf0/mp2_0.mol2')
    test = Molecule(datei)
    test.add_conformer_from_mol2(datei)

    datei = '/home/michael/resppol/resppol/tmp/butanol/conf1/mp2_1.mol2'
    test.add_conformer_from_mol2(datei)
    espfile = '/home/michael/resppol/resppol/tmp/butanol/conf0/molecule0.gesp'
    test.conformers[0].add_baseESP(espfile, )
    espfile = '/home/michael/resppol/resppol/tmp/butanol/conf1/molecule1.gesp'
    test.conformers[1].add_baseESP(espfile, )
    test.optimize_charges()
    print(test.q[:len(test._atoms)])
    print('FINISH')

    datei = os.path.join(ROOT_DIR_PATH, 'resppol/tmp/mol1/conf1/mol1_conf1.mol2')
    test = Molecule(datei)
    test.add_conformer_from_mol2(datei)
    espfile = '/home/michael/resppol/resppol/tmp/mol1/conf1/mol1_conf1.espf'
    test.conformers[0].add_baseESP(espfile)
    test.optimize_charges()
    print(test.q[:len(test._atoms)])
    print('FINISH')

    datei = os.path.join(ROOT_DIR_PATH, 'resppol/tmp/mol2/conf1/mol2_conf1.mol2')
    test = Molecule(datei)
    test.add_conformer_from_mol2(datei)
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/tmp/mol2/conf2/mol2_conf2.mol2')
    test.add_conformer_from_mol2(datei)
    espfile = '/home/michael/resppol/resppol/tmp/mol2/conf1/mol2_conf1.espf'
    test.conformers[0].add_baseESP(espfile)
    espfile = '/home/michael/resppol/resppol/tmp/mol2/conf2/mol2_conf2.espf'
    test.conformers[1].add_baseESP(espfile)
    test.optimize_charges()
    print(test.q[:len(test._atoms)])
print('FINISH')
