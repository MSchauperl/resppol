# !/usr/bin/env python

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
import copy
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


def return_sq_efield(matrix):
    dim1 = len(matrix)
    dim2 = len(matrix[0])
    sq_efield = np.zeros((dim1, dim1, dim2))
    for i, vector1 in enumerate(matrix):
        for j, vector2 in enumerate(matrix):
            sq_efield[i][j] = vector1 * vector2
    return sq_efield


# =============================================================================================
# TrainingSet
# =============================================================================================

class TrainingSet():
    """
    This file tells the program where it can find all the necessary input files.

    """

    def __init__(self, mode='q', scaleparameters=None, scf_scaleparameters=None, SCF=False, thole=False):
        self.molecules = list()
        self.B = np.zeros(0)
        self.A = np.zeros((0, 0))
        self.q = 0.0
        self.mode = mode
        self.scf_scaleparameters = scf_scaleparameters
        self.scaleparameters = scaleparameters
        self._SCF = SCF
        self._thole = thole
        self._mode = mode
        self.step = 0

    def load_from_file(self):
        f = open(datei)
        lines = f.readlines()
        f.close()
        for i, line in enumerate(lines):
            mol2file = line.split()[1]
            # noinspection PyTypeChecker
            self.add_molecule(Molecule(mol2file))

    def add_molecule(self, datei):
        self.number_of_lines_in_X = 0
        self.molecules.append(Molecule(datei, position=self.number_of_lines_in_X, trainingset=self))
        self.number_of_lines_in_X += self.molecules[-1]._lines_in_X

    def build_matrix_A(self):
        """
        Combines the matrixes of  esp objects in the diagonal

        Lagrange Multipliers have to be applied afterwords. Otherwise the optimisations
        are independent
        This function is only used for RESP-Pol
        """
        for molecule in self.molecules:
            molecule.build_matrix_A()
            X12 = np.zeros((len(self.A), len(molecule.A)))
            self.A = np.concatenate(
                (np.concatenate((self.A, X12), axis=1), np.concatenate((X12.transpose(), molecule.A), axis=1)), axis=0)

    def build_vector_B(self):
        for molecule in self.molecules:
            molecule.build_vector_B()
            self.B = np.concatenate((self.B, molecule.B))

    # def get_intramolecular_charge_rst()

    @property
    def get_intramolecular_polarization_rst(self):

        intramolecular_polarization_rst = []
        first_occurrence_of_parameter = {}
        for molecule in self.molecules:
            first_occurrence_of_parameter_in_molecule = {}
            for atom in molecule._natoms:
                if atom._parameter_id not in first_occurrence_of_parameter.keys():
                    first_occurrence_of_parameter[atom._parameter_id] = molecule._position_in_A + atom._id
                elif atom._parameter_id not in first_occurrence_of_parameter_in_molecule.keys():
                    intramolecular_polarization_rst.append(
                        [first_occurrence_of_parameter[atom._parameter_id], molecule._position_in_A + atom._id])

    def optimize_charges(self):
        self.build_matrix_A()
        self.build_vector_B()
        self.q = Q_(np.linalg.solve(self.A, self.B), 'elementary_charge')

        # Update the charges of the molecules below
        q_tmp = self.q
        for molecule in self.molecules:
            molecule.q = q_tmp[:len(molecule.A)]
            q_tmp = q_tmp[len(molecule.A):]
            molecule.update_q()


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

    def __init__(self, datei, position=0, trainingset=None):

        # Get Molecle name from filename
        self.B = 0.0
        self.A = 0.0
        self._name = datei.split('/')[-1].strip(".mol2")
        self._trainingset = trainingset

        # Number of optimization steps
        self.step = 0

        # Postion of this molecule in optimization matrix A
        self.position_in_A = position

        # Copy (scf) scaleparameters from the trainingset definition
        if trainingset == None:
            self.scf_scaleparameters = None
            self.scaleparameters = None
            self._thole = False
            self._SCF = False
            self._mode = 'q'
        else:
            self.scf_scaleparameters = self._trainingset.scf_scaleparameters
            self.scaleparameters = self._trainingset.scaleparameters
            self._thole = self._trainingset._thole
            self._SCF = self._trainingset._SCF
            self._mode = self._trainingset._mode
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
                raise Warning("""I detect an atom with atomic number: {}
                Are you sure you want to include such an atom in your dataset?
                Maybe you are using a mol2 file with amber atomtypes""".format(atom.GetAtomicNum()))

        # Generate the OFF molecule from the openeye molecule
        self.offmol = openff.Molecule.from_openeye(self.oemol)
        self.offtop = openff.Topology.from_molecules([self.offmol])

        # Label the atoms and bonds using a offxml file
        forcefield = ForceField(os.path.join(ROOT_DIR_PATH, 'resppol/data/test_data/BCCPOL.offxml'))

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

        self._nbonds = len(self._bonds)

        # Define all atomtypes for polarization
        for i, properties in enumerate(molecule_parameter_list[0]['vdW'].items()):
            atom_index, parameter = properties
            self.add_atom(i, atom_index, parameter.id)

        self._natoms = len(self._atoms)

        # Initialize and fill scaling matrix
        self.scale = np.ones((self._natoms, self._natoms))
        self.scale_scf = np.ones((self._natoms, self._natoms))
        self.scaling(scf_scaleparameters=self.scf_scaleparameters)

        # Initialize conformers
        self.conformers = list()

        # Number of lines for matrix X
        if self._mode == 'q':
            self._lines_in_X = self._natoms + len(self.chemical_eq_atoms) + 1
        if self._mode == 'q_alpha':
            self._lines_in_X = self._natoms + len(
                self.chemical_eq_atoms) + 1 + 3 * self._natoms + 2 * self._natoms + len(self.same_polarization_atoms)

        # Initiliaxe charges
        self.q_alpha = np.zeros(self._lines_in_X)

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

    @property
    def same_polarization_atoms(self):
        array_of_same_pol_atoms = []
        first_occurrence = {}
        for atom in self._atoms:
            if atom._parameter_id not in first_occurrence.keys():
                first_occurrence[atom._parameter_id] = atom._id
            else:
                array_of_same_pol_atoms.append([first_occurrence[atom._parameter_id], atom._id])
        return (array_of_same_pol_atoms)

    def build_matrix_A(self):
        for conformer in self.conformers:
            conformer.build_matrix_A()
            self.A += conformer.A

    def build_vector_B(self):
        for conformer in self.conformers:
            conformer.build_vector_B()
            self.B += conformer.B

    def build_matrix_X(self):
        for conformer in self.conformers:
            conformer.build_matrix_X()
            self.X += conformer.X

    def build_vector_Y(self):
        for conformer in self.conformers:
            conformer.build_vector_Y()
            self.B += conformer.Y

    def optimize_charges(self):
        self.build_matrix_A()
        self.build_vector_B()
        self.q = Q_(np.linalg.solve(self.A, self.B), 'elementary_charge')

    def optimize_charges_alpha(self):
        self.build_matrix_X()
        self.build_vector_Y()
        self.q_alpha = Q_(np.linalg.solve(self.X, self.Y), 'elementary_charge')

    def update_q(self):
        for conformer in self.conformers:
            conformer.q = self.q

    def scaling(self, scaleparameters=None, scf_scaleparameters=None):
        """
        Takes the bond information from a molecule instances and converts it to an scaling matrix.

        Parameters:
        ---------
        bonds:  list of [int,int]
            list of atoms connected with each other
        scaleparameters: [float,float,float]
            1-2 scaling, 1-3 scaling, 1-4 scaling parameter

        Attributes:
        ---------
        scale: matrix
            scaling matrix how atoms interact with each other

        """

        # Initializing
        scale = np.ones((self._natoms, self._natoms))
        bound12 = np.zeros((self._natoms, self._natoms))
        bound13 = np.zeros((self._natoms, self._natoms))
        bound14 = np.zeros((self._natoms, self._natoms))
        if scaleparameters is None:
            scaleparameters = [0.0, 0.0, 0.8333333333]

        # Building connection matrix
        for bond in self._bonds:
            bound12[bond._atom1][bond._atom2] = 1.0
            bound12[bond._atom2][bond._atom1] = 1.0

        for i in range(len(bound12)):
            b12 = np.where(bound12[i] == 1.0)[0]
            for j in range(len(b12)):
                b12t = np.where(bound12[b12[j]] == 1.0)[0]
                for k in range(len(b12t)):
                    if i != b12t[k]:
                        bound13[b12t[k]][i] = 1.0
                        bound13[i][b12t[k]] = 1.0

        for i in range(self._natoms):
            b13 = np.where(bound13[i] == 1.0)[0]
            for j in range(len(b13)):
                b13t = np.where(bound12[b13[j]] == 1.0)[0]
                for k in range(len(b13t)):
                    if bound12[b13t[k]][i] == 0.0:
                        bound14[b13t[k]][i] = 1.0
                        bound14[i][b13t[k]] = 1.0

        for i in range(self._natoms):
            self.scale[i][i] = 0.0
        # find values in matrix with value 1.0
        b12 = np.array(np.where(bound12 == 1.0)).transpose()
        b13 = np.array(np.where(bound13 == 1.0)).transpose()
        b14 = np.array(np.where(bound14 == 1.0)).transpose()

        # Fill scaling matrix with values
        for i in range(len(b12)):
            self.scale[b12[i][0]][b12[i][1]] = scaleparameters[
                0]  # Value for 1-2 interaction 0 means interactions are neglected
        for i in range(len(b13)):
            self.scale[b13[i][0]][b13[i][1]] = scaleparameters[
                1]  # Value for 1-3 interaction 0 means interactions are neglected
        for i in range(len(b14)):
            self.scale[b14[i][0]][b14[i][1]] = scaleparameters[2]  # Value for the 1-4 scaling

        # Different Scaling parameter for SCF
        if scf_scaleparameters != None:
            self.scale_scf = np.ones((self._natoms, self._natoms))
            for i in range(self._natoms):
                self.scale_scf[i][i] = 0.0
            for i in range(len(b12)):
                self.scale_scf[b12[i][0]][b12[i][1]] = scf_scaleparameters[
                    0]  # Value for 1-2 interaction 0 means interactions are neglected
            for i in range(len(b13)):
                self.scale_scf[b13[i][0]][b13[i][1]] = scf_scaleparameters[
                    1]  # Value for 1-3 interaction 0 means interactions are neglected
            for i in range(len(b14)):
                self.scale_scf[b14[i][0]][b14[i][1]] = scf_scaleparameters[2]  # Value for the 1-4 scaling


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
        self.polESPs = list()
        self._molecule = molecule
        self.q_alpha = self._molecule.q_alpha

        # Initiliaze Electric field vectors
        self.e_field_at_atom = np.zeros((3, self.natoms))

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

    def add_polESP(self, *args, e_field=Q_([0.0, 0.0, 0.0], 'bohr')):
        self.polESPs.append(BCCPolESP(*args, conformer=self, e_field=e_field))

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

    def build_matrix_D(self):
        # 1 dipole vector of length 3 per atom
        # restrains for isotropic polarization
        # Restaint atoms with same polarization parameters
        self.Dlines = (3 * self.natoms + 2 * self.natoms + len(self._molecule.same_polarization_atoms))
        self.D = np.zeros((self.Dlines, self.Dlines))

        for j in range(self.natoms):
            for k in range(self.natoms):
                self.D[k][j] = np.multiply(np.dot(self.dist_x[k], self.dist_x[j]),
                                           self.e_field_at_atom[0][k] * self.e_field_at_atom[0][j])
                self.D[j + self.natoms][k] = self.D[k][j + self.natoms] = np.multiply(
                    np.dot(self.dist_x[k], self.dist_y[j]), self.e_field_at_atom[0][k] * self.e_field_at_atom[1][j])
                self.D[j + 2 * self.natoms][k] = self.D[k][j + 2 * self.natoms] = np.multiply(
                    np.dot(self.dist_x[k], self.dist_z[j]), self.e_field_at_atom[0][k] * self.e_field_at_atom[2][j])
                self.D[k + self.natoms][j + self.natoms] = np.multiply(np.dot(self.dist_y[k], self.dist_y[j]),
                                                                       self.e_field_at_atom[1][k] *
                                                                       self.e_field_at_atom[1][j])
                self.D[j + 2 * self.natoms][k + self.natoms] = self.D[k + self.natoms][
                    j + 2 * self.natoms] = np.multiply(np.dot(self.dist_y[k], self.dist_z[j]),
                                                       self.e_field_at_atom[1][k] * self.e_field_at_atom[2][j])
                self.D[k + 2 * self.natoms][j + 2 * self.natoms] = np.multiply(
                    np.dot(self.dist_z[k], self.dist_z[j]), self.e_field_at_atom[2][k] * self.e_field_at_atom[2][j])
        # Add dipole restraints for equivalent atoms /only works for isotropic suff now
        for polESP in self.polESPs:
            for j in range(self.natoms):
                for k in range(self.natoms):
                    self.D[k][j] += np.multiply(np.dot(self.dist_x[k], self.dist_x[j]),
                                                polESP.e_field_at_atom[0][k] * polESP.e_field_at_atom[0][j])
                    self.D[j + self.natoms][k] = self.D[k][j + self.natoms] = self.D[k][j + self.natoms] + np.multiply(
                        np.dot(self.dist_x[k], self.dist_y[j]),
                        polESP.e_field_at_atom[0][k] * polESP.e_field_at_atom[1][j])
                    self.D[j + 2 * self.natoms][k] = self.D[k][j + 2 * self.natoms] = self.D[k][
                                                                                          j + 2 * self.natoms] + np.multiply(
                        np.dot(self.dist_x[k], self.dist_z[j]),
                        polESP.e_field_at_atom[0][k] * polESP.e_field_at_atom[2][j])
                    self.D[k + self.natoms][j + self.natoms] += np.multiply(np.dot(self.dist_y[k], self.dist_y[j]),
                                                                            polESP.e_field_at_atom[1][k] *
                                                                            polESP.e_field_at_atom[1][j])
                    self.D[j + 2 * self.natoms][k + self.natoms] = self.D[k + self.natoms][j + 2 * self.natoms] = \
                    self.D[k + self.natoms][j + 2 * self.natoms] + np.multiply(np.dot(self.dist_y[k], self.dist_z[j]),
                                                                               polESP.e_field_at_atom[1][k] *
                                                                               polESP.e_field_at_atom[2][j])
                    self.D[k + 2 * self.natoms][j + 2 * self.natoms] += np.multiply(
                        np.dot(self.dist_z[k], self.dist_z[j]),
                        polESP.e_field_at_atom[2][k] * polESP.e_field_at_atom[2][j])

        self.D = self.D / (len(self.polESPs) + 1)

        for j, atoms in enumerate(self._molecule.same_polarization_atoms):
            self.D[5 * self.natoms + j][atoms[0]] = 1
            self.D[5 * self.natoms + j][atoms[1]] = -1
            self.D[atoms[0]][5 * self.natoms + j] = 1
            self.D[atoms[1]][5 * self.natoms + j] = -1
        # Code to keep polarization parameters at their initial value. Implmentation will change
        # elif self.eqdipoles[j][1] < 0:
        #    self.D[self.ndipoles + self.aniso_lines + j][self.eqdipoles[j][0]] = 1
        #    self.D[self.eqdipoles[j][0]][self.ndipoles + self.aniso_lines + j] = 1

        # Add restraints for polarization isotropy
        for j in range(self.natoms):
            self.D[3 * self.natoms + j][j] = self.D[j][3 * self.natoms + j] = 1.0
            self.D[3 * self.natoms + j][j + self.natoms] = self.D[j + self.natoms][3 * self.natoms + j] = -1.0
            self.D[4 * self.natoms + j][j] = self.D[j][4 * self.natoms + j] = 1.0
            self.D[4 * self.natoms + j][j + 2 * self.natoms] = self.D[j + 2 * self.natoms][4 * self.natoms + j] = -1.0

    def build_matrix_X(self):
        """
        Creates Matrix X for the RESP-POl method.

        RESP and Polarization with the large matrix.
        Probably worth changing it to the new model.

        Again the math is shown in the manuscript.
        """
        self.build_matrix_A()
        self.get_electric_field()
        self.build_matrix_D()
        self.B = np.zeros((self.Alines, self.Dlines))
        self.C = np.zeros((self.Dlines, self.Alines))

        # Matrix element B see notes
        for k in range(self.natoms):
            for j in range(self.natoms):
                self.B[k][j] = np.multiply(np.dot(self.dist[k], self.dist_x[j]), self.e_field_at_atom[0][j])  # B1
                self.B[k][self.natoms + j] = np.multiply(np.dot(self.dist[k], self.dist_y[j]),
                                                         self.e_field_at_atom[1][j])  # B2
                self.B[k][2 * self.natoms + j] = np.multiply(np.dot(self.dist[k], self.dist_z[j]),
                                                             self.e_field_at_atom[2][j])  # B3

        for polESP in self.polESPs:
            for k in range(self.natoms):
                for j in range(self.natoms):
                    self.B[k][j] += np.multiply(np.dot(self.dist[k], self.dist_x[j]),
                                                polESP.e_field_at_atom[0][j])  # B1
                    self.B[k][self.natoms + j] += np.multiply(np.dot(self.dist[k], self.dist_y[j]),
                                                              polESP.e_field_at_atom[1][j])  # B2
                    self.B[k][2 * self.natoms + j] += np.multiply(np.dot(self.dist[k], self.dist_z[j]),
                                                                  polESP.e_field_at_atom[2][j])  # B3
        # matrix element C see notes
        # matrix element C
        for j in range(self.natoms):
            for k in range(self.natoms):
                self.C[k][j] = np.multiply(np.dot(self.dist[j], self.dist_x[k]), self.e_field_at_atom[0][k])
                self.C[self.natoms + k][j] = np.multiply(np.dot(self.dist[j], self.dist_y[k]),
                                                         self.e_field_at_atom[1][k])
                self.C[2 * self.natoms + k][j] = np.multiply(np.dot(self.dist[j], self.dist_z[k]),
                                                             self.e_field_at_atom[2][k])

        for polESP in self.polESPs:
            for j in range(self.natoms):
                for k in range(self.natoms):
                    self.C[k][j] += np.multiply(np.dot(self.dist[j], self.dist_x[k]), polESP.e_field_at_atom[0][k])
                    self.C[self.natoms + k][j] += np.multiply(np.dot(self.dist[j], self.dist_y[k]),
                                                              polESP.e_field_at_atom[1][k])
                    self.C[2 * self.natoms + k][j] += np.multiply(np.dot(self.dist[j], self.dist_z[k]),
                                                                  polESP.e_field_at_atom[2][k])
        # Normalize B and C
        self.B = self.B / (len(self.polESPs) + 1)
        self.C = self.C / (len(self.polESPs) + 1)

        # Combine all matrices
        self.X = np.concatenate(
            (np.concatenate((self.A, self.B), axis=1), np.concatenate((self.C, self.D), axis=1)), axis=0)

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
        for polESP in self.polESPs:
            esp_values = polESP.esp_values.to('elementary_charge / angstrom').magnitude
            for k in range(self.natoms):
                self.B[k] += np.dot(esp_values, self.dist[k])
            self.B[self.natoms] = self._molecule._charge

        self.B = self.B / (len(self.polESPs) + 1)
        for k, eq_atoms in enumerate(self._molecule.chemical_eq_atoms):
            if eq_atoms[1] > 0:
                self.B[self.natoms + 1 + k] = 0.0

    def build_vector_C(self):

        self.C = np.zeros(self.Dlines)
        self.esp_values = self.baseESP.esp_values.to('elementary_charge / angstrom').magnitude
        for k in range(self.natoms):
            self.C[k] = np.multiply(np.dot(self.esp_values, self.dist_x[k]), self.e_field_at_atom[0][k])
            self.C[k + self.natoms] = np.multiply(np.dot(self.esp_values, self.dist_y[k]), self.e_field_at_atom[1][k])
            self.C[k + self.natoms * 2] = np.multiply(np.dot(self.esp_values, self.dist_z[k]),
                                                      self.e_field_at_atom[2][k])

        for polESP in self.polESPs:
            esp_values = polESP.esp_values.to('elementary_charge / angstrom').magnitude
            for k in range(self.natoms):
                self.C[k] += np.multiply(np.dot(esp_values, self.dist_x[k]), polESP.e_field_at_atom[0][k])
                self.C[k + self.natoms] += np.multiply(np.dot(esp_values, self.dist_y[k]), polESP.e_field_at_atom[1][k])
                self.C[k + self.natoms * 2] += np.multiply(np.dot(esp_values, self.dist_z[k]),
                                                           polESP.e_field_at_atom[2][k])

        self.C = self.C / (len(self.polESPs) + 1)

        for j, atoms in enumerate(self._molecule.same_polarization_atoms):
            self.C[5 * self.natoms + j] = 0.0

    def build_vector_Y(self):
        """
        Creates the Vector Y for the RESP-Pol Method.
        :return:
        """
        self.build_vector_B()
        self.build_vector_C()
        self.Y = np.concatenate((self.B, self.C))

    def get_electric_field(self):
        self.baseESP.get_electric_field()
        self.e_field_at_atom = self.baseESP.e_field_at_atom
        for polESP in self.polESPs:
            polESP.get_electric_field()

    def optimize_charges_alpha(self):
        self.q_alpha = np.linalg.solve(self.X, self.Y)

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

        del self.diatomic_dist
        del self.diatomic_dist_3
        del self.diatomic_dist_5
        del self.diatomic_dist_x
        del self.diatomic_dist_y
        del self.diatomic_dist_z
        del self.diatomic_distb_x
        del self.diatomic_distb_y
        del self.diatomic_distb_z


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
                self.gridtype = 'respyte'
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
        elif self.gridtype == 'respyte':
            f = open(args[0], 'r')
            lines = f.readlines()
            f.close
            ndata = int(len(lines) / 2) if len(lines) % 2 == 0 else int((len(lines) - 1) / 2)
            grid = np.zeros((ndata, 4))
            for i in range(ndata):
                grid[i] = [float(ele) for ele in lines[2 * i].split()]
            self.positions = Q_(np.array(grid)[:, 0:3], 'angstrom')
            self.esp_values = Q_(np.array(grid)[:, 3], 'elementary_charge / bohr')
        elif self.gridtype == 'psi4':
            for ele in args:
                if "grid.dat" in ele:
                    gridfile = ele
                elif 'esp.dat' in ele:
                    espfile = ele
            np.loadtxt(espfile)
            self.positions = Q_(np.loadtxt(gridfile), 'angstrom')
            self.esp_values = Q_(np.loadtxt(espfile), 'elementary_charge / bohr')

    def get_electric_field(self, ):
        """
        Calculates the electric field at every atomic positions.
        :return:
        """
        e_field_at_atom_old = copy.copy(self.e_field_at_atom)
        dipole = self._conformer.q_alpha[self._conformer.Alines:self._conformer.Alines + self._conformer.natoms]
        dipole[np.where(dipole == 0.0)] += 10E-10

        # Load permanent charges for BCC method
        # For all other methods this is set to 0.0 STILL HAVE to implment
        try:
            self._conformer.q_am1
        except Exception:
            log.warning('I do not have AM1-type charges')
            self._conformer.q_am1 = np.zeros(self._conformer.natoms)

        for j in range(self._conformer.natoms):
            self.e_field_at_atom[0][j] = np.dot(
                np.multiply(self._conformer.q_alpha[:self._conformer.natoms], self._conformer._molecule.scale[j]),
                self._conformer.diatomic_dist_x[j]) + np.dot(
                np.multiply(self._conformer.q_am1[:self._conformer.natoms], self._conformer._molecule.scale[j]),
                self._conformer.diatomic_dist_x[j]) + self._ext_e_field[0].to(
                'elementary_charge / angstrom / angstrom').magnitude
            self.e_field_at_atom[1][j] = np.dot(
                np.multiply(self._conformer.q_alpha[:self._conformer.natoms], self._conformer._molecule.scale[j]),
                self._conformer.diatomic_dist_y[j]) + np.dot(
                np.multiply(self._conformer.q_am1[:self._conformer.natoms], self._conformer._molecule.scale[j]),
                self._conformer.diatomic_dist_y[j]) + self._ext_e_field[1].to(
                'elementary_charge / angstrom / angstrom').magnitude
            self.e_field_at_atom[2][j] = np.dot(
                np.multiply(self._conformer.q_alpha[:self._conformer.natoms], self._conformer._molecule.scale[j]),
                self._conformer.diatomic_dist_z[j]) + np.dot(
                np.multiply(self._conformer.q_am1[:self._conformer.natoms], self._conformer._molecule.scale[j]),
                self._conformer.diatomic_dist_z[j]) + self._ext_e_field[2].to(
                'elementary_charge / angstrom / angstrom').magnitude

        self.e = self.e_field_at_atom.flatten()

        if self._conformer._molecule._SCF and self._conformer._molecule.step > 0:
            if not hasattr(self, 'Bdip') or self._conformer._molecule._thole:
                if self._conformer._molecule._thole:
                    # thole_param=1.368711/BOHR**2
                    thole_param = 0.390
                    dipole_tmp = np.where(dipole < 0.0, -dipole, dipole)
                    thole_v = np.multiply(self.adist, np.float_power(
                        np.multiply(dipole_tmp[:self._conformer.natoms, None], dipole_tmp[:self._conformer.natoms]),
                        1. / 6))
                    di = np.diag_indices(self._conformer.natoms)
                    thole_v[di] = 1.0
                    thole_v = 1. / thole_v
                    thole_v[di] = 0.0

                    # Exponential thole
                    thole_fe = np.ones((self._conformer.natoms, self._conformer.natoms))
                    thole_ft = np.ones((self._conformer.natoms, self._conformer.natoms))
                    thole_fe -= np.exp(np.multiply(thole_param, np.power(-thole_v, 3)))
                    thole_ft -= np.multiply(np.multiply(thole_param, np.power(thole_v, 3)) + 1.,
                                            np.exp(np.multiply(thole_param, np.power(-thole_v, 3))))
                    # 1.5 was found in the OpenMM code. Not sure whuy it is there

                    # In original thole these lines should not be here
                    # thole_ft = np.multiply(thole_ft, self._conformer.scale)
                    # thole_fe = np.multiply(thole_fe, self._conformer.scale)
                    # Linear thole
                    # thole_fe=np.zeros((self._conformer.natoms,self._conformer.natoms))
                    # thole_fe=np.zeros((self._conformer.natoms,self._conformer.natoms))
                    # thole_fe=np.where(thole_v>1.0,1.0,4*np.power(thole_v,3)-3*np.power(thole_v,4))
                    # thole_ft=np.where(thole_v>1.0,1.0,np.power(thole_v,4))
                else:
                    try:
                        thole_ft = self._conformer.scale_scf
                        thole_fe = self._conformer.scale_scf
                    except Exception:

                        thole_ft = self._conformer.scale
                        thole_fe = self._conformer.scale
                    else:
                        print('Using different set of scaling for SCF interactions')
                        log.info('Using different set of scaling for SCF interactions')
                Bdip11 = np.add(np.multiply(thole_fe, self._conformer.diatomic_dist_3), np.multiply(thole_ft,
                                                                                                    -3 * np.multiply(
                                                                                                        np.multiply(
                                                                                                            self._conformer.diatomic_distb_x,
                                                                                                            self._conformer.diatomic_distb_x),
                                                                                                        self._conformer.diatomic_dist_5)))
                Bdip22 = np.add(np.multiply(thole_fe, self._conformer.diatomic_dist_3), np.multiply(thole_ft,
                                                                                                    -3 * np.multiply(
                                                                                                        np.multiply(
                                                                                                            self._conformer.diatomic_distb_y,
                                                                                                            self._conformer.diatomic_distb_y),
                                                                                                        self._conformer.diatomic_dist_5)))
                Bdip33 = np.add(np.multiply(thole_fe, self._conformer.diatomic_dist_3), np.multiply(thole_ft,
                                                                                                    -3 * np.multiply(
                                                                                                        np.multiply(
                                                                                                            self._conformer.diatomic_distb_z,
                                                                                                            self._conformer.diatomic_distb_z),
                                                                                                        self._conformer.diatomic_dist_5)))
                Bdip12 = np.multiply(thole_ft,
                                     -3 * np.multiply(np.multiply(self._conformer.diatomic_distb_x,
                                                                  self._conformer.diatomic_distb_y),
                                                      self._conformer.diatomic_dist_5))
                Bdip13 = np.multiply(thole_ft,
                                     -3 * np.multiply(np.multiply(self._conformer.diatomic_distb_x,
                                                                  self._conformer.diatomic_distb_z),
                                                      self._conformer.diatomic_dist_5))
                Bdip23 = np.multiply(thole_ft,
                                     -3 * np.multiply(np.multiply(self._conformer.diatomic_distb_y,
                                                                  self._conformer.diatomic_distb_z),
                                                      self._conformer.diatomic_dist_5))
                Bdip = np.concatenate((np.concatenate((Bdip11, Bdip12, Bdip13), axis=1),
                                       np.concatenate((Bdip12, Bdip22, Bdip23), axis=1),
                                       np.concatenate((Bdip13, Bdip23, Bdip33), axis=1)),
                                      axis=0)

            for j in range(self._conformer.natoms):
                for k in range(3):
                    for l in range(3):
                        Bdip[k * self._conformer.natoms + j][l * self._conformer.natoms + j] = 0.0

            for j in range(3 * self._conformer.natoms):
                Bdip[j][j] = 1. / dipole[j]
            dipole_scf = np.linalg.solve(Bdip, self.e)
            self.e = np.divide(dipole_scf, dipole[:self.ndipoles])
        self.e_field_at_atom[0] = 1.0 * self.e[:self._conformer.natoms]
        self.e_field_at_atom[1] = 1.0 * self.e[self._conformer.natoms:2 * self._conformer.natoms]
        self.e_field_at_atom[2] = 1.0 * self.e[2 * self._conformer.natoms:3 * self._conformer.natoms]

        # WARNING Have to implement this add a different position
        # self.step += 1


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
        self._conformer = conformer

        # External e-field is 0 in all directions
        vector = Q_([0, 0, 0], 'elementary_charge / bohr / bohr')
        self.set_ext_e_field(vector)

        self.load_grid(*args)

        self.e_field_at_atom = np.zeros((3, self._conformer.natoms))


# =============================================================================================
# BCCPolESP
# =============================================================================================

class BCCPolESP(ESPGRID):
    """

    """

    def __init__(self, *args, conformer=None, e_field=Q_([0.0, 0.0, 0.0], 'elementary_charge / bohr / bohr')):
        # Decide if we have a Gaussian grid or a psi4 grid
        self.gridtype = None
        self.natoms = -1
        self.atoms = []
        self.atom_positions = []
        self.define_grid(*args)
        self.esp_values = None
        self.positions = None
        self._conformer = conformer

        # Set external e-field
        self.set_ext_e_field(e_field)

        self.load_grid(*args)

        self.e_field_at_atom = np.zeros((3, self._conformer.natoms))


if __name__ == '__main__':
    """
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/tmp/phenol/conf0/mp2_0.mol2')
    test = TrainingSet(scf_scaleparameters=[0.0, 0.0, 0.5])
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    print(test.molecules[0].same_polarization_atoms)
    print(test.molecules[0].scale)
    print(test.molecules[0].scale_scf)
    espfile = '/home/michael/resppol/resppol/tmp/phenol/conf0/molecule0.gesp'
    test.molecules[0].conformers[0].add_baseESP(espfile)
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/tmp/butanol/conf0/mp2_0.mol2')
    test.add_molecule(datei)
    test.molecules[1].add_conformer_from_mol2(datei)
    espfile = '/home/michael/resppol/resppol/tmp/butanol/conf0/molecule0.gesp'
    test.molecules[1].conformers[0].add_baseESP(espfile)
    test.optimize_charges()
    test.molecules[0].conformers[0].build_matrix_X()
    for molecule in test.molecules:
        print(molecule.q)

    print('FINISH')
    """
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test2.mol2')
    test = TrainingSet()
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test1.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test1_Z+.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=[0, 0, 1])
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test1_Z-.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=[0, 0, -1])
    test.optimize_charges()
    print(test.q)
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test2.mol2')
    test = TrainingSet()
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3_Z+.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, 1.0], 'elementary_charge / bohr / bohr'))
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3_Z-.gesp')
    # test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, -1.0], 'elementary_charge / bohr / bohr') )
    test.molecules[0].conformers[0].build_matrix_X()
    test.molecules[0].conformers[0].build_vector_Y()
    test.molecules[0].conformers[0].optimize_charges_alpha()
    print(test.molecules[0].conformers[0].q_alpha)
    test.optimize_charges()
    print(test.q)
