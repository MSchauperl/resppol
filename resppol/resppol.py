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
import sys
try:
	from openeye import oechem
except:
	log.warning('Could not import openeye')
try:
	import openforcefield.topology as openff
	from openforcefield.typing.engines.smirnoff import ForceField
except:
	log.warning("Could not import openforcefield")
from scipy.spatial import distance
import os
# Initialize units
from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.define('bohr = 0.52917721067 * angstrom')

##### LOGGER

FORMAT = '%(asctime)s  - %(levelname)s - %(message)s'
log.basicConfig(format=FORMAT, level=log.INFO)
log.getLogger().addHandler(log.StreamHandler(sys.stdout))

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

    TODO:
     Include rdkit support for this function
    """

    qmol = oechem.OEQMol()

    # build OEQMol from OEGRAPHMOLECULE
    oechem.OEBuildMDLQueryExpressions(qmol, mol1)
    ss2 = oechem.OESubSearch(qmol)
    oechem.OEPrepareSearch(mol1, ss2)
    qmol = oechem.OEQMol()
    oechem.OEBuildMDLQueryExpressions(qmol, mol1)
    ss2 = oechem.OESubSearch(qmol)
    oechem.OEPrepareSearch(mol1, ss2)
    # it is not an error that i actually write this twice.
    # For phenol it seems to be necessary for an undefined reason.

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


def read_charges_from_g09(g09_output):
    pass




# =============================================================================================
# TrainingSet
# =============================================================================================

class TrainingSet():
    """
    The training set class is the top level class of the resspol program.

    It consist of multiple molecule instances and combines the optimization matrices and vectors across multiple molecules.
    """

    def __init__(self, mode='q', scaleparameters=None, scf_scaleparameters=None, SCF=False, thole=False, FF='resppol/data/test_data/BCCPOL.offxml'):
        """
        Initilize the class and sets the following parameters:

        :param mode:
        :param scaleparameters:
        :param scf_scaleparameters:
        :param SCF:
        :param thole:
        """
        #Reading input and setting default values
        # ToDo add logging messages to the definition
        self.molecules = list()
        # Initialize matrix B with no values
        self.B = np.zeros(0)
        # Initialize matrix A with 2 dimensions and no values
        self.A = np.zeros((0, 0))
        # Initialize vector q with charge 0.0
        #ToDo: Check if this makes sense
        self.q = 0.0
        # Mode defines if we are optimizing charges, BCCs or Polarizabilites
        # Simulatenous optimization of charges or BCCs with Polarizabilities is possible
        self.mode = mode
        # Set SCF scaling parameter for all molecules
        # ToDo maybe i want to add a default value here
        self.scf_scaleparameters = scf_scaleparameters
        # Scaling parameters for charge charge interaction and charge dipole interaction
        # ToDo maybe i want to add a default value here
        self.scaleparameters = scaleparameters
        log.info('Charge Dipole interactions are scaled using the following parameters {}. '.format(self.scaleparameters))
        log.info('If non are specified the default is 0.0, 0.0, 0.8333. '
                 'This means that 1-2 and 1-3 interactions are neglected and 1-4 are scaled by a factor 0.8333. ')
        # Define if we use SCF or not default is the direct approach
        self._SCF = SCF
        # If thole scaling is applied
        self._thole = thole
        if self._SCF == True and self._thole == False:
            log.info('''Dipoles are calculated using the self-consistent field approach
                        The interactions are scaled using the following scaling factors: {}'''.format(self.scf_scaleparameters))
        if self._SCF == True and self._thole == True:
            log.info('''Dipoles are calculated using the self-consistent field approach
                    The interactions are scaled using a thole distance based scheme''')
        # ToDo Delete due to repetion. Check if always _mode is used before doing that thought.
        self._mode = mode
        log.info('Running in Mode: {}'.format(self._mode))
        # counts the number of optimization steps required
        self.step = 0
        # Counts the number of the charge /BCC matrix part
        self.number_of_lines_in_X = 0
        # Not sure
        # ToDo Check what this is good for
        self.X_BCC = 0.0
        self.Y_BCC = 0.0
        # Set the force-field file which is used for all molecules
        self._FF = FF
        # This block describes how to label the atoms and bonds using a offxml file

        # Read in the forcefield using the ForceField class of the openforcefield toolkit
        forcefield = ForceField(os.path.join(ROOT_DIR_PATH, FF))
        log.info('The forcefield used for this run is {} '.format(os.path.join(ROOT_DIR_PATH, FF)))
        # Number of different polarization types
        self._nalpha = len(forcefield.get_parameter_handler('vdW').parameters)

        #Number of different BCCs definied in the FF
        self._nbccs = len(forcefield.get_parameter_handler('Bonds').parameters)
        # Assigns a number for every BCC
        self._bccs ={}
        for i,element in enumerate(forcefield.get_parameter_handler('Bonds').parameters):
            self._bccs[element.id] = i

        # The same for polarizabilities
        self._alpha = {}
        for i,element in enumerate(forcefield.get_parameter_handler('vdW').parameters):
            self._alpha[element.id] = i

        self.bccs_old = np.zeros(self._nbccs)
        self.alphas_old = np.zeros(self._nalpha)

    # This function is not done yet
    # Maybe replace with json format reader and writer
    def load_from_file(self,txtfile):
        """
        Allows to build a TrainingSet instance from an text file.
        File format as followed:

        :return:
        """
        f = open(txtfile)
        lines = f.readlines()
        f.close()
        for i, line in enumerate(lines):
            mol2file = line.split()[1]
            # noinspection PyTypeChecker
            self.add_molecule(Molecule(mol2file))

    def add_molecule(self, datei,am1_mol2_file=None):
        """
        Adds a molecule to the TrainingSet object.
        :param datei: Mol2 file of a molecule
        :return:
        """
        #ToDo Check matrix composition

        # Adds the molecule
        self.molecules.append(Molecule(datei, position=self.number_of_lines_in_X, trainingset=self,id=len(self.molecules),am1_mol2_file=am1_mol2_file))
        log.info('Added molecule number {} from file {}.'.format(len(self.molecules)-1,datei))
        # Defines the position of the molecule in the overall matrix.
        #   A   0       CT          =   B
        #   0    A(new) C(new)T     =   B(new)
        #   C    C(new) P           =   B(Pol)
        self.number_of_lines_in_X += self.molecules[-1]._lines_in_X








    def build_matrix_A(self):
        """
        Builds the matrix A of the underlying molecules and combines them.

        This function is only used for charge optimization with RESP
        """
        #
        for i,molecule in enumerate(self.molecules):
            # Building matrix A on the molecule level
            molecule.build_matrix_A()
            log.info('Build matrix A (only charge optimization) for molecule {}'.format(i))
            #Defines the interaction matrix 0 for charge of different molecules
            X12 = np.zeros((len(self.A), len(molecule.A)))
            # Combines the matrix A of the TrainingSet and the matrix A of the molecule
            self.A = np.concatenate(
                (np.concatenate((self.A, X12), axis=1), np.concatenate((X12.transpose(), molecule.A), axis=1)), axis=0)

    def build_vector_B(self):
        for i,molecule in enumerate(self.molecules):
            # Building the vector B of the molecule.
            molecule.build_vector_B()
            log.info('Build vector B (only charge optimization) for molecule {}'.format(i))
            # Adds the vector B of the molecuel to the existing vector B
            self.B = np.concatenate((self.B, molecule.B))

    def build_matrix_X(self):
        self.X = np.zeros((0,0))
        """
        Builds the matrix X of the underlying molecules and combines them.

        This function is only used for charge optimizatoin RESP
        """
        for i,molecule in enumerate(self.molecules):
            molecule.build_matrix_X()
            log.info('Build matrix X for molecule {}'.format(i))
            X12 = np.zeros((len(self.X), len(molecule.X)))
            self.X = np.concatenate(
                (np.concatenate((self.X, X12), axis=1), np.concatenate((X12.transpose(), molecule.X), axis=1)), axis=0)

        X12 = np.zeros((len(self.X),len(self.intramolecular_polarization_rst)))
        X22 =  np.zeros((len(self.intramolecular_polarization_rst),len(self.intramolecular_polarization_rst)))

        self.X = np.concatenate((np.concatenate((self.X, X12), axis=1), np.concatenate((X12.transpose(),X22), axis =1)), axis=0)
        for i,atoms in enumerate(self.intramolecular_polarization_rst):
            self.X[self.number_of_lines_in_X + i][atoms[0]] = self.X[atoms[0]][self.number_of_lines_in_X + i] = 1
            self.X[self.number_of_lines_in_X + i][atoms[1]] = self.X[atoms[1]][self.number_of_lines_in_X + i] = -1


    def build_vector_Y(self):
        self.Y = np.zeros(0)
        for i,molecule in enumerate(self.molecules):
            molecule.build_vector_Y()
            log.info('Build vector Y for molecule {}'.format(i))
            self.Y = np.concatenate((self.Y, molecule.Y))

        Y2 =  np.zeros(len(self.intramolecular_polarization_rst))
        self.Y = np.concatenate((self.Y, Y2))
    # def get_intramolecular_charge_rst()

    def build_matrix_X_BCC(self):
        """
        Builds the matrix X of the underlying molecules and combines them.

        This function is only used for optimization of BCCs
        """

        for i,molecule in enumerate(self.molecules):
            molecule.build_matrix_X_BCC()
            log.info('Build matrix X BCC for molecule {}'.format(i))
            self.X_BCC += molecule.X


    def build_vector_Y_BCC(self):
        """
        Builds the matrix X of the underlying molecules and combines them.

        This function is only used for optimization of BCCs
        """

        for i,molecule in enumerate(self.molecules):
            molecule.build_vector_Y_BCC()
            log.info('Build vector Y BCC for molecule {}'.format(i))
            self.Y_BCC += molecule.Y



    @property
    def intramolecular_polarization_rst(self):
        # Defines a list of atompairs which are restrained to be equal between different molecules
        # ToDo Check I do not include intramolecular polarization restraints here,just intermolecular ones
        # Doublecheck this behviour
        # Uses Lagrange formalism
        intramolecular_polarization_rst = []
        # First occurance of an atom with a given polarization type
        first_occurrence_of_parameter = {}
        for molecule in self.molecules:
            first_occurrence_of_parameter_in_molecule = {}
            for atom in molecule._atoms:
                if atom._parameter_id not in first_occurrence_of_parameter.keys():
                    first_occurrence_of_parameter[atom._parameter_id] = molecule._position_in_A + atom._id + molecule._lines_in_A
                elif atom._parameter_id not in first_occurrence_of_parameter_in_molecule.keys():
                    intramolecular_polarization_rst.append(
                        [first_occurrence_of_parameter[atom._parameter_id], molecule._position_in_A + atom._id + molecule._lines_in_A])
        log.info('Apply the following intramolecular polarization restaraints {}'.format(intramolecular_polarization_rst))
        return intramolecular_polarization_rst

    def optimize_charges_alpha(self,criteria = 10E-5):
        converged = False
        self.counter = 0
        while converged == False and self.counter<100:
        #while self.counter < 10:
            log.warning('Optimization Step {}'.format(self.counter))
            self.optimize_charges_alpha_step()
            converged = True
            self.counter += 1
            for num,molecule in enumerate(self.molecules):
                molecule.step +=1
                if not all(abs(molecule.q - molecule.q_old) <criteria) or not all(abs(molecule.alpha - molecule.alpha_old) <criteria) :
                    converged = False
                    log.info('Optimization step {}: Molekule {}: Charges'.format(self.counter,num))
                    log.info(molecule.q)
                    log.info('Optimization step {}: Molekule {}: Dipoles'.format(self.counter,num))
                    log.info(molecule.alpha)
                molecule.q_old = molecule.q
                molecule.alpha_old = molecule.alpha
                log.debug(molecule.q)
                log.debug(molecule.alpha)
            if self.counter ==99:
                log.warning('Optimization did not converge. Stopped after 100 cycles.')



    def optimize_charges_alpha_step(self):
        self.build_matrix_X()
        self.build_vector_Y()
        #log.info(self.X)
        self.q_alpha = np.linalg.solve(self.X, self.Y)
        # Update the charges of the molecules below
        q_alpha_tmp = self.q_alpha
        for molecule in self.molecules:
            molecule.q_alpha = q_alpha_tmp[:len(molecule.X)]
            q_alpha_tmp = q_alpha_tmp[len(molecule.X):]
            molecule.update_q_alpha()
        log.info('Updating charges and polarization parameters for all molecules')


    def optimize_bcc_alpha(self,criteria = 10E-3):
        converged = False
        self.counter =0
        # Remove potential from AM1 charges
        self.substract_am1_potential()
        while converged == False and self.counter <100:
            log.warning('Optimization Step {}'.format(self.counter))
            self.optimize_bcc_alpha_step()
            self.counter+=1
            for num,molecule in enumerate(self.molecules):
                molecule.step +=1
            log.info('Optimization step {}: BCCs'.format(self.counter))
            log.info(self.bccs)
            log.info('Optimization step {}: Dipoles'.format(self.counter))
            log.info(self.alphas)
            if all(abs(self.bccs - self.bccs_old) < criteria) and all(abs(self.alphas - self.alphas_old)< criteria):
                converged = True
            else:
                self.alphas_old = self.alphas
                self.bccs_old = self.bccs
            if self.counter ==99:
                log.warning('Optimization did not converge. Stopped after 100 cycles.')


    def optimize_bcc_alpha_step(self):
        self.build_matrix_X_BCC()
        self.build_vector_Y_BCC()

        # Check if bccs or alphas are not in the training set and set them to zero
        for i,row in enumerate(self.X_BCC):
            if all(row == 0.0):
                self.X_BCC[i][i]=1

        self.bcc_alpha = np.linalg.solve(self.X_BCC, self.Y_BCC)
        # Update the charges of the molecules below
        self.bccs = self.bcc_alpha[:self._nbccs]
        self.alphas = self.bcc_alpha[self._nbccs:]
        for molecule in self.molecules:
            molecule.update_bcc(self.bccs, self.alphas)
        log.info('Updating Bccs for all molecules')



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


    def substract_am1_potential(self):
        for molecule in self.molecules:
            molecule.substract_am1_potential()
# =============================================================================================
# Molecule
# =============================================================================================


class Molecule:
    """
        This class loads in a mol2 file and expose all relevant information. It combines the functionality of
        an openeye molecule with the functionality of the OFF molecule.

        The OE part is only needed to determine
        the chemical equivalent atoms. When this feature is implemented in OFF toolkit the OE molecule is not
        necessary anymore

        :param datei: mol2 file

        :return:
        """

    def __init__(self, datei, position=0, trainingset=None,id=None,am1_mol2_file=None):

        # Get Molecle name from filename
        self.id = id
        self.B = 0.0
        self.A = 0.0
        self.X = 0.0
        self.Y = 0.0
        self._name = datei.split('/')[-1].strip(".mol2")
        self._trainingset = trainingset

        # Number of optimization steps
        self.step = 0

        # Postion of this molecule in optimization matrix A
        self._position_in_A = position

        # Copy (scf) scaleparameters from the trainingset definition
        if trainingset is None:
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
        # Create OE molecule
        oechem.OEReadMol2File(ifs, self.oemol)

        # Check if molecule is a 3 dimensional object
        if self.oemol.GetDimension() != 3:
            log.error('Molecule either not found or does not have a 3D structure')
            raise Exception('Molecule either not found or does not have a 3D structure')

        # Check for strange atoms
        AtomicNumbers = []
        for atom in self.oemol.GetAtoms():
            AtomicNumbers.append(atom.GetAtomicNum())

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
        # Not sure if that is the most sensible thing to do. Maybe revisit this part at a later stage and define differently
        if trainingset is None:
            self._trainingset = TrainingSet()

        forcefield = ForceField(os.path.join(ROOT_DIR_PATH, self._trainingset._FF))

        # Run the parameter labeling
        molecule_parameter_list = forcefield.label_molecules(self.offtop)

        # set molecule charge
        # self._charge=openff.Molecule.total_charge
        # Change this back at a later point. To make charged molecules possible
        self._charge = 0

        # Initialize the bonds, atoms
        self._bonds = list()
        self._atoms = list()

        # Define all BCC bonds
        # ToDo write test for this function
        for i, properties in enumerate(molecule_parameter_list[0]['Bonds'].items()):
            atom_indices, parameter = properties
            self.add_bond(i, atom_indices, parameter.id)

        self._nbonds = len(self._bonds)

        # Define all atomtypes for polarization
        # ToDo write test for this function
        for i, properties in enumerate(molecule_parameter_list[0]['vdW'].items()):
            atom_index, parameter = properties
            self.add_atom(i, atom_index[0], parameter.id, atomic_number = AtomicNumbers[atom_index[0]])

        self._natoms = len(self._atoms)

        # Initialize and fill scaling matrix
        self.scale = np.ones((self._natoms, self._natoms))
        self.scale_scf = np.ones((self._natoms, self._natoms))
        self.scaling(scf_scaleparameters=self.scf_scaleparameters, scaleparameters=self.scaleparameters)

        # Defines the bond matrix T:
        self.create_BCCmatrix_T()
        # Defines the polarization type matrix R
        self.create_POLmatrix_R()

        # Initialize conformers
        self.conformers = list()

        # Load in charges from mol2 file:
        self.q_am1 = None
        if am1_mol2_file != None:
            self.q_am1 = []
            self.am1mol = oechem.OEMol()
            # Open Input File Stream
            ifs = oechem.oemolistream(am1_mol2_file)

            # Read from IFS to molecule
            # Create OE molecule
            oechem.OEReadMol2File(ifs, self.am1mol)
            for atom in self.am1mol.GetAtoms():
                self.q_am1.append(atom.GetPartialCharge())

        # Number of lines for matrix X
        if self._mode == 'q':
            self._lines_in_X = self._natoms + len(self.chemical_eq_atoms) + 1
            self.q_old = np.zeros(self._natoms)
        if self._mode == 'q_alpha':
            self._lines_in_X = self._natoms + len(
                self.chemical_eq_atoms) + 1 + 3 * self._natoms + 2 * self._natoms
            self.q_old = np.zeros(self._natoms)
            self.alpha_old = np.zeros(3*self._natoms)
        self._lines_in_A =  self._natoms + len(self.chemical_eq_atoms) + 1
        # Initiliaxe charges
        # Maybe I want to change that later to read charges from the mol2 file
        # or at least give the option to do so
        self.q_alpha = np.zeros(self._lines_in_X)

    def add_bond(self, index, atom_indices, parameter_id):
        """
        Adds a bond object ot the molecule

        :param index:
        :param atom_indices:
        :param parameter_id:
        :return:
        """
        atom_index1 = atom_indices[0]
        atom_index2 = atom_indices[1]
        self._bonds.append(Bond(index, atom_index1, atom_index2, parameter_id))
        log.info('Add bond with type {} between atom {} and atom {}'.format(parameter_id,atom_index1,atom_index2))

    def add_atom(self, index, atom_index, parameter_id, atomic_number=0):
        """
        Adds a atom object to the molecule

        :param index:
        :param atom_index:
        :param parameter_id:
        :return:
        """
        self._atoms.append(Atom(index, atom_index, parameter_id, atomic_number= atomic_number))
        log.info('Add atom type {} on atom {}'.format(parameter_id, atom_index))

    def add_conformer_from_mol2(self, mol2file):
        """
        Adds a conformer from a mol2 file
        Automatically checks if the conformer corresponds to the molecule

        :param mol2file:
        :return:
        """
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
            log.info('Added conformation {} to molecule {}'.format(len(self.conformers)-1,self.id))
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
                array_of_same_pol_atoms.append([first_occurrence[atom._parameter_id], atom._id ])
        log.info('The following atoms have the same polariztion type: {}'.format(array_of_same_pol_atoms))
        return (array_of_same_pol_atoms)


    def create_BCCmatrix_T(self):
        """
        Creates the bond matrix T for a molecule.

        Parameters:
        ----------
            bondtyps: list of [int,int]
                Bonds in the set between different BCC groups

        Attribute:
        ----------
            bondmatrix T: 2 dim array
                1 dim atom
                2 dim BCC
        See also AM1-BCC paper:
        https://onlinelibrary.wiley.com/doi/epdf/10.1002/%28SICI%291096-987X%2820000130%2921%3A2%3C132%3A%3AAID-JCC5%3E3.0.CO%3B2-P
        """
        bondsexcluded = []
        self.T = np.zeros((self._natoms, self._trainingset._nbccs))
        for bond in self._bonds:
            if bond._parameter_id not in bondsexcluded:
                    self.T[bond._atom1][self._trainingset._bccs[bond._parameter_id]] += 1
                    self.T[bond._atom2][self._trainingset._bccs[bond._parameter_id]] -= 1

    def create_POLmatrix_R(self):
        """
        Create a polarization matrix. Defines which atom has which polarizability parameter.

        Parameters:
        ----------
        groups: dictionary atomtyp -> int
            Stores the polarization group  of each atom

        Defines which atom belongs to the which pol group.
        Takes the list of atomtypes and assign the corresponding group value.
        Here I am using a slightly different approach in contrast to the atomtyps.
        as the number of different pol types is maximal the number of atom types. No pre-screening for the involved
        atom-types is needed
        """
        self.R = np.zeros((self._natoms, self._trainingset._nalpha))
        for atom in self._atoms:
            self.R[atom._id][self._trainingset._alpha[atom._parameter_id]] += 1

    def update_bcc(self, bccs, alphas):
        """
        Converts the optimized bccs and alphas to a qd object.

        :param mol2: molecule class object
        :return:
        """
        for i in range(self._natoms):
            self.q_alpha[i] = self.q_am1[i] + np.dot(self.T[i], bccs)
        self.alpha = [np.dot(self.R[i], alphas) for i in range(len(self.R))]
        self.q_alpha[self._lines_in_A:self._lines_in_A + 3* len(self.alpha)] = np.concatenate(
            (np.concatenate((self.alpha, self.alpha)), self.alpha))
        self.q = self.q_alpha[:self._natoms]
        for conformer in self.conformers:
            conformer.q_alpha = self.q_alpha


    def build_matrix_A(self):
        for conformer in self.conformers:
            conformer.build_matrix_A()
            self.A += conformer.A

    def build_vector_B(self):
        for conformer in self.conformers:
            conformer.build_vector_B()
            self.B += conformer.B

    def build_matrix_X(self):
        self.X=0.0
        for conformer in self.conformers:
            conformer.build_matrix_X()
            self.X += conformer.X

    def build_matrix_X_BCC(self):
        for conformer in self.conformers:
            conformer.build_matrix_X_BCC()
            self.X += conformer.X

    def build_vector_Y(self):
        self.Y=0.0
        for conformer in self.conformers:
            conformer.build_vector_Y()
            self.Y += conformer.Y

    def build_vector_Y_BCC(self):
        for conformer in self.conformers:
            conformer.build_vector_Y_BCC()
            self.Y += conformer.Y

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

    def update_q_alpha(self):
        self.q = self.q_alpha[:self._natoms]
        self.alpha = self.q_alpha[self._natoms+1+len(self.chemical_eq_atoms):4*self._natoms+1+len(self.chemical_eq_atoms):]
        for conformer in self.conformers:
            conformer.q_alpha = self.q_alpha

    def substract_am1_potential(self):
        if self.q_am1 is not None:
            for conformer in self.conformers:
                conformer.substract_am1_potential()
        else:
            self.q_am1 = np.zeros(self._natoms)
            log.warning('NO AM1 charges are definied')


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

    def __init__(self, index, atom_index, parameter_id, atomic_number = 0):
        self._id = index
        self._atom = atom_index
        self._parameter_id = parameter_id
        self._atomic_number = atomic_number


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
        # Read in all atomic positions and convert them to Angstrom
        # noinspection PyProtectedMember
        self.atom_positions = Q_(np.array(conf.conformers[0]._value), 'angstrom')
        self.atom_positions_angstrom = self.atom_positions.to('angstrom').magnitude
        self.natoms = len(self.atom_positions.magnitude)
        self.baseESP = None
        self.polESPs = list()
        self._molecule = molecule
        self.q_alpha = self._molecule.q_alpha
        self._lines_in_A = self._molecule._lines_in_A
        self.q_am1 = self._molecule.q_am1

        # Initiliaze Electric field vectors
        self.e_field_at_atom = np.zeros((3, self.natoms))

    def get_grid_coord(self):
        self.grid_coord_angstrom = self.baseESP.positions.to('angstrom').magnitude
        self.npoints = len(self.baseESP.positions.magnitude)

    def add_baseESP(self, *args, ):
        """
        Adds the unpolarized molecule to this conformation.

        :param args:  ESPF file
        1.) GESP file form g09
        2.) grid.dat file and esp.dat file generated via respyte and psi4

        :return:
        """
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
        """
        Adds the unpolarized molecule to this conformation.

        :param args:  ESPF file
        1.) GESP file form g09
        2.) grid.dat file and esp.dat file generated via respyte and psi4
        :param:e_field: Pint formatted electrif field
        e.g e_field=Q_([0.0, 0.0, 0.0], 'bohr'))

        :return:
        """
        self.polESPs.append(BCCPolESP(*args, conformer=self, e_field=e_field))

    # Build the matrix A for the charge optimization
    def build_matrix_A(self):

        """
        Fast method for only optimizing charges.

        :return:
        """
        # Determine size of matrix for this molecule
        # every atom is one line
        # one line to restrain the overall charge of the molecule
        # one line for every pair of equivalent atoms
        self.get_distances()
        self.A = np.zeros((self._lines_in_A, self._lines_in_A))
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
        """
        Method for building the polarization matrix D. Only used for fitting polarizations withotut charges or BCCs.

        :return:
        """
        # 1 dipole vector of length 3 per atom
        # restrains for isotropic polarization
        # Restaint atoms with same polarization parameters
        self.Dlines = (3 * self.natoms + 2 * self.natoms) #+ len(self._molecule.same_polarization_atoms))
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

        #for j, atoms in enumerate(self._molecule.same_polarization_atoms):
        #    self.D[5 * self.natoms + j][atoms[0]] = 1
        #    self.D[5 * self.natoms + j][atoms[1]] = -1
        #    self.D[atoms[0]][5 * self.natoms + j] = 1
        #    self.D[atoms[1]][5 * self.natoms + j] = -1
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
        self.B = np.zeros((self._lines_in_A, self.Dlines))
        self.C = np.zeros((self.Dlines, self._lines_in_A))

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


    # Combines BCC and polarizabilities
    def build_matrix_X_BCC(self):
        """
        Creates the Matrix A for the BCC-POL approach:

        :param mol2: molecule class object

        :return:

        The math and the optimization method is explained in the current paper draft.
        """
        self.get_distances()
        nbcc = self._molecule._trainingset._nbccs
        nalpha = self._molecule._trainingset._nalpha
        T = self._molecule.T
        R = self._molecule.R
        self.get_electric_field()
        # BCC part
        self.A = np.zeros((nbcc, nbcc))
        if self._molecule._mode == 'alpha':  # Do not optimize BCCs in that case
            for alpha in range(nbcc):
                self.A[alpha][alpha] = 1
        else:
            #for ESP in [self.baseESP] + self.polESPs: #lgtm [py/unused-loop-variable]
            for j in range(self.natoms):
                    for k in range(self.natoms):
                        for alpha in range(nbcc):
                            for beta in range(nbcc):
                                self.A[alpha][beta] += T[j][alpha] * T[k][beta] * np.dot(self.dist[j],
                                                                                               self.dist[k])
            self.A = self.A *(len(self.polESPs)+1)
        # Polarizabilities part
        self.D = np.zeros((nalpha, nalpha))
        if self._molecule._mode != 'bcconly':  # Mode is not optimizing polarizabilies
            for ESP in [self.baseESP] + self.polESPs:
                for j in range(self.natoms):
                    for k in range(self.natoms):
                        for alpha in range(nalpha):
                            for beta in range(nalpha):
                                self.D[alpha][beta] += R[j][alpha] * R[k][beta] * np.multiply(
                                    np.dot(self.dist_x[j], self.dist_x[k]), ESP.e_field_at_atom[0][j] * ESP.e_field_at_atom[0][k])
                                self.D[alpha][beta] += R[j][alpha] * R[k][beta] * np.multiply(
                                    np.dot(self.dist_x[j], self.dist_y[k]), ESP.e_field_at_atom[0][j] * ESP.e_field_at_atom[1][k])
                                self.D[alpha][beta] += R[j][alpha] * R[k][beta] * np.multiply(
                                    np.dot(self.dist_x[j], self.dist_z[k]), ESP.e_field_at_atom[0][j] * ESP.e_field_at_atom[2][k])
                                self.D[alpha][beta] += R[j][alpha] * R[k][beta] * np.multiply(
                                    np.dot(self.dist_y[j], self.dist_x[k]), ESP.e_field_at_atom[1][j] * ESP.e_field_at_atom[0][k])
                                self.D[alpha][beta] += R[j][alpha] * R[k][beta] * np.multiply(
                                    np.dot(self.dist_y[j], self.dist_y[k]), ESP.e_field_at_atom[1][j] * ESP.e_field_at_atom[1][k])
                                self.D[alpha][beta] += R[j][alpha] * R[k][beta] * np.multiply(
                                    np.dot(self.dist_y[j], self.dist_z[k]), ESP.e_field_at_atom[1][j] * ESP.e_field_at_atom[2][k])
                                self.D[alpha][beta] += R[j][alpha] * R[k][beta] * np.multiply(
                                    np.dot(self.dist_z[j], self.dist_x[k]), ESP.e_field_at_atom[2][j] * ESP.e_field_at_atom[0][k])
                                self.D[alpha][beta] += R[j][alpha] * R[k][beta] * np.multiply(
                                    np.dot(self.dist_z[j], self.dist_y[k]), ESP.e_field_at_atom[2][j] * ESP.e_field_at_atom[1][k])
                                self.D[alpha][beta] += R[j][alpha] * R[k][beta] * np.multiply(
                                    np.dot(self.dist_z[j], self.dist_z[k]), ESP.e_field_at_atom[2][j] * ESP.e_field_at_atom[2][k])
        else:
            for alpha in range(nalpha):
                self.D[alpha][alpha] = 1

        # Cross interaction between BCC charges and polarizations
        self.B = np.zeros((nbcc, nalpha))
        self.C = np.zeros((nalpha, nbcc))
        if self._molecule._mode != 'alpha':
            for ESP in [self.baseESP] + self.polESPs:
                for j in range(self.natoms):
                    for k in range(self.natoms):
                        for alpha in range(nbcc):
                            for beta in range(nalpha):
                                self.B[alpha][beta] += T[j][alpha] * R[k][beta] * np.multiply(
                                    np.dot(self.dist[j], self.dist_x[k]), ESP.e_field_at_atom[0][k])
                                self.B[alpha][beta] += T[j][alpha] * R[k][beta] * np.multiply(
                                    np.dot(self.dist[j], self.dist_y[k]), ESP.e_field_at_atom[1][k])
                                self.B[alpha][beta] += T[j][alpha] * R[k][beta] * np.multiply(
                                    np.dot(self.dist[j], self.dist_z[k]), ESP.e_field_at_atom[2][k])
        if self._molecule._mode != 'bcconly':
            for ESP in [self.baseESP] + self.polESPs:
                for j in range(self.natoms):
                    for k in range(self.natoms):
                        for alpha in range(nalpha):
                            for beta in range(nbcc):
                                self.C[alpha][beta] += R[j][alpha] * T[k][beta] * np.multiply(
                                    np.dot(self.dist[k], self.dist_x[j]), ESP.e_field_at_atom[0][j])
                                self.C[alpha][beta] += R[j][alpha] * T[k][beta] * np.multiply(
                                    np.dot(self.dist[k], self.dist_y[j]), ESP.e_field_at_atom[1][j])
                                self.C[alpha][beta] += R[j][alpha] * T[k][beta] * np.multiply(
                                    np.dot(self.dist[k], self.dist_z[j]), ESP.e_field_at_atom[2][j])

        # Restraints for polarizaton
        #if hasattr(self, 'wrst1'):
        #    for alpha in range(nalpha):
        #        self.D[alpha][alpha] += mol2.group_pop[alpha] * self.wrst1 * self.dipscale / np.sqrt(
        #           np.square(self.qd[nbcc + alpha]) + 0.01)

        # No implementation of bcc restraints.


        #Combine all matrices
        self.X = np.concatenate((np.concatenate((self.A, self.B), axis=1), np.concatenate((self.C, self.D), axis=1)),
                                axis=0)


        """
        X12= np.zeros((len(self.X), len(self._molecule.same_polarization_atoms)))
        X22 = np.zeros((len(self._molecule.same_polarization_atoms),len(self._molecule.same_polarization_atoms)))
        self.X = np.concatenate((np.concatenate((self.X, X12), axis=1), np.concatenate((X12.transpose, X22), axis=1)),axis=0)
        
        
        for j, atoms in enumerate(self._molecule.same_polarization_atoms):
            self.X[5 * self.natoms + j] = 0.0

        """
    def build_vector_Y_BCC(self):
        """
        Creates vector Y for the BCC Pol method

        :param mol2: molecule object

        :return:
        """

        nbcc = self._molecule._trainingset._nbccs
        nalpha = self._molecule._trainingset._nalpha
        T = self._molecule.T
        R = self._molecule.R

        # Vector belonging to the BCCs
        self.Y1 = np.zeros(nbcc)
        if self._molecule._mode != 'alpha':
            for ESP in [self.baseESP] + self.polESPs:
                esp_values = ESP.esp_values.to('elementary_charge / angstrom').magnitude
                for beta in range(nbcc):
                    for k in range(self.natoms):
                        self.Y1[beta] += T[k][beta] * np.dot(esp_values, self.dist[k])
        else:
            self.Y1 = self.qbond

        # Vector belonging to the polarizabilities
        self.Y2 = np.zeros(nalpha)
        if self._molecule._mode != 'bcconly':
            for ESP in [self.baseESP] + self.polESPs:
                esp_values = ESP.esp_values.to('elementary_charge / angstrom').magnitude
                for beta in range(nalpha):
                    for k in range(self.natoms):
                        self.Y2[beta] += R[k][beta] * np.multiply(np.dot(esp_values, self.dist_x[k]), ESP.e_field_at_atom[0][k])
                        self.Y2[beta] += R[k][beta] * np.multiply(np.dot(esp_values, self.dist_y[k]), ESP.e_field_at_atom[1][k])
                        self.Y2[beta] += R[k][beta] * np.multiply(np.dot(esp_values, self.dist_z[k]), ESP.e_field_at_atom[2][k])


        self.Y = np.concatenate((self.Y1, self.Y2))


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
        self._lines_in_A = self.natoms + 1 + len(self._molecule.chemical_eq_atoms)
        self.B = np.zeros(self._lines_in_A)
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
        """
        Vector C corresponds to matrix D and is only for pur polarization fitting.

        :return:
        """
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

        #for j, atoms in enumerate(self._molecule.same_polarization_atoms):
        #    self.C[5 * self.natoms + j] = 0.0

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

    def substract_am1_potential(self):
        self.get_distances()
        for ESPGRID in [self.baseESP] + self.polESPs:
            ESPGRID.substract_am1_potential()
        self.delete_distances()

    def optimize_charges_alpha(self):
        """
        Builds the necessary matrix and vector and performs a charge and polarizabilities optimization for this 1 conformation.
        :return:
        """
        self.build_matrix_X()
        self.build_vector_Y()
        self.q_alpha = np.linalg.solve(self.X, self.Y)

    def optimize_charges_alpha_bcc(self):
        """
        Builds the necessary matrix and vector and performs a charge and polarizabilities optimization for this 1 conformation.
        :return:
        """
        self.build_matrix_X_BCC()
        self.build_vector_Y_BCC()
        # Check if bccs or alphas are not in the training set and set them to zero
        for i,row in enumerate(self.X):
            if all(row == 0.0):
                self.X[i][i]=1


        self.q_alpha = np.linalg.solve(self.X, self.Y)


    # i constantly have to delete the distatnce matrixes as storing them is too expensive in terms of memory
    # recomputing them every optimization cyclces has shown to be faster as long as i have not an extensive
    # amount of RAM available.

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

    def write_res_esp(self, q_alpha= None ):
        """
        NOT FINISHED YET!!!

        Writes the residual ESP to a file.

        :param qd: list of float
            Point charges and polarizabilities
        :return:
        """
        if q_alpha is None:
            q_alpha = self.q_alpha
        atoms = []
        for i,atom in enumerate(self.atom_positions.to('bohr').magnitude):
            atoms.append([self._molecule._atoms[i]._atomic_number,atom])

        for ESP in [self.baseESP] + self.polESPs:
            ESP.write_res_esp(q_alpha, atoms= atoms)

# =============================================================================================
# ESPGRID
# =============================================================================================

# noinspection PyTypeChecker
class ESPGRID:
    """

    """

    def define_grid(self, *args, **kwargs):
        for ele in args:
            # Check if the ending of the file is gesp
            if 'gesp' in ele:
                self.gridtype = 'gesp'
            # Check if one of the files end with espf
            # This is the same as a psi4 file but respyte uses this name
            if 'espf' in ele:
                self.gridtype = 'respyte'
            # psi4 type of grid
            if 'grid.dat' in ele:
                self.gridtype = 'psi4'

    def set_ext_e_field(self, vector):
        self._ext_e_field = vector

    def load_grid(self, *args):
        if self.gridtype == 'gesp':
            self.name = args[0].rstrip('.gesp')
            f = open(args[0], 'r')
            lines = f.readlines()
            f.close()
            for i, line in enumerate(lines):
                if 'ATOMIC' in line and 'COORDINATES' in line:
                    self.natoms = int(line.strip('\n').split()[-1])
                    for j in range(self.natoms):
                        # Convert the very unique D in gaussian number to E (the usual)
                        entry = lines[i + 1 + j].replace('D', 'E').split()
                        self.atoms.append(entry[0])
                        # Units are in Bohr in Gaussian
                        # pint should allow us to make the conversion easy
                        self.atom_positions.append(Q_([float(entry[k]) for k in range(1, 4, 1)], 'bohr'))
                if 'GRID' in line:
                    # number of grid points
                    self.ngrid = int(line.strip('\n').split()[-1])
                    # Stores the grid
                    grid = lines[i + 1:i + 1 + self.ngrid]
                    break
            # ToDo combine this with above. Current implementation is correct but dose not make a lot of sense
            # noinspection PyUnboundLocalVariable
            for i, line in enumerate(grid):
                grid[i] = [float(ele) for ele in line.replace('D', 'E').split()]

            self.positions = Q_(np.array(grid)[:, 1:4], 'bohr')
            self.esp_values = Q_(np.array(grid)[:, 0], 'elementary_charge / bohr')
        elif self.gridtype == 'respyte':
            self.name = args[0].rstrip('.espf')
            f = open(args[0], 'r')
            lines = f.readlines()
            f.close
            # I only need every second line because the other line is the electric field at point i
            ndata = int(len(lines) / 2) if len(lines) % 2 == 0 else int((len(lines) - 1) / 2)
            grid = np.zeros((ndata, 4))
            for i in range(ndata):
                grid[i] = [float(ele) for ele in lines[2 * i].split()]
            self.positions = Q_(np.array(grid)[:, 0:3], 'angstrom')
            self.esp_values = Q_(np.array(grid)[:, 3], 'elementary_charge / bohr')
        # Assuming that we only have the ESP in the esp.dat file
        elif self.gridtype == 'psi4':
            for ele in args:
                if "grid.dat" in ele:
                    gridfile = ele
                elif 'esp.dat' in ele:
                    espfile = ele
            self.name = 'esp'
            np.loadtxt(espfile)
            self.positions = Q_(np.loadtxt(gridfile), 'angstrom')
            self.esp_values = Q_(np.loadtxt(espfile), 'elementary_charge / bohr')

    def get_electric_field(self, ):
        """
        Calculates the electric field at every atomic positions.
        :return:
        """
        alpha = self._conformer.q_alpha[self._conformer._lines_in_A:self._conformer._lines_in_A + 3*self._conformer.natoms]
        alpha[np.where(alpha == 0.0)] += 10E-10

        # Load permanent charges for BCC method
        # For all other methods this is set to 0.0 STILL HAVE to implment
        if self._conformer.q_am1 is None:
            log.warning('I do not have AM1-type charges')
            self._conformer.q_am1 = np.zeros(self._conformer.natoms)
        #else:
        #    log.info('Substracting AM1 charge potential from the ESP values')
        # ToDo Add Code to substract am1 charge potential

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
                    thole_param=1.368711 # Change back to 1.368711
                    #thole_param = 0.390
                    dipole_tmp = np.where(alpha < 0.0, -alpha, alpha)
                    thole_v = np.multiply(self._conformer.diatomic_dist, np.float_power(
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
                        thole_ft = self._conformer._molecule.scale_scf
                        thole_fe = self._conformer._molecule.scale_scf
                    except Exception:

                        thole_ft = self._conformer._molecule.scale
                        thole_fe = self._conformer._molecule.scale
                    else:
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
                Bdip[j][j] = 1. / alpha[j]
            dipole_scf = np.linalg.solve(Bdip, self.e)
            self.e = np.divide(dipole_scf, alpha)
            log.debug(self.e)
        self.e_field_at_atom[0] = 1.0 * self.e[:self._conformer.natoms]
        self.e_field_at_atom[1] = 1.0 * self.e[self._conformer.natoms:2 * self._conformer.natoms]
        self.e_field_at_atom[2] = 1.0 * self.e[2 * self._conformer.natoms:3 * self._conformer.natoms]

    # Analysation stuff
    def calc_esp_q_alpha(self, q_alpha, mode='q'):
        """
        Calculates the ESP for a given set of point charges and polarizabilities.

        :param qd: list of float
            Point charges and polarizabilities
        :return:

        Calculates the ESP on every point of the initially read GESP file
        """
        self.q_pot = np.zeros(len(self.esp_values))
        for i in range(len(self.esp_values)):
            if mode == 'q':
                self.q_pot[i] = np.dot(q_alpha[:self._conformer.natoms], np.transpose(self._conformer.dist)[i])
            else:
                dipole = q_alpha[self._conformer._lines_in_A:]
                # self.dipole[self.dipole<0.0]=0.0
                try:
                    # e_dip=np.dot(self.dipole_scf[:self.natoms],np.transpose(self.dist_x)[i])+np.dot(self.dipole_scf[self.natoms:2*self.natoms],np.transpose(self.dist_y)[i])+np.dot(self.dipole_scf[2*self.natoms:3*self.natoms],np.transpose(self.dist_z)[i])
                    e_dip = np.dot(np.multiply(dipole[:self.natoms], self.e_field_at_atom[0]),
                                   np.transpose(self._conformer.dist_x)[i]) + np.dot(
                        np.multiply(dipole[self.natoms:2 * self.natoms], self.e_field_at_atom[1]),
                        np.transpose(self._conformer.dist_y)[i]) + np.dot(
                        np.multiply(dipole[2 * self.natoms:3 * self.natoms], self.e_field_at_atom[2]),
                        np.transpose(self._conformer.dist_z)[i])
                    self.q_pot[i] = np.dot(q_alpha[:self.natoms], np.transpose(self._conformer.dist)[i]) + e_dip
                except:
                    self.q_pot[i] = np.dot(q_alpha[:self.natoms], np.transpose(self._conformer.dist)[i])
            # if self.mode=='d':
            #    self.dipole=qd
            #    e_dip=np.dot(np.multiply(self.dipole[:self.natoms],self.e_x),np.transpose(self.dist_x)[i])+np.dot(np.multiply(self.dipole[self.natoms:2*self.natoms],self.e_y),np.transpose(self.dist_y)[i])+np.dot(np.multiply(self.dipole[2*self.natoms:3*self.natoms],self.e_z),np.transpose(self.dist_z)[i])
            #    self.q_pot[i]=e_dip

    def substract_am1_potential(self):
        self.calc_esp_q_alpha(self._conformer.q_am1, mode='q')
        self.esp_values = Q_(np.subtract(self.esp_values.to('elementary_charge / angstrom').magnitude, self.q_pot),
                         'elementary_charge / angstrom')

    def sub_esp_q_alpha(self, q_alpha):
        """
        Subtracts the ESP create by a set of point charges and polarizabilities.

        :param qd: list of float
            Point charges and polarizabilities
        :return:

        """
        if self._conformer._molecule._mode != 'alpha':  # Change that in bcc that this line is unnecessary
            self.calc_esp_q_alpha(q_alpha)
            self.esp_values = Q_(np.subtract(self.esp_values.to('elementary_charge / angstrom').magnitude, self.q_pot),'elementary_charge / angstrom')

    def calc_sse(self, q_alpha):
        """
        Calculate the Squared Sum of Errors between the stored ESP and a ESP created from qd.
        :param qd: list of float
            Point charges and polarizabilities
        :return:
        """
        self.calc_esp_q_alpha(q_alpha)
        self.sse = np.square(self.esp_values.to('elementary_charge / angstrom').magnitude - self.q_pot).sum()

    def write_res_esp(self, q_alpha, atoms = []):
        """
        NOT FINISHED YET!!!

        Writes the residual ESP to a file.

        :param qd: list of float
            Point charges and polarizabilities
        :return:
        """
        self.calc_esp_q_alpha(q_alpha)
        res_pot = np.subtract(self.esp_values.to('elementary_charge / angstrom').magnitude, Q_(self.q_pot, 'elementary_charge / angstrom'))
        #res_pot = (self.esp_values - self.q_pot)#.to('elementary_charge / bohr').magnitude
        f = open(self.name + '.rgesp', 'w')
        f.write(' ESP FILE - ATOMIC UNITS\n')
        f.write(' CHARGE =  {0} - MULTIPLICITY =   1\n'.format(self._conformer._molecule._charge))
        f.write(' ATOMIC COORDINATES AND ESP CHARGES. #ATOMS =     {} \n'.format(np.sum(self.natoms)))
        for i in range(self._conformer.natoms):
            f.write(
                ' {} {} {} {} {}\n'.format(atoms[i][0], atoms[i][1][0], atoms[i][1][1], atoms[i][1][2],
                                           self._conformer._molecule.q_alpha[i]))
        f.write(' ESP VALUES AND GRID POINT COORDINATES. #POINTS =   {}\n'.format(len(self.esp_values)))
        for i in range(len(self.esp_values)):
            try:
                f.write(' {} {} {} {}\n'.format(res_pot[i].magnitude, self.positions[i][0].magnitude, self.positions[i][1].magnitude, self.positions[i][2].magnitude))
            except: # Difference between python 3.6 and 3.7
                f.write(' {} {} {} {}\n'.format(res_pot[i], self.positions[i][0].magnitude, self.positions[i][1].magnitude, self.positions[i][2].magnitude))
        f.close()

# =============================================================================================
# BCCUnpolESP
# =============================================================================================

# Inheritace from the ESPGRID class
class BCCUnpolESP(ESPGRID):
    """

    """

    def __init__(self, *args, conformer=None):
        # Decide if we have a Gaussian grid or a psi4 grid
        self.gridtype = None

        # Initilize values
        self.natoms = -1

        #Initialize array to store the atom position and the element from the ESP file
        #ToDo Check if those are the same as in the Conformer file
        self.atoms = []
        self.atom_positions = []

        # Checks what grid type psi4 or gaussian was used
        self.define_grid(*args)

        self.esp_values = None
        self.positions = None

        self._conformer = conformer

        # External e-field is 0 in all directions
        vector = Q_([0, 0, 0], 'elementary_charge / bohr / bohr')
        self.set_ext_e_field(vector)

        # Load the grid to self.positions and self.esp_values
        self.load_grid(*args)

        # As the electric field at the atom positions is different for every external polarization
        # the electric field for each atom has to be stored on the ESPGrid level
        self.e_field_at_atom = np.zeros((3, self._conformer.natoms))
        # No calculation of the e-fields at this stage only initializing

# =============================================================================================
# BCCPolESP
# =============================================================================================
# Inheritace from the ESPGRID class
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
    ifs = oechem.oemolistream('/home/mschauperl/kirk/charge_method/medium_set/molecule2/conf0/mp2_0.mol2')
    oemol=oechem.OEMol()
    oechem.OEReadMol2File(ifs, oemol)
    ifs2 = oechem.oemolistream('/home/mschauperl/kirk/charge_method/medium_set/molecule3/conf0/mp2_0.mol2')
    oemol2=oechem.OEMol()
    oechem.OEReadMol2File(ifs2, oemol2)
    ifs3 = oechem.oemolistream('/home/mschauperl/programs/resppol/resppol/data/test_data/butanol_0.mol2')
    oemol3=oechem.OEMol()
    oechem.OEReadMol2File(ifs3, oemol3)
    log.debug("This is a test")
    print(find_eq_atoms(oemol))
    print(find_eq_atoms(oemol))
    print(find_eq_atoms(oemol2))
    print(find_eq_atoms(oemol2))
    print(find_eq_atoms(oemol3))
    print(find_eq_atoms(oemol3))

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
    
    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test2.mol2')
    test = TrainingSet(mode='q_alpha',SCF= True,scf_scaleparameters=[1,1,1], scaleparameters=[1,1,1])
    #test = TrainingSet(mode='q_alpha')
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    #test.molecules[0].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    #test.molecules[0].conformers[1].add_baseESP(espfile)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3_Z+.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, 1.], 'elementary_charge / bohr / bohr'))
    #test.molecules[0].conformers[1].add_polESP(espfile, e_field=Q_([0.0, 0.0, 0.8],'elementary_charge / bohr / bohr'))
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3_Z-.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, -1.], 'elementary_charge / bohr / bohr') )
    #test.molecules[0].conformers[1].add_polESP(espfile, e_field=Q_([0.0, 0.0, -0.8], 'elementary_charge / bohr / bohr') )
    #test.build_matrix_X()
    #test.build_vector_Y()
    test.build_matrix_X_BCC()
    test.build_vector_Y_BCC()
    test.optimize_bcc_alpha()
    print(test.molecules[0].conformers[0].q_alpha)
    #test.optimize_charges_alpha()
    #print(test.q_alpha)
    print(test.molecules[0].conformers[0].baseESP.e_field_at_atom)
    print(test.molecules[0].conformers[0].polESPs[0].e_field_at_atom)
    

    datei = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test2.mol2')
    test = TrainingSet(mode='q_alpha',SCF= True, thole = True)
    test.add_molecule(datei)
    test.molecules[0].add_conformer_from_mol2(datei)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3.gesp')
    test.molecules[0].conformers[0].add_baseESP(espfile)
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3_Z+.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, 1.0], 'elementary_charge / bohr / bohr'))
    espfile = os.path.join(ROOT_DIR_PATH, 'resppol/data/fast_test_data/test3_Z-.gesp')
    test.molecules[0].conformers[0].add_polESP(espfile, e_field=Q_([0.0, 0.0, -1.0], 'elementary_charge / bohr / bohr') )
    test.optimize_charges_alpha()
    test.molecules[0].conformers[0].baseESP.calc_esp_q_alpha(test.q_alpha)
    test.molecules[0].conformers[0].baseESP.calc_sse(test.q_alpha)
    test.molecules[0].conformers[0].baseESP.sub_esp_q_alpha(test.q_alpha)
    test.molecules[0].conformers[0].write_res_esp()
    print(test.molecules[0].conformers[0].baseESP.q_pot)

    print(test.molecules[0].conformers[0].q_alpha)
    """
