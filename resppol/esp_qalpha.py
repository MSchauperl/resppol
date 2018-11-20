import numpy as np
from scipy.spatial import distance
import copy
import random
from collections import Counter
import logging as log

BOHR = float(0.52917722086)


class molecule():
    """
    Takes a mol2 file and stores the relevant information.



    Attributes:
    -----------
        natoms : int
            The number of atoms
        nbonds : int
            The number of bonds
        crds: list of [float,float,float]
            The atomic coordinates
        atoms: list of string
            Element of every atoms
        atomtyps: list of string
            Amber Atom typ of every molecule
        charges: list of float
            Atomic charges stored in the mol2 file
        bonds: list of [int,int]
            Atomic numbers of atoms forming a bond
        bondsgroup: list of [int,int]
            BBC group of every atom
        T: 2 dim matrix of int
            Definies which BCC charges are added and substracte to every atom
        R: 2 dim matrix of int
            Stores which atom has which polarization parameter
        nbondtyps: int
            Number of differnt BCCs in the set
        lenpoltypes:
            Number of all possible polarization parameters

    All bonds between atoms with the same BCC groups are treated equally.
    """

    counter = 0

    def __init__(self, datei):
        """
            The init molecule takes a mol2 file (datei) and stores the number of atoms (natoms), number bonds(nbonds),
        the atomic coordinates (crds), the atomic charges (charges), atom element (atoms), and the bonds in the molecule
        (bonds).The atom types have to be assigned in the mol2 file. Therefore use antechamber or something similiar
        before that to assignatom types

        Parameters:
        ----------
            datei : string
                Path to mol2 file
        """

        f = open(datei, 'r')
        lines = f.readlines()

        for i in range(len(lines)):
            if '@<TRIPOS>MOLECULE' in lines[i]:
                molecule.counter += 1
                self.natoms = int(lines[i + 2].split()[0])  # Number of atoms in mol2 file
                self.nbonds = int(lines[i + 2].split()[1])  # Number of bonds in mol2 file

            elif '@<TRIPOS>ATOM' in lines[i]:
                self.crds = np.zeros((self.natoms, 3))  # atomic coordinates
                self.atoms = ['' for j in range(self.natoms)]  # stores the element
                self.atomtyps = ['' for j in range(self.natoms)]  # stores the atomtype of all atoms
                self.charges = np.zeros(self.natoms + 1)  # charges assigned in the mol2 file

                # Read in the atom data
                for j in range(self.natoms):
                    for k in range(3):
                        self.crds[j][k] = float(lines[i + 1 + j].split()[2 + k]) / BOHR

                    self.atoms[j] = lines[i + 1 + j].split()[1]
                    self.charges[j] = lines[i + 1 + j].split()[8]  # stores the stomic charges in the mol2 file
                    self.atomtyps[j] = lines[i + 1 + j].split()[5]
                # Last element of charges is the total charge
                self.charges[self.natoms] = np.sum(self.charges[:-1])

            # Read in the bond information of the mol2 file
            elif '@<TRIPOS>BOND' in lines[i]:
                self.bonds = [[0,0,0.0] for j in range(self.nbonds)]
                for j in range(self.nbonds):
                    self.bonds[j][0] = int(lines[i + 1 + j].split()[1]) - 1
                    self.bonds[j][1] = int(lines[i + 1 + j].split()[2]) - 1

                    try:
                        int(lines[i + 1 + j].split()[3])

                    except ValueError:  # if aromatic bond is definied
                        if lines[i + 1 + j].split()[3] == 'ar':
                            self.bonds[j][2] = 1.5

                    else:
                        self.bonds[j][2] = float(lines[i + 1 + j].split()[3])

        f.close()

    def bondtypes(self, groups, bondtyps, bondnames):
        """
        This function assigns a BCC group to every bond.

        Parameters:
        ----------
            groups : Dictonary atomtyp --> int
                Assigns every atom a group where it belongs to.
                !Would be great to change this to a SMARTS string for bonds!
            bondtyps: list of [int,int]
                Bonds in the set between different groups
            bondnames: list of [string, string]
                The atomtypes involved in every bondtyp


        Uses thedefinition which atoms belong to the same group for the BCC assignment (groups).
        In bondtypes the already occured types of bonds in the set
        (definied by the 2 groups of the involved atoms) are stored.
        If new bonds are in this mol2 file the are appended to bondtyps
        Bondnames stores the name of the atomtypes involved in every appended bond.
        """

        self.bondsgroup = [[0 for k in range(2)] for j in range(self.nbonds)]

        for i in range(self.nbonds):
            self.bondsgroup[i] = [groups[self.atomtyps[self.bonds[i][0]]], groups[self.atomtyps[self.bonds[i][1]]]]

            # Make sure the first group is always the first one. We do not want to depend on the atom order of the bond
            if self.bondsgroup[i][1] < self.bondsgroup[i][0]:
                self.bondsgroup[i] = [self.bondsgroup[i][1], self.bondsgroup[i][0]]
                self.bonds[i] = [self.bonds[i][1], self.bonds[i][0]]

            if self.bondsgroup[i] not in bondtyps:
                # we do not want to have BCC between bonds of the same "atoms"
                if self.bondsgroup[i][1] != self.bondsgroup[i][0]:
                    bondtyps.append(self.bondsgroup[i])
                    bondnames.append([self.atomtyps[self.bonds[i][0]], self.atomtyps[self.bonds[i][1]]])

    def bondmatrix(self, bondtyps):
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
        self.nbondtyps = len(bondtyps)
        self.T = np.zeros((self.natoms, self.nbondtyps))
        for i in range(self.nbonds):
            if self.bondsgroup[i][1] != self.bondsgroup[i][0]:
                try:
                    bnum = bondtyps.index(self.bondsgroup[i])  # bondsgroup[i] is a 1 dim array of size 2
                    self.T[self.bonds[i][0]][bnum] += 1
                    self.T[self.bonds[i][1]][bnum] -= 1

                except:  # Just for safety issues Never really saw that message up to now
                    print('Something is wrong with the bondtyps definition')

    def poltypes(self, groups):
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
        self.lenpoltypes = len(Counter(groups.values()))
        self.group_pop = np.zeros(self.lenpoltypes)
        self.R = np.zeros((self.natoms, self.lenpoltypes))
        for i in range(self.natoms):
            self.R[i][groups[self.atomtyps[i]]] += 1
            self.group_pop[groups[self.atomtyps[i]]] += 1


# END of Class


class esp():
    """
    Takes an GESP and stores the information. Build optimisation matrixes A and vector B.

    Attributes:
    ----------
    efield_ext: [float,float,float]
        External electric field in x, y, z.
    freeh: Boolean
        If True, restraints for heavy atoms only.
    thole: Boolean
        If True, Thole screening is applied in addition to regular screening
        False (default)
    SCF: Boolean
        If True SCF approach is used for dip-dip interactions
        If False direct approach (default)
    eqatoms: list [int,int]
        list of intramolecular atoms[id] with the same charge
        if second entry < 0: atom with id[first entry] is restraint to qin value
    eqdipoles: list of [int, int]
        list of intramolecular atoms with the same polarizability
    step: int
        Number of optimisation runs already done
    natoms: int
        Number of atoms
    ndipoles: int
        Number of dipole compounds (x,y,z) 3*natoms
    stage:
        Defines the stage of the RESP fitting (only used for 2 stage RESP fitting
    atomcrd: list of [float,float,float]
        Atomic coordinates
    crd: list of [float,float,float]
        ESP Points coordinates
    pot: list of float
        ESP values
    npoints: int
        Number of ESP points
    qin: list of float
        Atomic charges from the GESP file
    charge: int
        Molecular charge
    qd: list of float
        Stores the charges and polarizabilities
    Alines: int
        Number of lines of matrix A
    Dlines: int
        Number of lines of matrix B
    totallines: int
        Total number of lines of optimisation matrix
    e_x,e_y,e_z: float
        Stores the electric field at every atom in x,y,z respectively.
    qfix: list of float
        Charges from the lower QM method, for which BCC should be applied


    This class is the central part of this program. Every gesp is its own object.
    The wrapper program is combining different esp classes to build one individual
    matrix which then has to be solved. One of the main tasks is to calculate all distances
    and they differ for almost every gesp.
    Therefore it seems not really sensible to combine multiple gesp in one esp.
    A later implementation also of psi4 esp files is planned but not implemented up to now.
    Should be very little effort to do this
    """
    counter = 0  # How many esp objects are definied
    res_esp_counter = 0  # how many residuel esps already wrote to file
    test_counter = 0  # if option test is used


    def __init__(self, datei, mol2, mode='q', eqatoms=None, eqatoms2=None, ext=None, eqdipoles=None, SCF=False, wrst1='', hrst1='',
                 thole=False, dipscale='0.01'):
        """
        Creates an instance on the basis of an gaussian ESP file. and sets parameters

        Parameters:
        ----------
        datei: string
            Path to gesp file
        mode: string
            Defines what is optimized [q,d,qd,dnoq,alpha,bcc,bcconly,alphabccfix]
        eqatoms: list of [int, int]
            list of intramolecular atoms[id] with the same charge
            if second entry < 0: atom with id[first entry] is restraint to qin value
        eqatoms: list of [int, int]
            list of intramolecular atoms[id] with the same charge
            Used for the 2nd stage of charge RESP fitting
        ext: [float,float,float]
            External electric field in x, y, z.
        eqdipoles: list of [int, int]
            list of intramolecular atoms with the same polarizability
        SCF: Boolean
            If True SCF approach is used for dip-dip interactions
            If False direct approach (default)
        wrst: float
            force constant for charge restraints (parabolic)
        hrst1: float
            force constant for harmonic charge restraints
        thole: Boolean
            If True, Thole screening is applied in addition to regular screening
        dipscale: float
            force constant for polarizability restraints

        There are two main differences how this class can be used.

        The RESP Pol Method with the modes:
        q only charges are optimized, polarizabilities are neglected
        qd charges and polarizabilities are co-optimized
        d only polarizabilities are optimized. Charges are restrained to their initial value
        dnoq only polarizabilities are optimized. Charges are neglected

        The BCC Pol Method:
        alpha: Only polarizabilities are optimized. Charges and BCCs are ignored
        bcc: Charges and BCC are co-optimized
        bcconly: Only BCCs are optimized. Polarizabilities are restrained to initial value
        alphabccfix: Only alphas are optimized, BCCs are restrained to the initial values
        """

        esp.counter += 1

        if eqatoms is None:
            eqatoms = []
        if eqatoms2 is None:
            eqatoms2 = []
        if eqdipoles is None:
            eqdipoles = []
        if ext is None:
            eqatoms = [0.0, 0.0, 0.0]

        self.efield_ext = ext
        self.freeh = False
        self.thole = thole
        self.SCF = SCF
        self.mode = mode
        self.step = 0
        self.eqatoms = eqatoms
        self.eqatoms2 = eqatoms2
        self.eqdipoles = eqdipoles
        f = open(datei, 'r')
        lines = f.readlines()
        f.close()

        self.stage = 1  # Was used to implement the 2 step RESP process.

        # Set restraints
        if dipscale != '':
            self.dipscale = float(dipscale)  # Value the dipole-restraints are scaled compared to the charge restraints

        else:
            self.dipscale = 0.01  # Standard value if a stupid value is given
            print('Dipolscale is set to 0.01 because the input value was crap')

        if wrst1 != '' and float(wrst1) != 0.0:  # Parabolic restraints activated
            log.debug("Using parabolic restraints")
            self.wrst1 = float(wrst1)

        if hrst1 != '':  # harmonic restraints activated
            log.debug("Using harmonic restraints")
            self.hrst1 = float(hrst1)
        # End of set restraints

        # Read in GESP file including atoms
        for i in range(len(lines)):
            if 'CHARGE =' in lines[i]:
                self.charge = int(lines[i].split()[2])  # Molecule Charge


            elif 'ATOMIC COORDINATES AND ESP CHARGES.' in lines[i]:
                """Reads in all atoms in the gesp files and stores their coordinates."""
                self.natoms = int(lines[i].split()[-1])  # number of atoms
                self.ndipoles = self.natoms * 3  # number of dipoles 3 per atom
                self.atomcrd = np.zeros((self.natoms, 3))
                self.qin = np.zeros(self.natoms)  # list of atomic charges
                self.atoms = ['' for j in range(self.natoms)]
                for j in range(self.natoms):
                    tmp = lines[i + 1 + j].replace('D', 'E',
                                                   4).split()  # gaussian uses D instead of E for scientific numbers. Just to annoy me

                    for k in range(3):
                        self.atomcrd[j][k] = tmp[k + 1]
                    self.qin[j] = tmp[4]
                    self.atoms[j] = tmp[0]


            elif 'ESP VALUES AND GRID POINT COORDINATES. #POINTS =' in lines[i]:
                """Read in all the ESP points (coordinates and values)"""
                self.npoints = int(lines[i].split()[-1])
                self.pot = np.zeros(self.npoints)  # potential for every point
                self.crd = np.zeros((self.npoints, 3))  # coordinated for every grid point
                # Read in the whole gesp. Very time consuming step but have not found a better way to read in the data.
                for j in range(self.npoints):
                    tmp = lines[i + 1 + j].replace('D', 'E', 4).split()
                    for k in range(3):
                        self.crd[j][k] = tmp[k + 1]
                    self.pot[j] = float(tmp[0])

        # Set initial values accorind to mode
        if mode == 'dnoq':
            self.charge = 0.0
            self.qin = np.zeros(
                self.natoms)  # Set initial charges to 0 as we do not want to restraint them to another value
            self.mode = 'd'

        if self.mode == 'd':  # restrain all charges to initial value. used for d after q. No other charge restraints are necessary.
            self.eqatoms = []
            for i in range(self.natoms - 1):
                self.eqatoms.append([i, -1])
        self.Alines = self.natoms + 1 + len(self.eqatoms)
        self.aniso_lines = 2 * self.natoms
        self.Dlines = self.ndipoles + self.aniso_lines + len(self.eqdipoles)  # Space for the isotropic restraints

        if mode == 'q':
            self.qd = np.zeros(self.Alines)  # only charges are optimized
            self.totallines = self.Alines
            self.qin = np.zeros(self.natoms)
        # elif mode=='qd' or mode=='d':
        else:
            self.qd = np.zeros(self.Alines + self.Dlines)
            self.totallines = self.Alines + self.Dlines

        # Checks if dipole restraints make sense. Can only be used in isotropic form with the current implementation
        self.dipole_rst = []
        for i in range(len(self.eqdipoles)):
            if self.eqdipoles[i][0] > self.natoms - 1 or self.eqdipoles[i][0] > self.natoms - 1:
                log.error("Dipoles restraints are not valid\n")
                exit()

        # Initialize E-vectors
        self.e_x = np.zeros(self.natoms)  # Stores the electric field at every atomic position
        self.e_y = np.zeros(self.natoms)
        self.e_z = np.zeros(self.natoms)
        # Initializing
        self.scale = np.ones((self.natoms, self.natoms))
        self.bound12 = np.zeros((self.natoms, self.natoms))
        self.bound13 = np.zeros((self.natoms, self.natoms))
        self.bound14 = np.zeros((self.natoms, self.natoms))


    def scaling(self, bonds, scaleparameters=None,scf_scaleparameters=None):
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
        if scaleparameters is None:
            scaleparameters=[0.0,0.0,0.8333333333]



        # Building connection matrix
        for k in range(len(bonds)):
            self.bound12[bonds[k][0]][bonds[k][1]] = 1.0
            self.bound12[bonds[k][1]][bonds[k][0]] = 1.0

        for i in range(len(self.bound12)):
            b12 = np.where(self.bound12[i] == 1.0)[0]
            for j in range(len(b12)):
                b12t = np.where(self.bound12[b12[j]] == 1.0)[0]
                for k in range(len(b12t)):
                    if i != b12t[k]:
                        self.bound13[b12t[k]][i] = 1.0
                        self.bound13[i][b12t[k]] = 1.0

        for i in range(self.natoms):
            b13 = np.where(self.bound13[i] == 1.0)[0]
            for j in range(len(b13)):
                b13t = np.where(self.bound12[b13[j]] == 1.0)[0]
                for k in range(len(b13t)):
                    if self.bound12[b13t[k]][i] == 0.0:
                        self.bound14[b13t[k]][i] = 1.0
                        self.bound14[i][b13t[k]] = 1.0

        for i in range(self.natoms):
            self.scale[i][i] = 0.0
        # find values in matrix with value 1.0
        b12 = np.array(np.where(self.bound12 == 1.0)).transpose()
        b13 = np.array(np.where(self.bound13 == 1.0)).transpose()
        b14 = np.array(np.where(self.bound14 == 1.0)).transpose()

        # Fill scaling matrix with values
        for i in range(len(b12)):
            self.scale[b12[i][0]][b12[i][1]] = scaleparameters[0]  # Value for 1-2 interaction 0 means interactions are neglected
        for i in range(len(b13)):
            self.scale[b13[i][0]][b13[i][1]] = scaleparameters[1]  # Value for 1-3 interaction 0 means interactions are neglected
        for i in range(len(b14)):
            self.scale[b14[i][0]][b14[i][1]] = scaleparameters[2]  # Value for the 1-4 scaling

        # Different Scaling parameter for SCF
        if scf_scaleparameters !=None:
            self.scale_scf = np.ones((self.natoms, self.natoms))
            for i in range(len(b12)):
                self.scale_scf[b12[i][0]][b12[i][1]] = scf_scaleparameters[0]  # Value for 1-2 interaction 0 means interactions are neglected
            for i in range(len(b13)):
                self.scale_scf[b13[i][0]][b13[i][1]] = scf_scaleparameters[1]  # Value for 1-3 interaction 0 means interactions are neglected
            for i in range(len(b14)):
                self.scale_scf[b14[i][0]][b14[i][1]] = scf_scaleparameters[2]  # Value for the 1-4 scaling



        if scaleparameters == 'onlyinter':
            #self.scale = np.ones((self.natoms, self.natoms))
            # Building connection matrix
            for k in range(1,len(bonds)):
                if np.all(self.bound12[:k].transpose()[k:] == 0):
                    self.log('Multiple Molecules detected: No connection between atom {} and {}'.format(k,k+1))

    def update_for_daq(self):
        self.eqatoms = []
        for i in range(self.natoms - 1):
                self.eqatoms.append([i, -1])
        self.Alines = self.natoms + 1 + len(self.eqatoms)
        self.aniso_lines = 2 * self.natoms
        self.Dlines = self.ndipoles + self.aniso_lines + len(self.eqdipoles)
        self.qd = np.zeros(self.Alines + self.Dlines)
        self.totallines = self.Alines + self.Dlines

    # noinspection PyAttributeOutsideInit
    def distances(self):
        """
        Calculates all the necessary distances.

        Attributes:
        -----------
        dist: list of float
            1 over distance from atom to esp point
        dist_3: list of float
            1 over distance from atom to esp point ^3
        dist_x: list of float
            distance from atom to esp point in x direction /  atom esp distance^3
        adist: list of float
            1 / diatomic distance
        adist_3: list of float
            1 / diatomic distance^3
        adist_5: list of float
            1 / diatomic distance^5
        adist_x: list of float
            diatomic distance in x direction / diatomic distance^3
        adistb_x: list of float
            diatomic distance in x direction

        """


        # Distances between atoms and ESP points
        self.dist = np.zeros((self.natoms, self.npoints))
        self.dist_3 = np.zeros((self.natoms, self.npoints))
        self.dist_x = np.zeros((self.natoms, self.npoints))
        self.dist_y = np.zeros((self.natoms, self.npoints))
        self.dist_z = np.zeros((self.natoms, self.npoints))

        self.dist = 1. / distance.cdist(self.atomcrd, self.crd)
        self.dist_3 = np.power(self.dist, 3)  # maybe free afterwards
        self.dist_x = -np.multiply(np.subtract.outer(np.transpose(self.atomcrd)[0], np.transpose(self.crd)[0]),
                                   self.dist_3)
        # self.dist_x2=np.multiply(np.transpose(np.subtract.outer(np.transpose(self.crd)[0],np.transpose(self.atomcrd)[0])),self.dist_3)
        self.dist_y = -np.multiply(np.subtract.outer(np.transpose(self.atomcrd)[1], np.transpose(self.crd)[1]),
                                   self.dist_3)
        self.dist_z = -np.multiply(np.subtract.outer(np.transpose(self.atomcrd)[2], np.transpose(self.crd)[2]),
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

        self.adist = distance.cdist(self.atomcrd, self.atomcrd)
        di = np.diag_indices(self.natoms)
        self.adist[di] = 1.0E10
        # self.adist=np.fill_diagonal(self.adist,1.0)
        self.adist = 1. / self.adist
        self.adist_3 = np.power(self.adist, 3)
        self.adist_5 = np.power(self.adist, 5)
        self.adist[di] = 0.0
        self.adist_x = np.multiply(np.subtract.outer(np.transpose(self.atomcrd)[0], np.transpose(self.atomcrd)[0]),
                                   self.adist_3)  # X distance between two atoms divided by the dist^3
        self.adist_y = np.multiply(np.subtract.outer(np.transpose(self.atomcrd)[1], np.transpose(self.atomcrd)[1]),
                                   self.adist_3)
        self.adist_z = np.multiply(np.subtract.outer(np.transpose(self.atomcrd)[2], np.transpose(self.atomcrd)[2]),
                                   self.adist_3)
        self.adistb_x = np.subtract.outer(np.transpose(self.atomcrd)[0],
                                          np.transpose(self.atomcrd)[0])  # X distances between two atoms
        self.adistb_y = np.subtract.outer(np.transpose(self.atomcrd)[1], np.transpose(self.atomcrd)[1])
        self.adistb_z = np.subtract.outer(np.transpose(self.atomcrd)[2], np.transpose(self.atomcrd)[2])

        # self.dist_d=np.multiply(self.dist_d,self.dist_3)
        # for i in range(len(self.atomcrd3)):
        #    for j in range(len(self.crd3)):
        #        self.dist_d[i][j]=1./(self.atomcrd3[i]-self.crd3[j])


    def delete_dist(self):
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


    """
    def make_Abcc(self, mol2): #Old
        self.A = np.zeros((mol2.nbondtyps, mol2.nbondtyps))
        for j in range(self.natoms):
            for k in range(self.natoms):
                for alpha in range(mol2.nbondtyps):
                    for beta in range(mol2.nbondtyps):
                        self.A[alpha][beta] += mol2.T[j][alpha] * mol2.T[k][beta] * np.dot(self.dist[j], self.dist[k])

    # Only polarisation in reduced matrix form
    def make_D(self, mol2): #Old
        self.D = np.zeros((mol2.lenpoltypes, mol2.lenpoltypes))
        for j in range(self.natoms):
            for k in range(self.natoms):
                for alpha in range(mol2.lenpoltypes):
                    for beta in range(mol2.lenpoltypes):
                        self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                            np.dot(self.dist_x[j], self.dist_x[k]), self.e_x[j] * self.e_x[k])
                        self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                            np.dot(self.dist_x[j], self.dist_y[k]), self.e_x[j] * self.e_y[k])
                        self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                            np.dot(self.dist_x[j], self.dist_z[k]), self.e_x[j] * self.e_z[k])
                        self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                            np.dot(self.dist_y[j], self.dist_x[k]), self.e_y[j] * self.e_x[k])
                        self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                            np.dot(self.dist_y[j], self.dist_y[k]), self.e_y[j] * self.e_y[k])
                        self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                            np.dot(self.dist_y[j], self.dist_z[k]), self.e_y[j] * self.e_z[k])
                        self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                            np.dot(self.dist_z[j], self.dist_x[k]), self.e_z[j] * self.e_x[k])
                        self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                            np.dot(self.dist_z[j], self.dist_y[k]), self.e_z[j] * self.e_y[k])
                        self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                            np.dot(self.dist_z[j], self.dist_z[k]), self.e_z[j] * self.e_z[k])
    """
    # Combines BCC and polarizabilities
    def make_Xn(self, mol2):
        """
        Creates the Matrix A for the BCC-POL approach:

        :param mol2: molecule class object

        :return:

        The math and the optimization method is explained in the current paper draft.
        """

        # BCC part
        self.A = np.zeros((mol2.nbondtyps, mol2.nbondtyps))
        if self.mode == 'alpha': # Do not optimize BCCs in that case
            for alpha in range(mol2.nbondtyps):
                self.A[alpha][alpha] = 1
        else:
            for j in range(self.natoms):
                for k in range(self.natoms):
                    for alpha in range(mol2.nbondtyps):
                        for beta in range(mol2.nbondtyps):
                            self.A[alpha][beta] += mol2.T[j][alpha] * mol2.T[k][beta] * np.dot(self.dist[j],
                                                                                               self.dist[k])
        # Polarizabilities part
        self.D = np.zeros((mol2.lenpoltypes, mol2.lenpoltypes))
        if self.mode != 'bcconly': #Mode is not optimizing polarizabilies
            for j in range(self.natoms):
                for k in range(self.natoms):
                    for alpha in range(mol2.lenpoltypes):
                        for beta in range(mol2.lenpoltypes):
                            self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                                np.dot(self.dist_x[j], self.dist_x[k]), self.e_x[j] * self.e_x[k])
                            self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                                np.dot(self.dist_x[j], self.dist_y[k]), self.e_x[j] * self.e_y[k])
                            self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                                np.dot(self.dist_x[j], self.dist_z[k]), self.e_x[j] * self.e_z[k])
                            self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                                np.dot(self.dist_y[j], self.dist_x[k]), self.e_y[j] * self.e_x[k])
                            self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                                np.dot(self.dist_y[j], self.dist_y[k]), self.e_y[j] * self.e_y[k])
                            self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                                np.dot(self.dist_y[j], self.dist_z[k]), self.e_y[j] * self.e_z[k])
                            self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                                np.dot(self.dist_z[j], self.dist_x[k]), self.e_z[j] * self.e_x[k])
                            self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                                np.dot(self.dist_z[j], self.dist_y[k]), self.e_z[j] * self.e_y[k])
                            self.D[alpha][beta] += mol2.R[j][alpha] * mol2.R[k][beta] * np.multiply(
                                np.dot(self.dist_z[j], self.dist_z[k]), self.e_z[j] * self.e_z[k])
        else:
            for alpha in range(mol2.lenpoltypes):
                self.D[alpha][alpha] = 1

        # Cross interaction between BCC charges and polarizations
        self.B = np.zeros((mol2.nbondtyps, mol2.lenpoltypes))
        self.C = np.zeros((mol2.lenpoltypes, mol2.nbondtyps))
        if self.mode != 'alpha':
            for j in range(self.natoms):
                for k in range(self.natoms):
                    for alpha in range(mol2.nbondtyps):
                        for beta in range(mol2.lenpoltypes):
                            self.B[alpha][beta] += mol2.T[j][alpha] * mol2.R[k][beta] * np.multiply(
                                np.dot(self.dist[j], self.dist_x[k]), self.e_x[k])
                            self.B[alpha][beta] += mol2.T[j][alpha] * mol2.R[k][beta] * np.multiply(
                                np.dot(self.dist[j], self.dist_y[k]), self.e_y[k])
                            self.B[alpha][beta] += mol2.T[j][alpha] * mol2.R[k][beta] * np.multiply(
                                np.dot(self.dist[j], self.dist_z[k]), self.e_z[k])
        if self.mode != 'bcconly':
            for j in range(self.natoms):
                for k in range(self.natoms):
                    for alpha in range(mol2.lenpoltypes):
                        for beta in range(mol2.nbondtyps):
                            self.C[alpha][beta] += mol2.R[j][alpha] * mol2.T[k][beta] * np.multiply(
                                np.dot(self.dist[k], self.dist_x[j]), self.e_x[j])
                            self.C[alpha][beta] += mol2.R[j][alpha] * mol2.T[k][beta] * np.multiply(
                                np.dot(self.dist[k], self.dist_y[j]), self.e_y[j])
                            self.C[alpha][beta] += mol2.R[j][alpha] * mol2.T[k][beta] * np.multiply(
                                np.dot(self.dist[k], self.dist_z[j]), self.e_z[j])

        # Restraints for polarizaton
        if hasattr(self, 'wrst1'):
            for alpha in range(mol2.lenpoltypes):
                self.D[alpha][alpha] += mol2.group_pop[alpha] * self.wrst1 * self.dipscale / np.sqrt(
                    np.square(self.qd[mol2.nbondtyps + alpha]) + 0.01)

        # No implementation of bcc restraints.

        # Combine all matrices
        self.X = np.concatenate((np.concatenate((self.A, self.B), axis=1), np.concatenate((self.C, self.D), axis=1)),
                                axis=0)


    def make_X(self):
        """
        Creates Matrix X for the RESP-POl method.

        RESP and Polarization with the large matrix.
        Probably worth changing it to the new model.

        Again the math is shown in the manuscript.
        """
        self.A = np.zeros((self.Alines, self.Alines))
        self.B = np.zeros((self.Alines, self.Dlines))
        self.D = np.zeros((self.Dlines, self.Dlines))
        self.C = np.zeros((self.Dlines, self.Alines))
        if self.stage == 1:
            for j in range(self.natoms):
                for k in range(j + 1):
                    self.A[j][k] = np.dot(self.dist[j], self.dist[k])

        # Symmetric matrix -> copy diagonal elements, add total charge restrain
        for j in range(self.natoms):
            for k in range(j):
                self.A[k][j] = self.A[j][k]
            self.A[self.natoms][j] = 1.0
            self.A[j][self.natoms] = 1.0

            # Matrix element B see notesd
            for k in range(self.natoms):
                for j in range(self.natoms):
                    self.B[k][j] = np.multiply(np.dot(self.dist[k], self.dist_x[j]), self.e_x[j])  # B1
                    self.B[k][self.natoms + j] = np.multiply(np.dot(self.dist[k], self.dist_y[j]), self.e_y[j])  # B2
                    self.B[k][2 * self.natoms + j] = np.multiply(np.dot(self.dist[k], self.dist_z[j]),
                                                                 self.e_z[j])  # B3

        if self.mode != 'q':
            # matrix element C see notes
            # matrix element C
            for j in range(self.natoms):
                for k in range(self.natoms):
                    self.C[k][j] = np.multiply(np.dot(self.dist[j], self.dist_x[k]), self.e_x[k])
                    self.C[self.natoms + k][j] = np.multiply(np.dot(self.dist[j], self.dist_y[k]), self.e_y[k])
                    self.C[2 * self.natoms + k][j] = np.multiply(np.dot(self.dist[j], self.dist_z[k]), self.e_z[k])
            # Polarizaton matrix : the large form
            for j in range(self.natoms):
                for k in range(self.natoms):
                    self.D[k][j] = np.multiply(np.dot(self.dist_x[k], self.dist_x[j]), self.e_x[k] * self.e_x[j])
                    self.D[j + self.natoms][k] = self.D[k][j + self.natoms] = np.multiply(
                        np.dot(self.dist_x[k], self.dist_y[j]), self.e_x[k] * self.e_y[j])
                    self.D[j + 2 * self.natoms][k] = self.D[k][j + 2 * self.natoms] = np.multiply(
                        np.dot(self.dist_x[k], self.dist_z[j]), self.e_x[k] * self.e_z[j])
                    self.D[k + self.natoms][j + self.natoms] = np.multiply(np.dot(self.dist_y[k], self.dist_y[j]),
                                                                           self.e_y[k] * self.e_y[j])
                    self.D[j + 2 * self.natoms][k + self.natoms] = self.D[k + self.natoms][
                        j + 2 * self.natoms] = np.multiply(np.dot(self.dist_y[k], self.dist_z[j]),
                                                           self.e_y[k] * self.e_z[j])
                    self.D[k + 2 * self.natoms][j + 2 * self.natoms] = np.multiply(
                        np.dot(self.dist_z[k], self.dist_z[j]), self.e_z[k] * self.e_z[j])

        # Add charge restraints for equivalent atoms
        for j in range(len(self.eqatoms)):
            if self.eqatoms[j][1] > 0:
                self.A[self.natoms + 1 + j][self.eqatoms[j][0]] = 1
                self.A[self.natoms + 1 + j][self.eqatoms[j][1]] = -1
                self.A[self.eqatoms[j][0]][self.natoms + 1 + j] = 1
                self.A[self.eqatoms[j][1]][self.natoms + 1 + j] = -1
            elif self.eqatoms[j][1] < 0:
                self.A[self.natoms + 1 + j][self.eqatoms[j][0]] = 1
                self.A[self.eqatoms[j][0]][self.natoms + 1 + j] = 1

        # Add dipole restraints for equivalent atoms /only works for isotropic suff now
        for j in range(len(self.eqdipoles)):
            if self.eqdipoles[j][1] > 0:
                self.D[self.ndipoles + self.aniso_lines + j][self.eqdipoles[j][0]] = 1
                self.D[self.ndipoles + self.aniso_lines + j][self.eqdipoles[j][1]] = -1
                self.D[self.eqdipoles[j][0]][self.ndipoles + self.aniso_lines + j] = 1
                self.D[self.eqdipoles[j][1]][self.ndipoles + self.aniso_lines + j] = -1
            elif self.eqdipoles[j][1] < 0:
                self.D[self.ndipoles + self.aniso_lines + j][self.eqdipoles[j][0]] = 1
                self.D[self.eqdipoles[j][0]][self.ndipoles + self.aniso_lines + j] = 1

        # Add restraints for polarization isotropy
        # if self.aniso=='OFF':
        for j in range(self.natoms):
            self.D[3 * self.natoms + j][j] = self.D[j][3 * self.natoms + j] = 1.0
            self.D[3 * self.natoms + j][j + self.natoms] = self.D[j + self.natoms][3 * self.natoms + j] = -1.0
            self.D[4 * self.natoms + j][j] = self.D[j][4 * self.natoms + j] = 1.0
            self.D[4 * self.natoms + j][j + 2 * self.natoms] = self.D[j + 2 * self.natoms][4 * self.natoms + j] = -1.0

        for j in range(len(self.dipole_rst)):
            self.D[-1 - j][self.dipole_rst[j]] = 1
            self.D[self.dipole_rst[j]][-1 - j] = 1
        # Combine all matrices
        if self.mode == 'qd' or self.mode == 'd':
            self.X = np.concatenate(
                (np.concatenate((self.A, self.B), axis=1), np.concatenate((self.C, self.D), axis=1)), axis=0)
        if self.mode == 'q':
            self.X = np.concatenate((self.A, self.B), axis=1)
        self.stage = 1  # that was 2 before not sure again for what i used i DANGEROUS!!!  I assume for 2 stage resp fit

        if hasattr(self, 'wrst1'):
            for j in range(self.natoms):
                if self.atoms[j] != 'H' or self.freeh == True:
                    self.X[j][j] += self.wrst1 * 1. / np.sqrt(np.square(self.qd[j]) + 0.01)
                    # if self.mode !='q':
                    #    for k in range(3):
                    #        self.X[self.Alines+j+k*self.natoms][self.Alines+j+k*self.natoms]+=self.wrst1*self.dipscale/np.sqrt(np.square(self.qd[self.Alines+j+k*self.natoms])+0.01)

        if hasattr(self, 'hrst1'):
            if not hasattr(self, 'qrst'):
                self.qrst = np.linalg.solve(self.A, self.Y1)
            for j in range(self.natoms):
                if self.atoms[j] != 'H' or self.freeh == True:
                    self.X[j][j] += self.hrst1
                    # if self.mode !='q':
                    #    for k in range(3):
                    #        self.X[self.Alines+j+k*self.natoms][self.Alines+j+k*self.natoms]+=self.wrst1*5.0/np.sqrt(np.square(self.qd[self.Alines+j+k*self.natoms])+0.01)

    # only charge matrix with restraints
    def make_Aq(self):
        """
        Fast method for only optimizing charges.

        :return:
        """
        self.A = np.zeros((self.Alines, self.Alines))
        if self.stage == 1:
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
        for j in range(len(self.eqatoms)):
            if self.eqatoms[j][1] > 0:
                self.A[self.natoms + 1 + j][self.eqatoms[j][0]] = 1
                self.A[self.natoms + 1 + j][self.eqatoms[j][1]] = -1
                self.A[self.eqatoms[j][0]][self.natoms + 1 + j] = 1
                self.A[self.eqatoms[j][1]][self.natoms + 1 + j] = -1
            elif self.eqatoms[j][1] < 0:
                self.A[self.natoms + 1 + j][self.eqatoms[j][0]] = 1
                self.A[self.eqatoms[j][0]][self.natoms + 1 + j] = 1


    # X*alpha=Yn
    def make_Yn(self, mol2):
        """
        Creates vector Y for the BCC Pol method

        :param mol2: molecule object

        :return:
        """
        #Vector belonging to the BCCs
        self.Y1 = np.zeros(mol2.nbondtyps)
        if self.mode != 'alpha':
            for beta in range(mol2.nbondtyps):
                for k in range(self.natoms):
                    self.Y1[beta] += mol2.T[k][beta] * np.dot(self.pot, self.dist[k])
        else:
            self.Y1 = self.qbond

        #Vector belonging to the polarizabilities
        self.Y2 = np.zeros(mol2.lenpoltypes)
        if self.mode != 'bcconly':
            for beta in range(mol2.lenpoltypes):
                for k in range(self.natoms):
                    self.Y2[beta] += mol2.R[k][beta] * np.multiply(np.dot(self.pot, self.dist_x[k]), self.e_x[k])
                    self.Y2[beta] += mol2.R[k][beta] * np.multiply(np.dot(self.pot, self.dist_y[k]), self.e_y[k])
                    self.Y2[beta] += mol2.R[k][beta] * np.multiply(np.dot(self.pot, self.dist_z[k]), self.e_z[k])
        else:
            self.Y2 = self.pol
        self.Y = np.concatenate((self.Y1, self.Y2))


    def get_e_int(self, ):
        """
        Calculates the electric field at every atomic positions.
        :return:
        """
        self.e_x_old = copy.copy(self.e_x)
        self.e_y_old = copy.copy(self.e_y)
        self.e_z_old = copy.copy(self.e_z)
        self.dipole = self.qd[self.Alines:self.Alines + self.ndipoles]
        self.dipole[np.where(self.dipole == 0.0)] += 10E-10

        # Load permanent charges for BCC method
        # for alpha qfix is set to 0.0 in the wrapper program
        if self.mode == 'bcc' or self.mode == 'alpha' or self.mode == 'bcconly' or self.mode=='analysis'  or self.mode=='analysisalpha'  or self.mode=='alphabccfix' or self.mode=='d':
            try:
                self.qfix
            except:
                log.error('I do not have AM1-type charges')
                exit()
            else:
                for j in range(self.natoms):
                    self.e_x[j] = np.dot(np.multiply(self.qd[:self.natoms], self.scale[j]), self.adist_x[j]) + np.dot(
                        np.multiply(self.qfix[:self.natoms], self.scale[j]), self.adist_x[j]) + self.efield_ext[0]
                    self.e_y[j] = np.dot(np.multiply(self.qd[:self.natoms], self.scale[j]), self.adist_y[j]) + np.dot(
                        np.multiply(self.qfix[:self.natoms], self.scale[j]), self.adist_y[j]) + self.efield_ext[1]
                    self.e_z[j] = np.dot(np.multiply(self.qd[:self.natoms], self.scale[j]), self.adist_z[j]) + np.dot(
                        np.multiply(self.qfix[:self.natoms], self.scale[j]), self.adist_z[j]) + self.efield_ext[2]

        elif self.mode != 'q':
            for j in range(self.natoms):
                self.e_x[j] = np.dot(np.multiply(self.qd[:self.natoms], self.scale[j]), self.adist_x[j]) + \
                              self.efield_ext[0]
                self.e_y[j] = np.dot(np.multiply(self.qd[:self.natoms], self.scale[j]), self.adist_y[j]) + \
                              self.efield_ext[1]
                self.e_z[j] = np.dot(np.multiply(self.qd[:self.natoms], self.scale[j]), self.adist_z[j]) + \
                              self.efield_ext[2]
        if self.mode != 'q':
            self.e = np.concatenate((self.e_x, self.e_y, self.e_z))

            if self.SCF and self.step > 0:
                if not hasattr(self, 'Bdip') or self.thole :
                    if self.thole :
                        # self.thole_param=1.368711/BOHR**2
                        self.thole_param = 0.390
                        self.dipole_tmp = np.where(self.dipole < 0.0, -self.dipole, self.dipole)
                        self.thole_v = np.multiply(self.adist, np.float_power(
                            np.multiply(self.dipole_tmp[:self.natoms, None], self.dipole_tmp[:self.natoms]), 1. / 6))
                        di = np.diag_indices(self.natoms)
                        self.thole_v[di] = 1.0
                        self.thole_v = 1. / self.thole_v
                        self.thole_v[di] = 0.0

                        # Exponential thole
                        self.thole_fe = np.ones((self.natoms, self.natoms))
                        self.thole_ft = np.ones((self.natoms, self.natoms))
                        self.thole_fe -= np.exp(np.multiply(self.thole_param, np.power(-self.thole_v, 3)))
                        self.thole_ft -= np.multiply(np.multiply(self.thole_param, np.power(self.thole_v, 3)) + 1.,
                                                np.exp(np.multiply(self.thole_param, np.power(-self.thole_v, 3))))
                        # 1.5 was found in the OpenMM code. Not sure whuy it is there

                        # In original thole these lines should not be here
                        #self.thole_ft = np.multiply(self.thole_ft, self.scale)
                        #self.thole_fe = np.multiply(self.thole_fe, self.scale)
                        # Linear thole
                        # self.thole_fe=np.zeros((self.natoms,self.natoms))
                        # self.thole_fe=np.zeros((self.natoms,self.natoms))
                        # self.thole_fe=np.where(self.thole_v>1.0,1.0,4*np.power(self.thole_v,3)-3*np.power(self.thole_v,4))
                        # self.thole_ft=np.where(self.thole_v>1.0,1.0,np.power(self.thole_v,4))
                    else:
                        try:
                            self.thole_ft = self.scale_scf
                            self.thole_fe = self.scale_scf
                        except:

                            self.thole_ft = self.scale
                            self.thole_fe = self.scale
                        else:
                            print('Using different set of scaling for SCF interactions')
                            log.info('Using different set of scaling for SCF interactions')
                    self.Bdip11 = np.add(np.multiply(self.thole_fe, self.adist_3), np.multiply(self.thole_ft,
                                                                                               -3 * np.multiply(
                                                                                                   np.multiply(
                                                                                                       self.adistb_x,
                                                                                                       self.adistb_x),
                                                                                                   self.adist_5)))
                    self.Bdip22 = np.add(np.multiply(self.thole_fe, self.adist_3), np.multiply(self.thole_ft,
                                                                                               -3 * np.multiply(
                                                                                                   np.multiply(
                                                                                                       self.adistb_y,
                                                                                                       self.adistb_y),
                                                                                                   self.adist_5)))
                    self.Bdip33 = np.add(np.multiply(self.thole_fe, self.adist_3), np.multiply(self.thole_ft,
                                                                                               -3 * np.multiply(
                                                                                                   np.multiply(
                                                                                                       self.adistb_z,
                                                                                                       self.adistb_z),
                                                                                                   self.adist_5)))
                    self.Bdip12 = np.multiply(self.thole_ft,
                                              -3 * np.multiply(np.multiply(self.adistb_x, self.adistb_y), self.adist_5))
                    self.Bdip13 = np.multiply(self.thole_ft,
                                              -3 * np.multiply(np.multiply(self.adistb_x, self.adistb_z), self.adist_5))
                    self.Bdip23 = np.multiply(self.thole_ft,
                                              -3 * np.multiply(np.multiply(self.adistb_y, self.adistb_z), self.adist_5))
                    self.Bdip = np.concatenate((np.concatenate((self.Bdip11, self.Bdip12, self.Bdip13), axis=1),
                                                np.concatenate((self.Bdip12, self.Bdip22, self.Bdip23), axis=1),
                                                np.concatenate((self.Bdip13, self.Bdip23, self.Bdip33), axis=1)),
                                               axis=0)

                for j in range(self.natoms):
                    for k in range(3):
                        for l in range(3):
                            self.Bdip[k * self.natoms + j][l * self.natoms + j] = 0.0

                for j in range(3 * self.natoms):
                    self.Bdip[j][j] = 1. / self.dipole[j]
                self.dipole_scf = np.linalg.solve(self.Bdip, self.e)
                self.e = np.divide(self.dipole_scf, self.dipole[:self.ndipoles])
            self.e_x = 1.0 * self.e[:self.natoms] + 0. * self.e_x_old
            self.e_y = 1.0 * self.e[self.natoms:2 * self.natoms] + 0. * self.e_y_old
            self.e_z = 1.0 * self.e[2 * self.natoms:3 * self.natoms] + 0. * self.e_z_old
            self.step += 1

    def subtract_base_esp(self, esp_base):
        """
        Substract the ESP ground Potential from a polarized ESP.

        :param esp_base: pot of other esp object of the same molecule and conformation

        :return:
        """
        self.pot = np.add(self.pot, -esp_base.pot)

    def test_bcc(self, mol2):
        """Old stuff. Should not be necessary anymore. Delete the next time
        Replaced by bcc_to_qd"""
        self.bcc_to_qd(mol2)

    def bcc_to_qd(self, mol2):
        """
        Converts the optimized bccs and alphas to a qd object.

        :param mol2: molecule class object
        :return:
        """
        for i in range(self.natoms):
            self.qd[i] = np.dot(mol2.T[i], self.qbond)
        self.dipole = [np.dot(mol2.R[i], self.pol) for i in range(len(mol2.R))]
        tmp = np.concatenate((np.concatenate((self.dipole, self.dipole)), self.dipole))
        self.qd[self.Alines:self.Alines + self.ndipoles] = np.concatenate(
            (np.concatenate((self.dipole, self.dipole)), self.dipole))

    def add_molecule(self, esp2):
        """
        Combines the matrixes of 2 esp objects in the diagonal

        :param esp2: esp class object

        :return:
        Lagrange Multipliers have to be applied afterwords. Otherwise the optimisations
        are independent
        This function is only used for RESP-Pol
        """
        X12 = np.zeros((self.totallines, esp2.totallines))
        self.X = np.concatenate(
            (np.concatenate((self.X, X12), axis=1), np.concatenate((X12.transpose(), esp2.X), axis=1)), axis=0)
        self.Y = np.concatenate((self.Y, esp2.Y))
        self.atoms = self.atoms + esp2.atoms
        self.totallines = len(self.Y)

    def add_molecule_rst(self, rstlist):
        """
        Adds lagrange multipliers to the optimisation matrix
        :param rstlist: list [[int,int],[int,int]]
            molecule and atom ids which should be the same
        :return:

        This function ist only used for RESP-Pol
        """
        X12 = np.zeros((len(rstlist), self.totallines)).transpose()
        X22 = np.zeros((self.totallines + len(rstlist), len(rstlist))).transpose()
        self.X = np.concatenate((np.concatenate((self.X, X12), axis=1), X22), axis=0)
        Y2 = np.zeros(len(rstlist))
        self.Y = np.concatenate((self.Y, Y2))
        for i in range(len(rstlist)):
            self.X[i + self.totallines][rstlist[i][0]] = self.X[rstlist[i][0]][i + self.totallines] = 1
            self.X[i + self.totallines][rstlist[i][1]] = self.X[rstlist[i][1]][i + self.totallines] = -1
            # Point to add restraints for dipole to set to a default/initial value

    # Analysation stuff
    def calc_esp_qd(self, qd):
        """
        Calculates the ESP for a given set of point charges and polarizabilities.

        :param qd: list of float
            Point charges and polarizabilities
        :return:

        Calculates the ESP on every point of the initially read GESP file
        """
        self.q_pot = np.zeros(self.npoints)
        for i in range(self.npoints):
            if self.mode == 'q':
                self.q_pot[i] = np.dot(qd[:self.natoms], np.transpose(self.dist)[i])
            if self.mode == 'qd' or self.mode == 'd' or self.mode == 'bcc' or self.mode == 'alpha' or self.mode == 'bcconly' or self.mode == 'analysis'or self.mode == 'analysisalpha' or self.mode=='alphabccfix':
                self.dipole = qd[self.Alines:]
                # self.dipole[self.dipole<0.0]=0.0
                try:
                    # e_dip=np.dot(self.dipole_scf[:self.natoms],np.transpose(self.dist_x)[i])+np.dot(self.dipole_scf[self.natoms:2*self.natoms],np.transpose(self.dist_y)[i])+np.dot(self.dipole_scf[2*self.natoms:3*self.natoms],np.transpose(self.dist_z)[i])
                    e_dip = np.dot(np.multiply(self.dipole[:self.natoms], self.e_x),
                                   np.transpose(self.dist_x)[i]) + np.dot(
                        np.multiply(self.dipole[self.natoms:2 * self.natoms], self.e_y),
                        np.transpose(self.dist_y)[i]) + np.dot(
                        np.multiply(self.dipole[2 * self.natoms:3 * self.natoms], self.e_z),
                        np.transpose(self.dist_z)[i])
                    self.q_pot[i] = np.dot(qd[:self.natoms], np.transpose(self.dist)[i]) + e_dip
                except:
                    self.q_pot[i] = np.dot(qd[:self.natoms], np.transpose(self.dist)[i])
                    # print('DEBUG no electric field ')
            # if self.mode=='d':
            #    self.dipole=qd
            #    e_dip=np.dot(np.multiply(self.dipole[:self.natoms],self.e_x),np.transpose(self.dist_x)[i])+np.dot(np.multiply(self.dipole[self.natoms:2*self.natoms],self.e_y),np.transpose(self.dist_y)[i])+np.dot(np.multiply(self.dipole[2*self.natoms:3*self.natoms],self.e_z),np.transpose(self.dist_z)[i])
            #    self.q_pot[i]=e_dip

    def sub_esp_qd(self, qd):
        """
        Subtracts the ESP create by a set of point charges and polarizabilities.

        :param qd: list of float
            Point charges and polarizabilities
        :return:

        """
        if self.mode != 'alpha':  # Change that in bcc that this line is unnecessary
            self.calc_esp_qd(qd)
            self.pot = np.subtract(self.pot, self.q_pot)

    def calc_sse(self, qd):
        """
        Calculate the Squared Sum of Errors between the stored ESP and a ESP created from qd.
        :param qd: list of float
            Point charges and polarizabilities
        :return:
        """
        self.calc_esp_qd(qd)
        self.sse = np.square(self.pot - self.q_pot).sum()

    def write_res_esp(self, qd):
        """
        Writes the residual ESP to a file.

        :param qd: list of float
            Point charges and polarizabilities
        :return:
        """
        self.calc_esp_qd(qd)
        self.res_pot = self.pot - self.q_pot
        f = open('res_esp' + str(esp.res_esp_counter) + '.gesp', 'w')
        esp.res_esp_counter += 1
        f.write(' ESP FILE - ATOMIC UNITS\n')
        f.write(' CHARGE =  {0} - MULTIPLICITY =   1\n'.format(self.charge))
        f.write(' ATOMIC COORDINATES AND ESP CHARGES. #ATOMS =     {} \n'.format(np.sum(self.natoms)))
        for i in range(self.natoms):
            f.write(
                ' {} {} {} {} {}\n'.format(self.atoms[i], self.atomcrd[i][0], self.atomcrd[i][1], self.atomcrd[i][2],
                                           self.qd[i]))
        f.write(' ESP VALUES AND GRID POINT COORDINATES. #POINTS =   {}\n'.format(np.sum(self.npoints)))
        for i in range(self.npoints):
            f.write(' {} {} {} {}\n'.format(self.res_pot[i], self.crd[i][0], self.crd[i][1], self.crd[i][2]))
        f.close()

    def make_Y(self):
        """
        Creates the Vector Y for the RESP-Pol Method.
        :return:
        """
        self.Y1 = np.zeros(self.Alines)
        self.Y2 = np.zeros(self.Dlines)
        for k in range(self.natoms):
            self.Y1[k] = np.dot(self.pot, self.dist[k])
            self.Y1[self.natoms] = self.charge
        for k in range(len(self.eqatoms)):
            if self.eqatoms[k][1] > 0:
                self.Y1[self.natoms + 1 + k] = 0.0
            elif self.eqatoms[k][1] < 0:
                self.Y1[self.natoms + 1 + k] = self.qin[self.eqatoms[k][0]]
        for k in range(self.natoms):
            self.Y2[k] = np.multiply(np.dot(self.pot, self.dist_x[k]), self.e_x[k])
            self.Y2[k + self.natoms] = np.multiply(np.dot(self.pot, self.dist_y[k]), self.e_y[k])
            self.Y2[k + self.natoms * 2] = np.multiply(np.dot(self.pot, self.dist_z[k]), self.e_z[k])

        for k in range(len(self.eqdipoles)):
            if self.eqdipoles[k][1] > 0:
                self.Y2[self.ndipoles + self.aniso_lines + k] = 0.0
            elif self.eqdipoles[k][1] < 0:
                self.Y2[self.ndipoles + self.aniso_lines + k] = -self.eqdipoles[k][1]

        if self.mode == 'q':
            self.Y = self.Y1
        if self.mode == 'd':
            self.Y = np.concatenate((self.Y1, self.Y2))
        if self.mode == 'qd':
            self.Y = np.concatenate((self.Y1, self.Y2))
        if hasattr(self, 'hrst1') and hasattr(self, 'qrst'):
            for j in range(self.natoms):
                if self.atoms[j] != 'H' or self.freeh == True:
                    self.Y[j] += self.hrst1 * self.qrst[j]
                    print('DEBUG')

    def opt_scf(self):
        try:
            self.qd = np.linalg.solve(self.X, self.Y)
        except:
            self.q_tmp = np.linalg.solve(self.A, self.Y1)
            print(
                "Optimizing Dipoles and Charges results in a Singular Matrix --> One initial Optimization of charges only")
            self.qd[:self.Alines] = self.q_tmp
            self.get_e_int()
            self.make_Y()
            self.make_X()
            self.opt_scf()
            # self.dipoles_old=np.ones(self.ndipoles)

    # teststuff

    def make_test(self, qtest, dtest):
        self.qtest = []
        self.dtest = []
        if esp.test_counter == 0:
            f = open('test.d', 'w')
            g = open('test.q', 'w')
            for i in range(self.natoms):
                if i != self.natoms - 1:
                    x = random.random() * 2 - 1
                    d = np.random.rand(3) * -0.5
                else:
                    x = -np.sum(self.qtest)
                self.qtest.append(x)
                self.dtest.append(d)
                f.write('{}\n'.format(d))
                g.write('{}\n'.format(x))
            f.close()
            g.close()
        else:
            self.qtest = qtest
            self.dtest = dtest
        f = open('test' + str(esp.test_counter) + '.gesp', 'w')
        esp.test_counter += 1
        f.write(' ESP FILE - ATOMIC UNITS\n')
        f.write(' CHARGE =  {0} - MULTIPLICITY =   1\n'.format(int(np.sum(self.qtest))))
        f.write(' ATOMIC COORDINATES AND ESP CHARGES. #ATOMS =     {} \n'.format(np.sum(self.natoms)))
        for i in range(self.natoms):
            f.write(
                ' {} {} {} {} {}\n'.format(self.atoms[i], self.atomcrd[i][0], self.atomcrd[i][1], self.atomcrd[i][2],
                                           0.0000))
        f.write(' ESP VALUES AND GRID POINT COORDINATES. #POINTS =   {}\n'.format(np.sum(self.npoints)))
        for i in range(self.npoints):
            pot = 0.0
            for j in range(self.natoms):
                if self.mode == 'qd':
                    pot += self.qtest[j] * self.dist[j][i] + (
                                self.dtest[j][0] * self.dist_x[j][i] + self.dtest[j][1] * self.dist_y[j][i] +
                                self.dtest[j][2] * self.dist_z[j][i])
                if self.mode == 'd':
                    pot += (self.dtest[j][0] * self.dist_x[j][i] + self.dtest[j][1] * self.dist_y[j][i] + self.dtest[j][
                        2] * self.dist_z[j][i])
                if self.mode == 'q':
                    pot += self.qtest[j] * self.dist[j][i]
            pot += random.random() * 0.01
            f.write(' {} {} {} {}\n'.format(pot, self.crd[i][0], self.crd[i][1], self.crd[i][2]))
        f.close()

    # old stuff
    # Seems that I do not need that with the new implmentation
    # def update_q(self):
    #    self.qd_old=copy.deepcopy(self.qd)
    #    self.qd=np.linalg.solve(self.X2,self.Y)

    def stage2(self, eqatoms2):
        self.tmp = copy.copy(self.X)
        self.tmpY = copy.copy(self.Y)
        self.qin = self.qd
        self.make_X(eqatoms2)
        self.X[:self.natoms, :self.natoms] = self.tmp[:self.natoms, :self.natoms]
        self.make_Y(eqatoms2)
        self.Y[:self.natoms] = self.tmpY[:self.natoms]

    def restrain_neg(self):
        add = 0
        self.dipole = self.qd[self.Alines:]
        if self.aniso == 'OFF':
            for i in range(self.natoms):
                if self.dipole[i] < 0.0:
                    self.dipole_rst.append(i)
                    add += 1
        else:
            for i in range(self.ndipoles):
                if self.dipole[i] < 0.0:
                    self.dipole_rst.append(i)
                    add += 1

        self.Dlines = self.Dlines + add
        self.totallines = self.totallines + add

    def restrain_neg_other(self, other):
        add = 0
        other.dipole = other.qd[self.Alines:]
        for i in range(self.natoms):
            if other.dipole[i] < 0.0:
                if i not in np.array(self.eqdipoles).transpose()[1]:
                    self.dipole_rst.append(i)
                    add += 1
        self.Dlines = self.Dlines + add
        self.totallines = self.totallines + add

    def restrain_neg_other_mult(self, other):
        add = 0
        other_dipole = other[self.Alines:]
        other_ndipoles = len(other_dipole)
        for i in range(self.natoms):
            if other_dipole[i] < 0.0:
                if len(self.eqdipoles) != 0:
                    if i not in np.array(self.eqdipoles).transpose()[1]:
                        self.dipole_rst.append(i)
                        self.qd = np.append(self.qd, 0.0)
                        add += 1
        self.Dlines = self.Dlines + add
        self.totallines = self.totallines + add

    def make_Yq(self):
        self.Y1 = np.zeros(self.Alines)
        for k in range(self.natoms):
            self.Y1[k] = np.dot(self.pot, self.dist[k])
            self.Y1[self.natoms] = self.charge
        for k in range(len(self.eqatoms)):
            if self.eqatoms[k][1] > 0:
                self.Y1[self.natoms + 1 + k] = 0.0
            elif self.eqatoms[k][1] < 0:
                self.Y1[self.natoms + 1 + k] = self.qin[self.eqatoms[k][0]]

    def opt_bcc(self, mol2):  # I think I am not using this function at the moment
        self.qbond = np.linalg.solve(self.A, self.Y1)
        self.bcc_to_qd(mol2)

    """
    def update_X(self,conf,weight):
            self.wrst=weight
            self.X2=copy.deepcopy(self.X)
            for j in range(self.natoms):
                if self.atoms[j]!='H' or self.freeh==1:
                    self.X2[j][j]+=self.wrst*float(conf)/np.sqrt(np.square(self.qd[j])+0.01)
                    if self.mode !='q':
                        for k in range(3):
                            self.X2[self.Alines+j+k*self.natoms][self.Alines+j+k*self.natoms]+=self.wrst*5*float(conf)/np.sqrt(np.square(self.qd[self.Alines+j+k*self.natoms])+0.01)



    def make_Dold(self):
            self.D = np.zeros((self.ndipoles, self.ndipoles))
            for j in range(self.natoms):
                for k in range(self.natoms):
                    self.D[k][j] = np.dot(self.dist_x[k], self.dist_x[j])
                    self.D[j + self.natoms][k] = self.D[k][j + self.natoms] = np.dot(self.dist_x[k], self.dist_y[j])
                    self.D[j + 2 * self.natoms][k] = self.D[k][j + 2 * self.natoms] = np.dot(self.dist_x[k],
                                                                                             self.dist_z[j])
                    self.D[k + self.natoms][j + self.natoms] = np.dot(self.dist_y[k], self.dist_y[j])
                    self.D[j + 2 * self.natoms][k + self.natoms] = self.D[k + self.natoms][
                        j + 2 * self.natoms] = np.dot(self.dist_y[k], self.dist_z[j])
                    self.D[k + 2 * self.natoms][j + 2 * self.natoms] = np.dot(self.dist_z[k], self.dist_z[j])

    def update_A(self, conf):
            self.A2 = copy.deepcopy(self.A)
            for j in range(self.natoms):
                if self.atoms[j] != 'H':
                    self.A2[j][j] += wrst * float(conf) / np.sqrt(np.square(self.q[j]) + 0.01)

    def make_A_old(self, molecule):
            self.A = np.zeros((molecule.natoms + 1 + len(eqatoms)), molecule.natoms + 1 + len(eqatoms))
            for j in range(molecule.natoms):
                for k in range(molecule.natoms):
                    sum = 0.0
                    for i in range(self.npoints):
                        sum += 1 / (self.dist[i][j] * self.dist[i][k])
                    self.A[j][k] = sum
                self.A[molecule.natoms][j] = 1.0
                self.A[j][molecule.natoms] = 1.0

    def opt_old(self):
        if self.mode=='q':
            self.qd=self.multipoles(self.A,self.Y1)
        if self.mode=='d':
            self.qd=self.multipoles(self.D,self.Y2)
        if self.mode=='qd':
            self.qd=self.multipoles(self.X,self.Y)

    def make_B_old(self, molecule):
        self.B = np.zeros(molecule.natoms + 1)
        for k in range(molecule.natoms):
            sum = 0.0
            for i in range(self.npoints):
                sum += self.pot[i] / self.dist[i][k]
            self.B[k] = sum
            self.B[molecule.natoms] = self.charge

    def make_Aq(self,eqatoms):
        self.A=np.zeros((self.natoms+1+len(eqatoms),self.natoms+1+len(eqatoms)))
        for j in range(self.natoms):
            for k in range(j+1):
                self.A[j][k]=np.dot(self.dist[j],self.dist[k])
        #Symmetric matrix -> copy diagonal elements, add total charge restrain
        for j in range(self.natoms):
            for k in range(j):
                self.A[k][j]=self.A[j][k]
            self.A[self.natoms][j]=1.0
            self.A[j][self.natoms]=1.0
        #Add restraints for equivalent atoms
        for j in range(len(eqatoms)):
            self.A[self.natoms+1+j][eqatoms[j][0]]=1
            self.A[self.natoms+1+j][eqatoms[j][1]]=-1
            self.A[eqatoms[j][0]][self.natoms+1+j]=1
            self.A[eqatoms[j][1]][self.natoms+1+j]=-1



    def make_X_old(self,eqatoms):
        self.A=np.zeros((self.natoms+1+len(eqatoms),self.natoms+1+len(eqatoms)))
        self.B=np.zeros((self.Alines,self.Dlines))
        self.D=np.zeros((self.Dlines,self.Dlines))
        self.C=np.zeros((self.Dlines,self.Alines))
        if self.stage==1:
            for j in range(self.natoms):
                for k in range(j+1):
                    self.A[j][k]=np.dot(self.dist[j],self.dist[k])
        #Symmetric matrix -> copy diagonal elements, add total charge restrain
        for j in range(self.natoms):
            for k in range(j):
                self.A[k][j]=self.A[j][k]
            self.A[self.natoms][j]=1.0
            self.A[j][self.natoms]=1.0
        if self.mode!='q':
            #Matrix element B see notes
            for k in range(self.natoms):
                for j in range(self.natoms):
                    self.B[k][j]=np.dot(self.dist[k],self.dist_x[j]) #B1
                    self.B[k][self.natoms+j]=np.dot(self.dist[k],self.dist_y[j]) #B2
                    self.B[k][2*self.natoms+j]=np.dot(self.dist[k],self.dist_z[j]) #B3
            #matrix element C see notes
            #for d in (range.self.)
            #matrix element C
            for j in range(self.natoms):
                for k in range(self.natoms):
                    self.C[k][j]=np.dot(self.dist[j],self.dist_x[k])
                    self.C[self.natoms+k][j]=np.dot(self.dist[j],self.dist_y[k])
                    self.C[2*self.natoms+k][j]=np.dot(self.dist[j],self.dist_z[k])

            for j in range(self.natoms):
                for k in range(self.natoms):
                    self.D[k][j]=np.dot(self.dist_x[k],self.dist_x[j])
                    self.D[j+self.natoms][k]=self.D[k][j+self.natoms]=np.dot(self.dist_x[k],self.dist_y[j])
                    self.D[j+2*self.natoms][k]=self.D[k][j+2*self.natoms]=np.dot(self.dist_x[k],self.dist_z[j])
                    self.D[k+self.natoms][j+self.natoms]=np.dot(self.dist_y[k],self.dist_y[j])
                    self.D[j+2*self.natoms][k+self.natoms]=self.D[k+self.natoms][j+2*self.natoms]=np.dot(self.dist_y[k],self.dist_z[j])
                    self.D[k+2*self.natoms][j+2*self.natoms]=np.dot(self.dist_z[k],self.dist_z[j])

        #Add charge restraints for equivalent atoms
        for j in range(len(self.eqatoms)):
            if eqatoms[j][1]>0:
                self.A[self.natoms+1+j][self.eqatoms[j][0]]=1
                self.A[self.natoms+1+j][self.eqatoms[j][1]]=-1
                self.A[self.eqatoms[j][0]][self.natoms+1+j]=1
                self.A[self.eqatoms[j][1]][self.natoms+1+j]=-1
            elif eqatoms[j][1]<0:
                self.A[self.natoms+1+j][self.eqatoms[j][0]]=1
                self.A[self.eqatoms[j][0]][self.natoms+1+j]=1
        #Combine all matrices
        if self.mode=='qd':
            self.X=np.concatenate((np.concatenate((self.A,self.B),axis=1),np.concatenate((self.C,self.D),axis=1)),axis=0)
        if self.mode=='q':
            self.X=self.A
        if self.mode=='d':
            self.X=self.D
        self.stage=2

"""
