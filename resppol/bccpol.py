#!/usr/bin/python
# version 2018/07/03

import sys
from typing import Dict, Any, Union

import numpy as np
import time
import copy
from esp_qalpha import esp, molecule
from helper import readlog, readqd, readmul
from collections import Counter
import logging as log
import datetime

"""
This function is used to fit BCCs and polarizations.

Parameters:
-----------
ngesp
nmol2
out
-SCF
-thole
-bccpol
-charge
-dipscale
-wrst
-hrst
-zeromodel

Important varibles:
---------
    pol_groups: dict
        Definition which atom gets which pol parameter
    bond_groups: dict
        Definition how many BCCs and between which atoms.
    

The definition which atoms are allowed to have different polarization is currently
implemented in a way that is using AMBER atom types. This should be replaced
soon with a SMIRKS pattern. Similar is the definition for BCCs. BCCs are allowed
to be different between two atoms of different bond_groups. SMIRKS can help to
improve this scheme.

The module is using the esp and the molecule class. The molecule class is used
to read in the information about bond terms and are therefore used to 
define the scaling of the atoms and for the definitions of the bond in the BCCs
The esp class reads in one gaussian ESP file and creates the corresponding matrix 
for the optimisation.

This program is combining the matrix from multiple gaussian ESP and setting the corresponding 
values in the ESP class.
"""

# Definition of bond groups
"""
pol_groups: Dict[Union[str, Any], Union[int, Any]] = {
    'c': 0,
    'c2': 4,
    'c3': 0,
    'ca': 4,
    'cp': 4,
    'h1': 1,
    'h2': 1,
    'h4': 1,
    'ha': 1,
    'hc': 1,
    'hn': 1,
    'hx': 1,
    'ho': 1,
    'n': 2,
    'n4': 2,
    'na': 2,
    'o': 3,
    'oh': 3,
    'os': 3, }

bond_groups: Dict[Union[str, Any], Union[int, Any]] = {
    'c': 0,
    'c2': 1,
    'c3': 2,
    'ca': 3,
    'cp': 4,
    'h1': 5,
    'h2': 6,
    'h4': 7,
    'ha': 8,
    'hc': 9,
    'hn': 10,
    'hx': 11,
    'ho': 12,
    'n': 13,
    'n4': 14,
    'na': 15,
    'o': 16,
    'oh': 17,
    'os': 18, }
"""


def read_groups(group_file):
    '''
    Read in BCC and POL groups.
    :param group_file: string
        Path to file containing the group definition
    :return: bcc_dict: dictonary
                define which atom types are belonging together
            pol_dict: dictonary
                define which atom types have the same polarizability
    The groups define which atoms have the same polarizability and which bonds share the
    same bond charge correction.
    '''
    f = open(group_file, 'r')
    lines = f.readlines()
    f.close()
    bcc_dict = {}
    pol_dict = {}
    for i, line in enumerate(lines):
        if 'Polarisation groups' in line:
            n_pol = int(lines[i].split()[-1])
            for j in range(n_pol):
                key = lines[i + 1 + j].split()[0].strip(':').strip("'")
                value = int(lines[i + 1 + j].split()[1].strip(','))
                pol_dict[key] = value
        elif 'BCC groups' in line:
            n_bcc = int(lines[i].split()[-1])
            for j in range(n_bcc):
                key = lines[i + 1 + j].split()[0].strip(':').strip("'")
                value = int(lines[i + 1 + j].split()[1].strip(','))
                bcc_dict[key] = value
    return bcc_dict, pol_dict


# returns list of input file names
def read_nmol2(nmol2):
    """
    Takes a file and return the lines of this file in a list.

    :param nmol2: string
        Path to file with Path to all mol2 files of the set.
    :return: list of strings

    Used to write in the nmol2 files in this project
    """
    mol1 = []
    f = open(nmol2, 'r')
    l = f.readlines()
    for o in range(len(l)):
        if l[o] != '\n':
            mol1.append(molecule(l[o].strip()))
    return (mol1)


def readngesp(txtfile):
    """
    Reads in a ngesp file and stores the information.
    :param txtfile: string
        Path to file with ngeps file
    :return: list of string, list of [float, float, float], {list of string}

    This function does not read in the gesp files per se.
    It only reads in the Path of the gesp file and the
    corresponding electric fields. If a fifth argument is given
    in a line it corresponds to a baseline QM calculation which is substracted
    from the gesp given by the first argument.
    """
    f = open(txtfile, 'r')
    ngesp = []  # List of Paths to all gesp files
    eext = []
    lines = f.readlines()
    base = []
    molecules = []
    for line in lines:
        if line != '\n':
            entry = line.split()
            if entry[0] not in molecules:
                ngesp.append([])
                eext.append([])
                molecules.append(entry[0])
                base.append([])
            ngesp[int(entry[0])].append(entry[1])
            eext[int(entry[0])].append([float(entry[2]), float(entry[3]), float(entry[4])])
            if len(entry) == 6:
                base[int(entry[0])].append(entry[5])
    return (ngesp, eext, base)


def loadalpha(datei, npol):
    """
    Takes the output of an old  BCC-Pol calculation and reads in the values
    :param datei: string
        Path to file
    :return: list of float
        polarizations.

    Up to now there is no check if the definition of BCCs is the same
    Should be implemendet in the next version.

    """
    f = open(datei, 'r')
    lines = f.readlines()
    f.close()
    bond = np.zeros(len(bondtypes))
    pol = [0.0 for i in range(npol)]
    readout = 0
    for line in lines:
        entry = line.split()
        if entry[0] == 'Bond':
            readout = 1
        elif entry[0] == 'Polarizabilities:':
            readout = 2
        elif readout == 1:
            for o in range(len(bondtypes)):
                if bondnames[o][0] == entry[1] and bondnames[o][1] == entry[2]:
                    bond[o] = entry[0]
        elif readout == 2:
            num = pol_groups[entry[1]]
            pol[num] = float(entry[0])
    pol = np.array(pol)
    return pol


def loadbcc(datei):
    f = open(datei, 'r')
    lines = f.readlines()
    f.close()
    bond = np.zeros(len(bondtypes))
    readout = 0
    for line in lines:
        entry = line.split()
        if entry[0] == 'Bond':
            readout = 1

        elif entry[0] == 'Polarizabilities:':
            readout = 2
        elif readout == 1:
            for o in range(len(bondtypes)):
                if bondnames[o][0] == entry[1] and bondnames[o][1] == entry[2]:
                    bond[o] = entry[0]
    return bond


def loadbccpol(datei, npol):
    alpha = loadalpha(datei, npol)
    bond = loadbcc(datei)
    bondpol = np.concatenate((bond, alpha), axis=0)
    return bondpol


"""Constants"""
BOHR = float(0.52917722086)

# Timing
stime = time.time()  # Starttime
timef = open('timing.dat', 'w')

# Setting default values
outputf = 'output.txt'
outputlog = 'output.log'
path1 = './molecule'
ngesp = []
nconf = []
mode = 'bcc'
test = False
SCF = False
dipscale = 0.01
wrst = ''
thole = False
zeromodel = False
groups = None

# Read in input
for i in range(len(sys.argv)):
    if sys.argv[i] == '-nmol2':  # the usual input file format for the mol2
        nmol2 = sys.argv[i + 1]
    if sys.argv[i] == '-chg':  # prefix of the gaussain output for the atomic charges
        chg = sys.argv[i + 1]
    if sys.argv[i] == '-ngesp':  # list of esps have to be defined in this file with the Eext
        gesp = sys.argv[i + 1]
        mconf = 1
    if sys.argv[i] == '-out':  # Outfile
        outputf = sys.argv[i + 1]
        outputlog = outputf.split('.')[0] + '.log'
    if sys.argv[i] == '-mode':  # BCC or alpha(polarizaton only)
        mode = sys.argv[i + 1]
    if sys.argv[i] == '-SCF':
        SCF = True
    if sys.argv[i] == '-thole' or sys.argv[i] == '-Thole':
        thole = True
    if sys.argv[i] == '-dipscale':
        dipscale = float(sys.argv[i + 1])
    if sys.argv[i] == '-wrst':  # restraint constant for parabolic restraints
        wrst = float(sys.argv[i + 1])
        rst_on = 1
    if sys.argv[i] == '-bccpol':  # Read in starting values of a previous run
        bccpol = sys.argv[i + 1]
    if sys.argv[i] == '-startpol':  # Read in starting values of a previous run
        startpol = sys.argv[i + 1]
    if sys.argv[i] == '-startbcc':  # Read in starting values of a previous run
        startbcc = sys.argv[i + 1]
    if sys.argv[i] == '-groups':  # Read group defintion
        groups = sys.argv[i + 1]
    if sys.argv[i] == '-zeromodel':  # All charges qfix are set to zero
        zeromodel = True
    if sys.argv[i] == '-path':  # All charges qfix are set to zero
        path1 = sys.argv[i + 1].strip('/') + '/molecule'
# Finish reading commmand line arguments

log.basicConfig(filename=outputlog, level=10)
cmdline = ''
for i in range(len(sys.argv)):
    cmdline = cmdline + sys.argv[i].strip("\'") + ' '
log.info(cmdline)

if groups == None:
    log.error('Group Definition is missing')
    exit()
else:
    bond_groups, pol_groups = read_groups(groups)

# Debug
log.debug('Pol Groups:')
log.debug(pol_groups)
log.debug('BCC Groups:')
log.debug(bond_groups)

bondtypes = []  # used to save all occuring bond types: Every combination of different bond numbers is one bond type
bondnames = []  # just for output reason. So that I can easily print out which bond atoms are involved in this BCC

# Initalize molecules with bond matrix and atom types
if 'nmol2' in globals():
    mol1 = read_nmol2(nmol2)

    # Check what kinds of bonds exists in the whole test set
    for i in range(len(mol1)):
        mol1[i].bondtypes(bond_groups, bondtypes,
                          bondnames)  # the group definiton; the already occuring bonds and the corresponding bond names
        # print(bondtypes)

    # Build matrixes defining which bcc occurs in which atom and waht polarizability is used for which atom (R/T)
    for i in range(len(mol1)):
        mol1[i].bondmatrix(bondtypes)
        mol1[i].poltypes(pol_groups)

else:  # for single molecule use only
    print('nmol2 file is not definied.')
# Timing
time1 = time.time()
duration = time1 - stime
timef.write('Time for reading in mol2files: {}\n'.format(duration))

# Initialize all esps
ngesp, eext, base = readngesp(gesp)
bases = [[] for i in range(len(base))]
esps = [[] for i in range(len(ngesp))]
for k in range(len(ngesp)):
    for i in range(len(base[k])):
        bases[k].append(esp(base[k][i], mol1[k], mode=mode))
    for i in range(len(ngesp[k])):
        # esps.append(esp(ngesp[i], mode, aniso='OFF', ext=eext[i], eqdipoles=eqdipoles, eqatoms=eqatoms))
        esps[k].append(esp(ngesp[k][i], mol1[k], mode=mode, ext=eext[k][i], SCF=SCF, thole=thole, wrst1=wrst,
                           dipscale=dipscale))  # initilaze all esps
    if len(bases[k]) == len(esps[k]):
        for i in range(len(ngesp[k])):
            esps[k][i].subtract_base_esp(bases[k][i])
    elif len(bases[k]) == 0:
        pass
    else:
        log.error("Something terrible is happening right now\n")
    nconf.append(len(ngesp[k]))
    log.info('Using {} molecule-conformations'.format(len(ngesp[k])))
del bases

# Timing
time2 = time.time()
duration = time2 - time1
timef.write('Time for reading in espfiles: {}\n'.format(duration))

# Main part of the program

# Create scaling matrix and intialize necessary lists
for k in range(len(ngesp)):
    if test:  # Just for testing purposes
        for i in range(nconf[k]):
            esps[k][i].distances()
            esps[k][i].make_test(esps[0].testq, esps[0].testd)
        print('Making a test example')
        exit()
    for i in range(nconf[k]):  # Start
        esps[k][i].scaling(mol1[k].bonds)  # calculate all scaling matrix
Xmol = [None for i in range(len(nconf))]  # Stores all matrixes
Ymol = [None for i in range(len(nconf))]  # Stores all vectors of the euqation
q = [None for i in range(len(nconf))]  # Stores all atomic charges
qtmp = [None for i in range(len(nconf))]

# Timing Stuff
summe3 = 0.0  # Stores quadratic sum of all esp points
time3 = time.time()
duration = time3 - time2
timef.write('Time for calculation distances: {}\n'.format(duration))

# Read in Atomic charges. Most of the time from gaussian outputs Attetntion: Absolut Path
# Subtracts the ESP of the charges from the input ESP, except we are fitting to
# ESP differences
for k in range(len(nconf)):
    for i in range(nconf[k]):
        if 'Mulliken' in chg:
            try:
                q[k] = readmul(
                    path1 + str(k + 1) + '/conf0/' + chg + '.log',
                    natoms=esps[k][0].natoms)
            except:
                q[k] = readmul(
                    path1.replace('mschauperl', 'mis') + str(k + 1) + '/conf0/' + chg + '.log',
                    natoms=esps[k][0].natoms)
        elif 'RESP' in chg:
            chgfile = chg.split('=')[1]
            try:
                q[k] = readqd(
                    '/home/mschauperl/kirkwood/charge_method/medium_set/charges/' + chgfile + '.txt_' + str(k))
            except:
                q[k] = readqd('/home/mis/kirkwood/charge_method/medium_set/charges/' + chgfile + '.txt_' + str(k))

        elif 'zero_chg' in chg:
            q[k] = [0.0, 0.0]
        elif 'blaconf1' in chg:
            try:
                q[k] = readqd(
                    '/home/mschauperl/kirkwood/charge_method/medium_set/charges/RESPconf1_co_opt_direct.txt_' + str(k))
            except:
                q[k] = readqd(
                    '/home/mis/kirkwood/charge_method/medium_set/charges/RESPconf1_co_opt_direct.txt_' + str(k))
        elif 'resp' in chg:
            try:
                q[k] = readqd(
                    path1 + str(k + 1) + '/conf0/' + chg)
            except:
                q[k] = readqd(path1.replace('mschauperl', 'mis') + str(k + 1) + '/conf0/' + chg)

        else:
            try:
                q[k] = readlog(
                    path1 + str(k + 1) + '/conf0/' + chg + '.log',
                    natoms=esps[k][0].natoms)
            except:
                q[k] = readlog(
                    path1.replace('mschauperl', 'mis') + str(k + 1) + '/conf0/' + chg + '.log',
                    natoms=esps[k][0].natoms)
        if zeromodel:
            if esps[k][i].charge == 0:
                q[k] = np.zeros(len(q[k]))
            elif k == 4:
                q[k] = np.zeros(len(q[k]))
                q[k][6] = -0.5
                q[k][7] = -0.5
            elif k == 7:
                q[k] = np.zeros(len(q[k]))
                q[k][4] = 1.0

            else:
                q[k] = np.array([esps[k][i].charge / len(q[k]) for l in range(len(q[k]))])
        # qtmp[k]=readqd('/home/mis/data_kirkwood/charge_method/smallset/molecule'+str(k+1)+'/conf'+str(i)+'/MP2aug-cc-pVTZ'+str(i)+'.resp')
    """ 
    #q[k]=readqd('/home/mis/data_kirkwood/charge_method/smallset/molecule'+str(k+1)+'/conf0/MP2aug-cc-pVTZ0.resp')
    """

for k in range(len(nconf)):
    for i in range(nconf[k]):
        esps[k][i].distances()  # Calculated all distances
        esps[k][i].calc_sse(q[k])  # Calculate initital error/
        summe3 += esps[k][i].sse / nconf[k]  # Store overall error
        # Include charges from the lower QM calculation (AM1 calculation)
        if mode == 'analysisalpha' or mode == 'alpha':
            esps[k][i].qfix = np.zeros(len(q[k]))
            q[k] = np.zeros(len(q[k]))
            esps[k][i].step = 2
        else:
            esps[k][i].qfix = q[k]
        esps[k][i].sub_esp_qd(q[k])  # Substract static esp from the atomic charges
        esps[k][i].delete_dist()

# Finish reading input charges

# Timing stuff again
time4 = time.time()
duration = time4 - time3
timef.write('Time for substracting fixed charge esp {}\n'.format(duration))

# Read in starting values from an old run
if 'bccpol' in globals():
    try:
        npoltypes = len(Counter(pol_groups.values()))
        bcc_alp = loadbccpol(bccpol, npoltypes)
    except:
        print('Using the Zeromodel option')
    if mode == 'alpha' or mode == 'analysisalpha':
        bcc_alp[:len(bondtypes)] = np.zeros(len(bondtypes))
    qbond = bcc_alp[:len(bondtypes)]
    pol = bcc_alp[len(bondtypes):]

if 'startpol' in globals():
    npoltypes = len(Counter(pol_groups.values()))
    pol = loadalpha(startpol, npoltypes)
    if 'startbcc' not in globals():
        qbond = np.zeros(len(bondtypes))
        bcc_alp = np.concatenate((qbond, pol), axis=0)
if 'startbcc' in globals():
    qbond = loadbcc(startbcc)
    if 'startpol' not in globals():
        pol = np.array([0.0 for i in range(len(Counter(pol_groups.values())))])
    bcc_alp = np.concatenate((qbond, pol), axis=0)

# Optimization process
while True:
    # Timing stuff
    ecaltime = 0.0
    matrixtime = 0.0

    for k in range(len(nconf)):

        for i in range(nconf[k]):
            etime_s = time.time()
            """
            Initializing values for the next run. If no starting values are given. 
            Starting values are set to 0
            Calculating all necesaary distances and the electric field at every atom
            position
            """
            try:
                esps[k][i].qbond = bcc_alp[:len(bondtypes)]
                esps[k][i].pol = bcc_alp[len(bondtypes):]
                esps[k][i].bcc_to_qd(mol1[k])
            except:
                # esps[k][i].qd[:esps[k][i].natoms]=np.zeros(len(q[k]))
                esps[k][i].qbond = np.zeros(len(bondtypes))
                esps[k][i].pol = np.array([0.0 for i in range(len(Counter(pol_groups.values())))])
                esps[k][i].bcc_to_qd(mol1[k])

            log.debug(esps[k][i].qd)
            esps[k][i].distances()  # Calculated all distances
            if mode == 'alphabccfix':
                esps[k][i].sub_esp_qd(esps[k][i].qd)
                esps[k][i].qfix = esps[k][i].qfix + esps[k][i].qd[:esps[k][i].natoms]
                esps[k][i].mode = 'alpha'
                esps[k][i].qbond = np.zeros(len(bondtypes))

            esps[k][i].get_e_int()  # get initial electric field

    if mode == 'alphabccfix':
        mode = 'alpha'

    if mode == 'analysis' or mode == 'analysisalpha':
        """No optimization required. Just calculating the error with the given starting values """
        npoltypes = len(Counter(pol_groups.values()))  # Number of different polariztion groups
        break

    for k in range(len(nconf)):
        for i in range(nconf[k]):
            """
            Creating the matrixes X and Y for all GESP files used.
            Combine all matrixes for one molecule to Xmol and then combining
            all matrixes of all molecules to one total matrix Xtot.
            """
            # Timing stuff
            etime_e = time.time()
            ecaltime += etime_e - etime_s
            matrixtime_s = time.time()

            esps[k][i].make_Xn(mol1[k])  # Make Matrix
            esps[k][i].make_Yn(mol1[k])  # Make Vectors

            # Timing stuff
            matrixtime_e = time.time()
            matrixtime += matrixtime_e - matrixtime_s

            esps[k][i].delete_dist()  # Calculated all distances
        npoltypes = len(Counter(pol_groups.values()))  # Number of different polariztion groups

        # Initialize Molecule Matrix
        Xmol[k] = np.zeros(
            (len(bondtypes) + npoltypes, len(bondtypes) + npoltypes))  # stores the matrixes for molecule k
        Ymol[k] = np.zeros(len(bondtypes) + npoltypes)  # stores the vector Y for molecule k

        # Combines all conformations
        for j in range(nconf[k]):
            Xmol[k] = np.add(Xmol[k], esps[k][j].X)  # Add all matrices from molecule k
            Ymol[k] = np.add(Ymol[k], esps[k][j].Y)  # Add all vectors from molecule k

        # Combine all molecules
        if k == 0:
            Xtot = copy.copy(Xmol[0])
            Xtot = Xtot / nconf[k]  # Normalize by the number of conformations
            Ytot = copy.copy(Ymol[0])
            Ytot = Ymol[0] / nconf[k]
        else:
            Xmol[k] = Xmol[k] / nconf[k]
            Ymol[k] = Ymol[k] / nconf[k]
            Xtot = np.add(Xtot, Xmol[k])
            Ytot = np.add(Ytot, Ymol[k])

    # If one polarizability does not occur in the molecules the value is set to 0.0
    for i in range(len(Xtot)):
        if all(tmp == 0.0 for tmp in Xtot[i]):
            Xtot[i][i] = 1

    try:
        bcc_alp_old = copy.copy(bcc_alp)
    except:
        bcc_alp_old = np.zeros(len(Ytot))

    # Timing stuff
    time5 = time.time()
    duration = time5 - time4
    timef.write('Time for calculation matrixes: {}\n'.format(duration))
    timef.write('Time for calculation efield: {}\n'.format(ecaltime))
    timef.write('Time for calculation matrixes(pure): {}\n'.format(matrixtime))

    # Solve the equation
    bcc_alp = np.linalg.solve(Xtot, Ytot)
    log.debug("1 optimisation step")
    # Timing stuff
    time6 = time.time()
    duration = time6 - time5
    timef.write('Time for solve lin eq: {}\n'.format(duration))
    time4 = time.time()
    if np.abs(np.subtract(bcc_alp_old, bcc_alp)).sum() < 0.001:  # Determination condition
        break

# End of optimization


"""
This is all just output writing. 
"""
summe = 0.0  # stores the initial error
summe2 = 0.0  # store the residual error
outfile = open(outputf, 'w')
outfile.write(str(datetime.datetime.now()) + '\n')
outfile.write(str(cmdline) + '\n')
for k in range(len(nconf)):
    sumk = 0.0
    sumk2 = 0.0
    for i in range(nconf[k]):
        esps[k][i].distances()  # Calculated all distances

        summe += np.square(esps[k][i].pot).sum() / nconf[k]
        sumk += np.square(esps[k][i].pot).sum() / nconf[k]
        single = np.square(esps[k][i].pot).sum()
        log.info('Molekule ' + str(k + 1))
        log.info('The initial sum of square errors:  {0:5.3f}'.format(single))
        esps[k][i].qbond = bcc_alp[:len(bondtypes)]
        esps[k][i].pol = bcc_alp[len(bondtypes):]
        esps[k][i].bcc_to_qd(mol1[k])  # calculates also the new charges
        esps[k][i].calc_sse(esps[k][i].qd)
        tmp_sse = esps[k][i].sse
        esps[k][i].delete_dist()  # delete all distances
        log.info('The residual sum of square errors:  {0:10.8f}'.format(esps[k][i].sse))
        summe2 += esps[k][i].sse / nconf[k]
        sumk2 += esps[k][i].sse / nconf[k]
    outfile.write('{}\t{}\t{}\n'.format(k, sumk, sumk2))
outfile.write('{}\t{}\t{}\n'.format('SUM', summe, summe2))
"""print('DEBUG')
for i in range(3):
    for j in range(3):
        esps[0][0].qd[i+3*j]+=0.0001
esps[0][0].calc_sse(esps[0][0].qd)
print(esps[0][0].sse-tmp_sse)
for i in range(3):
    for j in range(3):
        esps[0][0].qd[i+3*j]-=0.0002
esps[0][0].calc_sse(esps[0][0].qd)
print(esps[0][0].sse-tmp_sse)
for i in range(3):
    for j in range(3):
        esps[0][0].qd[i+3*j]+=0.0001
"""

# Print out summary
log.info(summe)  # before bond charge correction and polarization
log.info(summe2)  # residual error
log.info(summe3)  # sum of squared ESP values
# Write out bond charge corrections
outfile.write('Bond Charge Corrections:\n')
for i in range(len(bondtypes)):
    outfile.write('{}\t{}\t{}\n'.format(bcc_alp[i], bondnames[i][0], bondnames[i][1]))
outfile.write('Polarizabilities:\n')
for i in range(npoltypes):
    outfile.write(
        '{}\t{}\n'.format(bcc_alp[i + len(bondtypes)], list(pol_groups.keys())[list(pol_groups.values()).index(i)]))
# if 'bccpol' in globals() and mode!='analysis':
#    outfile.write('{}\t{}\n'.format(esps[7][0].pol[4],'n4'))
outfile.close()

"""
this part was just to investigate if I am in a minimum or if the approximate analytical gradient does something bad.
for t in range(len(qbond)):
    qbond[t]+=0.01
    summet=0.0
    summe2t=0.0
    for k in range(len(nconf)):
            for i in range(nconf[k]):
                summet += np.square(esps[k][i].pot).sum()
                single=np.square(esps[k][i].pot).sum()
                #print('The initial sum of square errors:  {0:5.3f}'.format(single))
                esps[k][i].qbond=qbond
                esps[k][i].test_bcc(mol1[k])
                esps[k][i].calc_sse(esps[k][i].qd)
                #print('The residual sum of square errors:  {0:5.3f}'.format(esps[k][i].sse))
                summe2t+=esps[k][i].sse/nconf[k]
    print(summe2t-summe2)
    if summe2t-summe2<0.0:
        print('Warning')
    qbond[t]-=0.01
print('Finish')

for t in range(len(qbond)):
    qbond[t]-=0.01
    summet=0.0
    summe2t=0.0
    for k in range(len(nconf)):
            for i in range(nconf[k]):
                summet += np.square(esps[k][i].pot).sum()
                single=np.square(esps[k][i].pot).sum()
                #print('The initial sum of square errors:  {0:5.3f}'.format(single))
                esps[k][i].qbond=qbond
                esps[k][i].test_bcc(mol1[k])
                esps[k][i].calc_sse(esps[k][i].qd)
                #print('The residual sum of square errors:  {0:5.3f}'.format(esps[k][i].sse))
                summe2t+=esps[k][i].sse/nconf[k]
    print(summe2t-summe2)
    if summe2t-summe2<0.0:
        print('Warning')
        print(bondtypes[k])
    qbond[t]+=0.01
"""
"""
esps=[]
esps.append(esp(gesp,mode='q'))
esps[0].distances()
summe = 0.0
summe += np.square(esps[0].pot).sum()
print('The initial sum of square errors:  {0:5.3f}'.format(summe))
q=readlog('/home/mis/data_kirkwood/charge_method/smallset/molecule1/conf0/hf631AM1Mulliken.log',natoms=esps[0].natoms)
print(q)
esps[0].sub_esp_qd(q)
summe = 0.0
summe += np.square(esps[0].pot).sum()
print('The initial sum of square errors:  {0:5.3f}'.format(summe))
esps[0].make_A(mol1[0])
esps[0].make_Y1(mol1[0])
esps[0].opt_bcc(mol1[0])
print(bondtypes)
esps[0].calc_sse(esps[0].qd)
print(esps[0].sse)
"""

"""
sse0=esps[0].sse
print('TEST')
for i in range(len(esps[0].qbond)):
    esps[0].qbond[i]+=0.01
    esps[0].test_bcc(mol1[0])
    esps[0].calc_sse(esps[0].qd)
    print(esps[0].sse-sse0)
    esps[0].qbond[i]-=0.01
    esps[0].qbond[i]-=0.01
    esps[0].test_bcc(mol1[0])
    esps[0].calc_sse(esps[0].qd)
    print(esps[0].sse-sse0)
    esps[0].qbond[i]+=0.01
esps[0].make_X()
esps[0].make_Y()
esps[0].opt_scf()
print(esps[0].qd)
esps[0].calc_sse(esps[0].qd)
print(esps[0].sse)
"""
