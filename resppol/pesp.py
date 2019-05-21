#!/usr/bin/env python
# version 2018/07/03

import sys
from typing import List
import os
import numpy as np
import time
import copy
from esp_qalpha import esp, molecule
import logging as log
import datetime

stime = time.time()
BOHR = float(0.52917722086)


def read_groups_intra(group_file, n_mol):
    '''
    Read in the intramoleuclar charge and polarization restraints

    :param group_file: string
        Path to group definiton and rst file
    :param n_mol:
        Number of molecules in test set
    :return: list of list of[int,int], list of list [int,int]
    For every molecule a list of equally treated atoms (same charge, same polarization, respectively)
    '''
    f = open(group_file, 'r')
    lines = f.readlines()
    f.close()
    rst_intra_pol = [[] for i in range(n_mol)]
    rst_intra_chg = [[] for i in range(n_mol)]
    for i, line in enumerate(lines):
        if 'Intramolecular Polarisation' in line:
            n_intra_pol = int(line.split()[-1])
            for j in range(n_intra_pol):
                entry = lines[j + 1 + i].split()
                if n_mol > int(entry[0]) == int(entry[2]):
                    l0 = int(entry[1])
                    try:
                        l1 = int(entry[3])
                    except:
                        l1 = float(entry[3])
                    if mode != 'q':
                        rst_intra_pol[int(entry[0])].append([l0, l1])
        elif 'Intramolecular Charge' in line:
            n_intra_chg = int(line.split()[-1])
            for j in range(n_intra_chg):
                entry = lines[j + 1 + i].split()
                if n_mol > int(entry[0]) == int(entry[2]):
                    l0 = int(entry[1])
                    try:
                        l1 = int(entry[3])
                    except:
                        l1 = float(entry[3])
                    if mode != 'd':
                        rst_intra_chg[int(entry[0])].append([l0, l1])
    return rst_intra_pol, rst_intra_chg


def read_groups_inter(group_file, Alines, startlines):
    """
    List of atoms (not in the same molecule) restraint to the same polarazibility.
    :param group_file:
    :param Alines:
    :param startlines:
    :return: list of [int,int]
    Returns the absolute number of the polarization which have to be restrained.
    """
    f = open(group_file, 'r')
    lines = f.readlines()
    f.close()
    pol_dict = []
    rst_inter_pol = []
    for i, line in enumerate(lines):
        if 'Intermolecular Polarisation' in line:
            n_inter_pol = int(line.split()[-1])
            for j in range(n_inter_pol):
                entry = lines[i + 1 + j].split()
                if int(entry[0]) < len(Alines) and int(entry[2]) < len(Alines):
                    l0 = startlines[int(entry[0])] + Alines[int(entry[0])] + int(entry[1])
                    l1 = startlines[int(entry[2])] + Alines[int(entry[2])] + int(entry[3])
                    if mode != 'q':
                        rst_inter_pol.append([l0, l1])
                if [int(entry[0]), int(entry[1])] not in pol_dict:
                    pol_dict.append([int(entry[0]), int(entry[1])])
        if 'Intermolecular Charges' in line:
            n_inter_pol = int(line.split()[-1])
            for j in range(n_inter_pol):
                entry = lines[i + 1 + j].split()
                if int(entry[0]) < len(Alines) and int(entry[2]) < len(Alines):
                    l0 = startlines[int(entry[0])] + int(entry[1])
                    l1 = startlines[int(entry[2])] + int(entry[3])
                    if mode != 'd':
                        rst_inter_pol.append([l0, l1])
        if len(pol_dict) == 5:
            pol_dict.append([int(7), int(4)])
    return rst_inter_pol, pol_dict


def readngesp(txtfile):
    """Reads in a file of the following format int file Ex Ey Ez (file).
    and gives back a list of files to open to read in alll gesp files and the list of external fields
    If specified additionally a reference  (base) esp can be read in for everz gesp"""
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


def read_nmol2(nmol2):
    """Reads in a list of mol2 files"""
    mol1 = []

    f = open(nmol2, 'r')
    l = f.readlines()
    for i in range(len(l)):
        if l[i] != '\n':
            mol1.append(molecule(l[i].strip()))
    return (mol1)


def read_restraints(txtfile, Alines, startlines):
    """Reads in the restraint between different molecules. Needs additonal the list at what line which molecule starts and at which line the dipoles start"""
    f = open(txtfile, 'r')
    lines = f.readlines()
    rst = []
    for line in lines:
        if line != '\n':
            entry = line.split()
            if int(entry[0]) < len(Alines) and int(entry[2]) < len(Alines):
                l0 = startlines[int(entry[0])] + Alines[int(entry[0])] + int(entry[1])
                l1 = startlines[int(entry[2])] + Alines[int(entry[2])] + int(entry[3])
                rst.append([l0, l1])
    return rst


# Setting default values
outputf = 'output.txt'
outputlog = 'output.log'
rst_on = 0
rst2_on = 0
nconf = []
mode = "d"
modetmp = ''
test = False
ngesp = []
SCF = False
dipscale = 0.01
wrst = ''
thole = False
groups = 'bond_rst_definitions.txt'
eqatoms = None
eqatoms2 = None
eqdipoles = None

# Read in input
for i in range(len(sys.argv)):
    if sys.argv[i] == '-mol2':  # old
        mol2 = sys.argv[i + 1]
    if sys.argv[i] == '-nmol2':
        nmol2 = sys.argv[i + 1]
    if sys.argv[i] == '-ngesp':
        gesp = sys.argv[i + 1]
        mconf = 1
    if sys.argv[i] == '-gesp':  # old
        gesp = sys.argv[i + 1]
        mconf = 0
    if sys.argv[i] == '-out':
        outputf = sys.argv[i + 1]
        outputlog = outputf.split('.')[0] + '.log'
    if sys.argv[i] == '-wrst':
        wrst = float(sys.argv[i + 1])
        rst_on = 1
    if sys.argv[i] == '-dipscale':
        dipscale = float(sys.argv[i + 1])
    if sys.argv[i] == '-wrst2':  # old
        wrst2 = float(sys.argv[i + 1])
        rst2_on = 1
    if sys.argv[i] == '-groups':
        groups = sys.argv[i + 1]
    if sys.argv[i] == '-eqatoms':  # old
        eqfile = sys.argv[i + 1]
    if sys.argv[i] == '-eqatoms2':  # old
        eqfile2 = sys.argv[i + 1]
    if sys.argv[i] == '-im_rst':  # old
        im_rstf = sys.argv[i + 1]
    if sys.argv[i] == '-test':
        test = True
    if sys.argv[i] == '-SCF':
        SCF = True
    if sys.argv[i] == '-thole':
        thole = True
    if sys.argv[i] == '-mode':
        if sys.argv[i + 1] == 'q':
            mode = 'q'
        if sys.argv[i + 1] == 'd':
            mode = 'd'
        if sys.argv[i + 1] == 'qd':
            mode = 'qd'
        if sys.argv[i + 1] == 'qa':
            mode = 'qa'
        if sys.argv[i + 1] == 'dnoq':
            mode = 'dnoq'
        if sys.argv[i + 1] == 'd_after_q':
            modetmp = 'daq'
            mode = 'qd'

# Setting up logger
log.basicConfig(filename=outputlog, level=10)
# print cmd line to logger
cmdline = ''
for i in range(len(sys.argv)):
    cmdline = cmdline + sys.argv[i].strip("\'") + ' '
log.info(cmdline)
log.info(str(datetime.datetime.now()))

# logging
if mode == 'qd':
    log.info('Optimizing Charges and Dipoles ')
elif mode == 'q':
    log.info('Optimizing Charges')
elif mode == 'd':
    log.info('Optimizing Dipoles ')
elif modetmp == 'daq':
    log.info('Sequential optimization of Charges and Dipoles ')

# Check if all files are specified
if 'gesp' not in globals():
    print(
        "Usage: %s -gesp <gesp-file>/-ngesp <textfilewithpathstogespfiles> [-wrst <float> -eqatoms <restraints.txt> -mode <q/d/qd> -out <oufile> -test -SCF] ".format(
            sys.argv[0]))
    exit()
else:
    print("Performing ESP fit")

if 'nmol2' in globals():
    mol1: List[molecule] = read_nmol2(nmol2)
else:
    print('Please specify nmol2 file')
    exit()

# Reads in all the gesp and base files
if 'gesp' in globals():
    ngesp, eext, base = readngesp(gesp)
else:
    print('Please specify ngesp file')
    exit()

# Reads in intramolecular restraints
#eqdipoles, eqatoms = read_groups_intra(groups, len(ngesp))
eqdipoles = [[]]
eqatoms = [[]]
# Load gesp files into esp objects
bases = [[] for i in range(len(base))]
esps = [[] for i in range(len(ngesp))]
for k in range(len(ngesp)):
    for i in range(len(base[k])):
        bases[k].append(esp(base[k][i], mol1[k], mode=mode))
    for i in range(len(ngesp[k])):
        esps[k].append(
            esp(ngesp[k][i], mol1[k], mode=mode, ext=eext[k][i], eqdipoles=eqdipoles[k], eqatoms=eqatoms[k], SCF=SCF,
                thole=thole, wrst1=wrst, dipscale=dipscale))
    if len(bases[k]) == len(esps[k]):
        log.info("Substracting a reference ESP for molecule {}".format(k))
        for i in range(len(ngesp[k])):
            esps[k][i].subtract_base_esp(bases[k][i])
    elif len(bases[k]) == 0:
        pass
    else:
        print("Something terrible is happening right now\n")
    nconf.append(len(ngesp[k]))
    log.info('Using {} molecule-conformations'.format(len(ngesp[k])))

# Main part of the program


# Making a test run
for k in range(len(ngesp)):
    if test:
        for i in range(nconf[k]):
            esps[k][i].distances()
            esps[k][i].make_test(esps[0][0].qd)
        log.info('Making a test example')
        exit()

    # Calculate Distances and Scaling for every gesp
    for i in range(nconf[k]):
        esps[k][i].distances()
        esps[k][i].scaling(mol1[k].bonds)

esptot = copy.copy(esps[0][0])
espmol = [None for i in range(len(nconf))]
if modetmp == 'daq':
    log.info('Mode is daq')
    for k in range(len(nconf)):
        for i in range(nconf[k]):
            esps[k][i].mode = 'q'
            esps[k][i].make_X()
            esps[k][i].make_Y()
            # esps[i].opt_scf()
        # Combine all conformers of one molecule
        espmol[k] = copy.deepcopy(esps[k][0])
        for j in range(nconf[k] - 1):
            espmol[k].X = np.add(espmol[k].X, esps[k][j + 1].X)
            espmol[k].Y = np.add(espmol[k].Y, esps[k][j + 1].Y)
        qd = np.linalg.solve(espmol[k].X, espmol[k].Y)
        for i in range(nconf[k]):
            esps[k][i].qin = np.zeros(len(qd))
        for i in range(nconf[k]):
            esps[k][i].sub_esp_qd(qd)
            esps[k][i].mode = 'd'
            esps[k][i].qfix = qd
            esps[k][i].charge = 0.0
            esps[k][i].update_for_daq()
        espmol[k].mode = 'd'
    esptot.mode = 'd'

# Stores the start and the total number of lines for every sub-matrix (molecule)
totallines = 0
startlines = []

allAlines = []  # Number of lines for the charge only matrix
for k in range(len(nconf)):
    startlines.append(totallines)
    allAlines.append(esps[k][0].Alines)
    totallines += esps[k][0].totallines
    if modetmp != 'daq':
        for i in range(nconf[k]):
            esps[k][i].qfix = np.zeros(esps[k][0].natoms)
startlines.append(totallines)
esptot = copy.copy(esps[0][0])
esptot.qd = np.zeros(totallines)
espmol = [None for i in range(len(nconf))]

im_rst, poltypes = read_groups_inter(groups, allAlines, startlines)

if mode == 'analysis':
    for k in range(len(nconf)):
        f = open('./charges/' + bccpol + '_' + str(k), 'w')

counter = 0
while True:
    if mode == 'analysis':
        break
    qd_old = copy.copy(esptot.qd)
    for k in range(len(nconf)):
        for i in range(nconf[k]):
            # Copy the result from the previous run to all molecules
            esps[k][i].qd = copy.copy(esptot.qd[startlines[k]:startlines[k] + esps[k][i].totallines])
    for k in range(len(nconf)):
        for i in range(nconf[k]):
            # Update the electric field and the matrixes
            esps[k][i].get_e_int()
            esps[k][i].make_X()
            esps[k][i].make_Y()
        # Combine all conformers of one molecule
        espmol[k] = copy.copy(esps[k][0])
        for j in range(nconf[k] - 1):
            espmol[k].X = np.add(espmol[k].X, esps[k][j + 1].X)
            espmol[k].Y = np.add(espmol[k].Y, esps[k][j + 1].Y)
        # Combine all molecules
        if k == 0:
            esptot = copy.copy(espmol[0])
            esptot.X = esptot.X / nconf[k]
            esptot.Y = esptot.Y / nconf[k]
        else:
            espmol[k].X = espmol[k].X / nconf[k]
            espmol[k].Y = espmol[k].Y / nconf[k]
            esptot.add_molecule(espmol[k])

    esptot.add_molecule_rst(im_rst)
    esptot.opt_scf()

    # Check for convergence
    converged = True
    for k in range(len(nconf)):
        if np.abs(qd_old[startlines[k]:esps[k][0].natoms + startlines[k]] - esptot.qd[startlines[k]:esps[k][0].natoms +
                                                                                                    startlines[
                                                                                                        k]]).sum() > 0.00001:
            converged = False
        if mode != 'q':
            if np.abs(qd_old[esps[k][0].Alines + startlines[k]:esps[k][0].ndipoles + esps[k][0].Alines + startlines[
                k]] - esptot.qd[esps[k][0].Alines + startlines[k]:esps[k][0].ndipoles + esps[k][0].Alines +
                                                                  startlines[k]]).sum() > 0.01:
                converged = False
    counter += 1
    if converged or counter > 1000:
        break

log.info("Number of iterativ cycles required: {}".format(counter + 1))
if SCF:
    log.info("Including dipole dipole polarization")
else:
    log.info("Neglecting dipole dipole polarization")

etime = time.time()
timeing = etime - stime
output = open(outputf, 'w')
output.write(str(datetime.datetime.now()) + '\n')
output.write(str(cmdline) + '\n')

# Writing out the errors for every molecule in the testset
ssetotal = 0.0
summe = 0.0  # stores the initial error
summe2 = 0.0  # store the residual error
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
        esps[k][i].calc_sse(esptot.qd[startlines[k]:startlines[k + 1]])
        tmp_sse = esps[k][i].sse
        esps[k][i].delete_dist()  # delete all distances
        log.info('The residual sum of square errors:  {0:10.8f}'.format(esps[k][i].sse))
        summe2 += esps[k][i].sse / nconf[k]
        sumk2 += esps[k][i].sse / nconf[k]
    output.write('{}\t{}\t{}\n'.format(k, sumk, sumk2))
output.write('{}\t{}\t{}\n'.format('SUM', summe, summe2))

# Writing out charges and polarizailities for the whole testset
output.write('Polarizabilities:\n')

if not os.path.exists('./charges'):
    os.makedirs('./charges')

for k in range(len(nconf)):
    # Generating and Writing Output Summary
    chargeout = open('./charges/' + outputf + '_' + str(k), 'w')
    log.info('Molekule ' + str(k + 1))
    if esptot.mode == 'q' or esptot.mode == 'qd':
        log.info('Sum of molecular charges: {}'.format(esptot.Y[esps[k][0].natoms + startlines[k]]))

    log.info('Total time for charge fitting: {0:5.3f}s'.format(timeing))

    if esptot.mode == 'q' or esptot.mode == 'qd' or esptot.mode == 'd':
        log.info('Atoms\t{}'.format(esps[k][0].natoms + startlines[k]))
        chargeout.write('Atoms\t{}\n'.format(esps[k][0].natoms))
        log.info('Charges\tAtomtype  \tOptQ   \tAvgQ  \tSDevQ  \tMinQ  \tMaxQ')
        chargeout.write('Charges\tAtomtype  \tOptQ   \tAvgQ  \tSDevQ  \tMinQ  \tMaxQ\n')
        # if modetmp =='daq':
        #    esptot.qd[startlines[k]:startlines[k]+esps[k][0].natoms]=esps[k][0].qfix[:esps[k][0].natoms]
        for i in range(esps[k][0].natoms):  # Achtung
            log.info('{0:6d}\t{1:>8s}\t{2:12.9f} '.format(i, esps[k][0].atoms[i], esptot.qd[i + startlines[k]]))
            chargeout.write(
                '{0:6d}\t{1:>8s}\t{2:12.9f} \n'.format(i, esps[k][0].atoms[i], esptot.qd[i + startlines[k]]))
    if esptot.mode == 'd' or esptot.mode == 'qd':
        diptot = np.zeros(esps[k][0].natoms)
        for i in range(esps[k][0].natoms):
            tmp_dipoles = esptot.qd[startlines[k] + esps[k][0].Alines + i]
            diptot[i] = tmp_dipoles

            # Writing the unique polarizabilities to the output file
            for o in range(len(poltypes)):
                if k == poltypes[o][0] and i == poltypes[o][1]:
                    output.write('{}\t{}\n'.format(diptot[i], mol1[k].atomtyps[i]))
        log.info('Dipoles Atomtype  alpha   ')
        for i in range(esps[k][0].natoms):
            log.info('{0:6d} {1:>8s}  {2:12.9f}'.format(i, esps[k][0].atoms[i], diptot[i]))
    etime = time.time()
    timeing = etime - stime
    chargeout.close()
    log.info('Total time for fitting: {}'.format(timeing))
output.close()
