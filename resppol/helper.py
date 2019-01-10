#!/usr/bin/env python
# version 2018/07/03

import sys
from typing import List
import os
import numpy as np
import time
import copy
from resppol.esp_qalpha import esp, molecule
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
                    if mode!='q':
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
                    if mode!='d':
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
                    if mode !='d':
                        rst_inter_pol.append([l0, l1])
        if len(pol_dict)==5:
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





def get_respinput(mol2file,charge):
    basename=mol2file[:-5]
    ac_file=mol2file[:-5]+'.ac'
    r_file=[mol2file[:-5]+'.resp1', mol2file[:-5]+'.resp2']

    #p1='antechamber -i {} -fi mol2 -o {} -fo ac -nc {}'.format(mol2file,ac_file,int(charge))
    #os.system(p1)
    #p2='respgen -i {} -o {} -f resp1'.format(ac_file,r_file[0])
    #os.system(p2)
    #p3='respgen -i {} -o {} -f resp2'.format(ac_file,r_file[1])
    #os.system(p3)

    eqatoms=[[],[]]
    for k in range(2):
        f=open(r_file[k],'r')
        lines=f.readlines()[1:]
        for i in range(len(lines)):
            if 'Resp charges' in lines[i]:
                atomnum=int(lines[i+1].split()[1])
                for j in range(atomnum):
                    entry0=int(lines[i+2+j].split()[0])
                    entry1=int(lines[i+2+j].split()[1])
                    if entry1==0:
                        pass
                    elif entry1<0:
                        eqatoms[k].append([j,-99])
                    elif entry1>0:
                        eqatoms[k].append([entry1-1,j])
        f.close()

    #r1='rm {} {} {}'.format(ac_file,r_file[0],r_file[1])
    #os.system(r1)
    return(eqatoms)


def get_respinputh2o(mol2file,charge):
    basename=mol2file[:-5]
    ac_file=mol2file[:-5]+'.ac'
    r_file=[mol2file[:-5]+'.resp1', mol2file[:-5]+'.resp2']

    p1='antechamber -i {} -fi mol2 -o {} -fo ac -nc {}'.format(mol2file,ac_file,int(charge))
    os.system(p1)
    p2='respgen -i {} -o {} -f resp1'.format(ac_file,r_file[0])
    os.system(p2)
    p3='respgen -i {} -o {} -f resp2'.format(ac_file,r_file[1])
    os.system(p3)

    eqatoms=[[],[]]
    for k in range(2):
        f=open(r_file[k],'r')
        lines=f.readlines()[1:]
        for i in range(len(lines)):
            if 'Resp charges' in lines[i]:
                atomnum=int(lines[i+1].split()[1])
                for j in range(atomnum):
                    entry0=int(lines[i+2+j].split()[0])
                    entry1=int(lines[i+2+j].split()[1])
                    if entry1==0:
                        pass
                    elif entry1<0:
                        eqatoms[k].append([j,-99])
                    elif entry1>0:
                        eqatoms[k].append([entry1-1,j])
        f.close()

    r1='rm {} {} {}'.format(ac_file,r_file[0],r_file[1])
    os.system(r1)
    return(eqatoms)
def readqd(qf):
    f=open(qf,'r')
    lines=f.readlines()
    mode='na'
    for i in range(len(lines)):
        if 'Atoms' in lines[i]:
            natoms=int(lines[i].split()[1])
        if 'Charge' in lines[i]:
            mode='q'
            q=np.zeros(natoms)
            for j in range(natoms):
                q[j]=lines[j+i+1].split()[2]

        if 'Dipoles' in lines[i]:
            if mode=='q':
                mode='qd'
            else:
                mode='d'
            dip=np.array((natoms,3))
            for j in range(natoms):
                    for k in range(3):
                        dip[j][k]=lines[i+1+j].split()[k+2]
    return(q)

def readlog(qf,natoms=0):
    f=open(qf,'r')
    lines=f.readlines()
    mode='na'
    for i in range(len(lines)):
        if 'NAtoms' in lines[i]:
            natoms=int(lines[i].split()[1])
        if 'Mulliken charges:' in lines[i]:
            mode='q'
            q=np.zeros(natoms)
            for j in range(natoms):
                q[j]=lines[j+i+2].split()[2]
        if 'ESP charges:' in lines[i]:
            mode='q'
            q=np.zeros(natoms)
            for j in range(natoms):
                q[j]=lines[j+i+2].split()[2]
        if 'Hirshfeld charges,' in lines[i]:
            mode='q'
            q=np.zeros(natoms)
            for j in range(natoms):
                q[j]=lines[j+i+2].split()[2]
        if 'Lowdin Atomic Charges:' in lines[i]:
            mode='q'
            q=np.zeros(natoms)
            for j in range(natoms):
                q[j]=lines[j+i+2].split()[2]
        if 'Job cpu time:' in lines[i]:
            entry=lines[i].split()
            time=float(entry[9])+float(entry[7])*60+float(entry[5])*60*60+float(entry[3])*60*60*24
    return(q)

def readmul(qf,natoms=0):
    f=open(qf,'r')
    lines=f.readlines()
    mode='na'
    for i in range(len(lines)):
        if 'NAtoms' in lines[i]:
            natoms=int(lines[i].split()[1])
        if 'Mulliken charges:' in lines[i]:
            mode='q'
            q=np.zeros(natoms)
            for j in range(natoms):
                q[j]=lines[j+i+2].split()[2]
    return(q)


def loadbcc(datei):
    f=open(datei,'r')
    lines=f.readlines()
    f.close()
    bond=np.zeros(len(bondtypes))
    pol=[]
    readout=0
    for line in lines:
        entry=line.split()
        if entry[0]=='Bond':
            readout=1
        elif entry[0]=='Polarizabilities:':
            readout=2
        elif readout==1:
            for i in range(len(bondtypes)):
                if bondnames[i][0]==entry[1] and bondnames[i][1]==entry[2]:
                    bond[i]=entry[0]
        elif readout==2:
            pol.append(float(entry[0]))
    return(bond, pol)
