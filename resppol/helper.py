import sys
import numpy as np
import subprocess
import os

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