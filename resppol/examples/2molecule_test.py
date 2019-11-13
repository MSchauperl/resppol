#import resppol
import resppol.resppol
import os
from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.define('bohr = 0.52917721067 * angstrom')
dir(resppol.resppol)
resppol.resppol.__file__

alldirections=[['X+',[-0.01,0.0,0.0]],['X-',[0.01,0.0,0.0]],['Y+',[0.0,-0.01,0.0]],['Y-',[0.0,0.01,0.0]],['Z+',[0.0,0.0,-0.01]],['Z-',[0.0,0.0,0.01]]]
ROOT_DIR_PATH='/home/mschauperl/kirk/charge_method/medium_set/'
test = resppol.resppol.TrainingSet(mode='q_alpha',SCF= False, thole = False, FF='resppol/data/test_data/BCCelementPOL.offxml')


for nmol in range(1,2,1):
    print('Read in Molecule {}'.format(nmol))
    nconf=0
    datei = os.path.join(ROOT_DIR_PATH, 'molecule{}/conf0/mp2_0.mol2'.format(nmol))
    test.add_molecule(datei)
    directory=os.path.join(ROOT_DIR_PATH,'molecule{}'.format(nmol))
    for dire in os.listdir(directory):
        if os.path.isdir(os.path.join(directory,dire)):
            datei = os.path.join(ROOT_DIR_PATH, 'molecule{0}/conf{1}/mp2_{1}.mol2'.format(nmol,nconf))
            test.molecules[nmol-1].add_conformer_from_mol2(datei)
            espfile = os.path.join(ROOT_DIR_PATH, 'molecule{0}/conf{1}/molecule{1}.gesp'.format(nmol,nconf))
            test.molecules[nmol-1].conformers[nconf].add_baseESP(espfile)
            print("Read in {}".format(espfile))
            for direction,vector in alldirections:
                espfile = os.path.join(ROOT_DIR_PATH, 'molecule{}/conf{}/molecule_{}.gesp'.format(nmol,nconf,direction))
                test.molecules[nmol-1].conformers[nconf].add_polESP(espfile, e_field=Q_(vector, 'elementary_charge / bohr / bohr'))
                print("Read in {} {}".format(espfile,vector))
            nconf+=1
test.optimize_charges_alpha()
for molecule in test.molecules:
    print(molecule.q)
    print(molecule.alpha)
