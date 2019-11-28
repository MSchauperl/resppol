from openeye import oechem
import sys


atomic_number={'H':1,
               'C':6,
               'N':7,
               'O':8}

def read_mulliken_from_g09(g09_output):
    """
    Searches a gaussian output file for the Mulliken charge block and return the element and charge of every atom

    :param g09_output: Gaussian output file
    :return: List of [element,charge]
    """
    f=open(g09_output)
    lines=f.readlines()
    f.close()
    element_charge_array=[]
    for linenum,line in enumerate(lines):
        if 'Mulliken charges' in line:
            while True:
                if 'Sum' in lines[linenum+2]:
                    break
                else:
                    num,element,charge=lines[linenum+2].split()
                    element_charge_array.append([element,charge])
                    linenum+=1
            return(element_charge_array)



def load_g09_charges_to_oemol(oemol, g09_output, mol2_output=None):
    """
    This functions takes a oemol and a link to a gaussian output file and matches the charges to the mol2 file.
    Optional a new mol2 file with updated charges can be created with the optional keyword mol2_output.

    :param oemol: openeye molecule object
    :param g09_output: gaussian output file
    :param mol2_output: mol2 output file with updated charges (optional)

    :return: True
    """

    #1.) Read in Mulliken charge from a gaussian output file.
    element_charges=read_mulliken_from_g09(g09_output)
    #2.) Check if structure in gaussian output file is the same as in the mol2 file
    # This comparison is just based on elements are therefore not ideal
    for i,atom in enumerate(oemol.GetAtoms()):
        if oemol.NumAtoms() != len(element_charges):
            raise LookupError('Molecule and gaussian output file do not match')
        if atom.GetAtomicNum == atomic_number[element_charges[i][0]]:
            raise LookupError('Molecule and gaussian output file do not match')

    #3.) Assign Charges to oemol
    for i,atom in enumerate(oemol.GetAtoms()):
        atom.SetPartialCharge(float(element_charges[i][1]))

    #4.) Write mol2 output if requested.
    ofs = oechem.oemolostream()

    #Check if I can open the output file
    if mol2_output is not None:
        if ofs.open(mol2_output):
            #If yes I write the mol2 file
            ofs.SetFormat(oechem.OEFormat_MOL2)
            oechem.OEWriteMolecule(ofs, oemol)

        else:
            # If not, I throw an exception
            oechem.OEThrow.Fatal("Unable to create {}".format(mol2_output))


