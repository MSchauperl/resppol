#!/usr/bin/env python

#=============================================================================================
# MODULE DOCSTRING
#=============================================================================================
"""
This module should convert a mol2 file int a Molecule object and expose all necessary information
.. todo::
   * Load in the molecule
   * Determine the equivalent atoms in the molecule
   * Determine the bonds based on a smirnoff input file
"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from openeye import oechem


from simtk import unit

#=============================================================================================
# GLOBAL PARAMETERS
#=============================================================================================

#=============================================================================================
# PRIVATE SUBROUTINES
#=============================================================================================

#=============================================================================================
# Molecule
#=============================================================================================


class Molecule():
    """
    This class loads in a mol2 file and expose all relevant information.
    """
    def __init__(self,datei):

        #Initialize OE Molecule
        mol = oechem.OEGraphMol()

        # Open Input File Stream
        ifs = oechem.oemolistream(datei)

        # Read from IFS to molecule
        oechem.OEReadMol2File(ifs, mol)


class Atom():
    """

    """
    pass


class Bond():
    pass