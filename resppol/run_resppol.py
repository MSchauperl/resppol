#!/usr/bin/env python

# =============================================================================================
# MODULE DOCSTRING
# =============================================================================================
"""
RESPOL is a program to fit charges or bond charge corrections and polarizabilities to ESPs of 
a molecule. 

This is the main part of the program and calls all othe subroutines:

1.) Reading in all given options from the given input file
	A detailed description of all input files is given in parser.py
	The basic concepts of options parsing was taken from Lee-Ping's ForceBalance

"""

# =============================================================================================
# GLOBAL IMPORTS
# =============================================================================================
import sys
import logging as log
from resppol.parser import parse_input

# =============================================================================================
# GLOBAL PARAMETERS
# =============================================================================================

# Read in input
for i in range(len(sys.argv)):
    if sys.argv[i] == '-in':  # Options for RESPPOL
        nmol2 = sys.argv[i + 1]

options = parse_input(input)
