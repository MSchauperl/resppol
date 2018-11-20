"""
Unit and regression test for the resppol package.
"""

# Import package, test suite, and other packages as needed
import resppol
import pytest
import sys

def test_resppol_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "resppol" in sys.modules

def test_add():
    result=resppol.add_func(2,3)
    assert result == 5
