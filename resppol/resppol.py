"""
resppol.py
A tool for electrostatic fitting including polarization

Handles the primary functions
"""
import resppol.efield as efield


def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format)

    Replace this function and doc string for your own project

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


def add_func(a, b):
    result = a + b
    return result


def main():
    print(efield.efield(2))


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    main()
