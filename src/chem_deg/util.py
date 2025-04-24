from copy import copy

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges


def compute_partial_charges(compound: Chem.Mol) -> Chem.Mol:
    """
    Compute the partial charges of a compound using Gasteiger charges.

    Parameters
    ----------
    compound : Chem.Mol
        The compound to compute the partial charges for.

    Returns
    -------
    Chem.Mol
        The compound with partial charges added.
    """
    # Make a copy of the compound to avoid modifying the original
    compound = copy(compound)
    # Compute Gasteiger charges
    ComputeGasteigerCharges(compound)

    return compound


def annotate_partial_charges(compound: Chem.Mol) -> Chem.Mol:
    """
    Annotate the partial charges of a compound for visualization.

    Parameters
    ----------
    compound : Chem.Mol
        The compound to annotate the partial charges for.

    Returns
    -------
    Chem.Mol
        The compound with annotated partial charges.
    """
    # Make a copy of the compound to avoid modifying the original
    compound = copy(compound)

    # Check if partial charges have been computed
    if not compound.HasProp("_GasteigerCharge"):
        compound = compute_partial_charges(compound)

    # Annotate partial charges so they will show up on a 2D plot
    for atom in compound.GetAtoms():
        charge = atom.GetProp("_GasteigerCharge")
        charge = round(float(charge), 3)
        atom.SetProp("atomNote", str(charge))

    return compound


def annotate_atom_indexes(compound: Chem.Mol) -> Chem.Mol:
    """
    Annotate the atom indexes of a compound for visualization.

    Parameters
    ----------
    compound : Chem.Mol
        The compound to annotate the atom indexes for.

    Returns
    -------
    Chem.Mol
        The compound with annotated atom indexes.
    """
    # Make a copy of the compound to avoid modifying the original
    compound = copy(compound)

    # Annotate atom indexes so they will show up on a 2D plot
    for atom in compound.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))

    return compound


def draw_image(
    compound: Chem.Mol, size: tuple[int, int] = (300, 300), out_file: str = None
) -> str | bytes:
    """
    Draw the compound and save it to a file.

    Parameters
    ----------
    compound : Chem.Mol
        The compound to draw.
    out_file : str, optional
        The output file name. If None, the image will be returned as bytes.

    Returns
    -------
    str | bytes
        The output file name or the image as bytes.
    """

    # Draw the molecule with partial charges
    img = Draw.MolToImage(compound, size=size, kekulize=True, wedgeBonds=True)

    # Write to file if out_file is provided
    if out_file:
        with open(out_file, "wb") as f:
            img.save(f)

    return img
