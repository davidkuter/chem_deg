from rdkit import Chem

from chem_deg.util import (
    compute_partial_charges,
    annotate_atom_indexes,
    annotate_partial_charges,
    draw_image,
)


def test_compute_partial_charges():
    """
    Test the compute_partial_charges function.
    """
    # Create a simple molecule (ethanol)
    smiles = "CCO"
    compound = Chem.MolFromSmiles(smiles)

    # Compute partial charges
    compound_with_charges = compute_partial_charges(compound)

    # Check if partial charges are present
    for atom in compound_with_charges.GetAtoms():
        assert atom.HasProp("_GasteigerCharge")
        assert isinstance(float(atom.GetProp("_GasteigerCharge")), float)


def test_annotate_partial_charges():
    """
    Test the annotate_partial_charges function.
    """
    # Create a simple molecule (ethanol)
    smiles = "CCO"
    compound = Chem.MolFromSmiles(smiles)

    # Annotate partial charges
    compound_with_annotations = annotate_partial_charges(compound)

    # Check if annotations are present
    for atom in compound_with_annotations.GetAtoms():
        assert atom.HasProp("atomNote")
        assert isinstance(float(atom.GetProp("atomNote")), float)


def test_annotate_atom_indexes():
    """
    Test the annotate_atom_indexes function.
    """
    # Create a simple molecule (ethanol)
    smiles = "CCO"
    compound = Chem.MolFromSmiles(smiles)

    # Annotate atom indexes
    compound_with_annotations = annotate_atom_indexes(compound)

    # Check if annotations are present
    for atom in compound_with_annotations.GetAtoms():
        assert atom.HasProp("atomNote")
        assert isinstance(int(atom.GetProp("atomNote")), int)


def test_draw_image():
    """
    Test the draw_image function.
    """
    # Create a simple molecule (ethanol)
    smiles = "CCO"
    compound = Chem.MolFromSmiles(smiles)

    # Draw the image
    img = draw_image(compound, size=(200, 200))

    # Check if the image is not None
    assert isinstance(img, bytes)
