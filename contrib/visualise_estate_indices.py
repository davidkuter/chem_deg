from rdkit import Chem
from rdkit.Chem import EState

from chem_deg.util import draw_image

smiles = [
    "CCOP(=S)(OCC)OC1=NC(=C(C=C1Cl)Cl)Cl",  # Chlorpyrifos
    # "OC1=NC(=C(C=C1Cl)Cl)Cl",  # Chlorpyrifos-product 1
    # "C(C)O",  # Chlorpyrifos-product 2 (Ethanol)
    "CNC(=O)CSP(=S)(OC)OC",  # Dimethoate
    "CNC(=O)CS",  # Dimethoate-product 1
    "CO",  # Dimethoate-product 2 (Methanol)
    "CCOP(=O)(NC(C)C)OC1=CC(=C(C=C1)SC)C",  # Fenamiphos
    "CC1=C(C=CC(=C1)OP(=S)(OC)OC)[N+](=O)[O-]",  # Fenthion
]

smiles = ["CCC(=O)N(C1=CC=CC=C1)C2(CCN(CC2)CCC(=O)OC)C(=O)OC", "CC(=O)OC1=CC=CC=C1C(=O)O"]
smiles = ["CC(=O)OC", "COC=O"]

for n, smi in enumerate(smiles):
    # Get partial charges
    mol = Chem.MolFromSmiles(smi)
    estate_indices = EState.EStateIndices(mol)

    # Annotate partial charges so they will show up on a 2D plot
    for atom, estate in zip(mol.GetAtoms(), estate_indices):
        atom.SetProp("atomNote", str(round(estate, 3)))

    # Draw the molecule with partial charges
    draw_image(mol, out_file=f"estate_indices_{n}.png", size=(500, 500))