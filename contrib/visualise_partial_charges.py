from rdkit import Chem

from chem_deg.util import compute_partial_charges, draw_image

smiles = [
    "CCOP(=S)(OCC)OC1=NC(=C(C=C1Cl)Cl)Cl",  # Chlorpyrifos
    # "OC1=NC(=C(C=C1Cl)Cl)Cl",  # Chlorpyrifos-product 1
    # "C(C)O",  # Chlorpyrifos-product 2 (Ethanol)
    "CNC(=O)CSP(=S)(OC)OC",  # Dimethoate
    "CNC(=O)CS", # Dimethoate-product 1
    "CO",  # Dimethoate-product 2 (Methanol)
    "CCOP(=O)(NC(C)C)OC1=CC(=C(C=C1)SC)C",  # Fenamiphos
    "CC1=C(C=CC(=C1)OP(=S)(OC)OC)[N+](=O)[O-]",  # Fenthion
]

for n, smi in enumerate(smiles):
    # Get partial charges
    mol = Chem.MolFromSmiles(smi)
    charge_mol = compute_partial_charges(mol)

    # Annotate partial charges so they will show up on a 2D plot
    for atom in charge_mol.GetAtoms():
        charge = atom.GetProp("_GasteigerCharge")
        charge = round(float(charge), 3)
        atom.SetProp("atomNote", str(charge))

    # Draw the molecule with partial charges
    draw_image(charge_mol, out_file=f"partial_charges_{n}.png", size=(500, 500))
