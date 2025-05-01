"""
This code is based on the Abiotic Hydrolysis Reaction Library from the EPA. The reaction smarts were
determined from the schemas provided here:

https://qed.epa.gov/static_qed/cts_app/docs/Hydrolysis%20Lib%20HTML/HydrolysisRxnLib_ver1-8.htm
"""

from rdkit import Chem

from chem_deg.kinetics.halflife import (
    HALFLIFE1,
    HALFLIFE3,
    HALFLIFE4,
    HALFLIFE5,
    HALFLIFE6,
    HALFLIFE7,
)
from chem_deg.reactions.base import Reaction


class EpoxideHydrolysis(Reaction):
    def __init__(self):
        super().__init__(
            name="Epoxide Hydrolysis",
            reaction_smarts="[C:1]1[O:2][C:3]1>>[C:1]([OH:2])-[C:3]-[OH]",
            examples={
                # Examples from the EPA
                "C1CCC2OC2C1": "OC1CCCCC1O",
                "ClCC1CO1": "OCC(O)CCl",
                "ClC1=C(Cl)C2(Cl)C3C4CC(C5OC45)C3C1(Cl)C2(Cl)Cl": "OC1C(O)C2CC1C1C2C2(Cl)C(Cl)=C(Cl)C1(Cl)C2(Cl)Cl",  # noqa: E501
                "c1ccc2c(c1)CCC1OC21": "OC1CCc2ccccc2C1O",
            },
            halflife5=HALFLIFE7,
            halflife7=HALFLIFE5,
            halflife9=HALFLIFE1,
        )


class AmideHydrolysis(Reaction):
    """
    Hydrolysis of amides.
    """

    def __init__(self):
        super().__init__(
            name="Amide Hydrolysis",
            # [H;C;N] attached to [#7:4] is not specified because that would require an explicit
            # H atom to be present in the SMILES. I tried addingHs to the molecule but that caused
            # issues with other reactions so for now we will leave it out.
            reaction_smarts="[#6:3][#6:1](=[#8:2])[#7:4]>>[#6:3][#6:1](=[#8:2])[OH].[#7:4]",
            examples={
                # Examples from the EPA
                "CCNC(=O)[C@@H](C)OC(=O)Nc1ccccc1": "CCN.C[C@@H](OC(=O)Nc1ccccc1)C(=O)O",
                "C#CC(C)(C)NC(=O)c1cc(Cl)cc(Cl)c1": "C#CC(C)(C)N.O=C(O)c1cc(Cl)cc(Cl)c1",
                "CCC(=O)Nc1ccc(Cl)c(Cl)c1": "CCC(=O)O.Nc1ccc(Cl)c(Cl)c1",
                "O=C(NC(=O)c1c(F)cccc1F)Nc1ccc(Cl)cc1": "NC(=O)Nc1ccc(Cl)cc1.O=C(O)c1c(F)cccc1F",
            },
            halflife5=HALFLIFE1,
            halflife7=HALFLIFE1,
            halflife9=HALFLIFE3,
        )


class NitrileHydrolysis(Reaction):
    """
    Hydrolysis of nitriles.
    """

    def __init__(self):
        super().__init__(
            name="Nitrile Hydrolysis",
            reaction_smarts="[C:2](#[N:1])[#6,#7:3]>>[NH2:1][C:2](=[O])[#6,#7:3]",
            examples={
                # Examples from the EPA
                "CC#N": "CC(N)=O",
                "N#Cc1ccccc1": "NC(=O)c1ccccc1",
                "N#CC(Cl)(Cl)Cl": "NC(=O)C(Cl)(Cl)Cl",
                "N#CNC#N": "N#CNC(N)=O",
                "N#Cc1nn(-c2c(Cl)cc(C(F)(F)F)cc2Cl)c(N)c1S(=O)C(F)(F)F": "NC(=O)c1nn(-c2c(Cl)cc(C(F)(F)F)cc2Cl)c(N)c1S(=O)C(F)(F)F",  # noqa: E501
            },
            halflife5=HALFLIFE4,
            halflife7=HALFLIFE5,
            halflife9=HALFLIFE6,
        )


class NSHydrolysis(Reaction):
    """
    Hydrolysis of N-S bonds.
    """

    def __init__(self):
        super().__init__(
            name="N-S Hydrolysis",
            reaction_smarts="[#6:2][#7:1]([#6:3])[#16:4][#6,#7,#8:5]>>[#6:2][#7:1]([#6:3]).[OH][#16:4][#6,#7,#8:5]",
            examples={
                # Examples from the EPA
                # A number of examples have R1-N-S-N-R2 bonds which can cleave to give
                # R1-NH2 or R2-NH2. The code only expects 1 product so these examples have been
                # omitted. In the end, the product will undergo both N-S hydrolysis so it's not a
                # big deal.
                "O=C1C2CC=CCC2C(=O)N1SC(Cl)(Cl)Cl": "O=C1NC(=O)C2CC=CCC12.OSC(Cl)(Cl)Cl",
                "O=C1c2ccccc2C(=O)N1SC(Cl)(Cl)Cl": "O=C1NC(=O)c2ccccc21.OSC(Cl)(Cl)Cl",
            },
            halflife5=HALFLIFE5,
            halflife7=HALFLIFE5,
            halflife9=HALFLIFE5,
        )


class AcidHalideHydrolysis(Reaction):
    """
    Hydrolysis of acid halides.
    """

    def __init__(self):
        super().__init__(
            name="Acid Halidee Hydrolysis",
            reaction_smarts="[#6:2](=[#8:3])[F,Cl,Br,I]>>[#6:2](=[#8:3])[OH]",
            examples={
                # Examples from the EPA
                "O=CCl": "O=CO",
                "CC(=O)Cl": "CC(=O)O",
                "CC(C)OC(=O)Cl": "CC(C)OC(=O)O",
                "CSC(=O)Cl": "CSC(=O)O",
                "O=C(Cl)Cl": "O=C(O)Cl",
                "O=C(Cl)c1ccccc1": "O=C(O)c1ccccc1",
                "O=C(F)c1cccc(Cl)c1": "O=C(O)c1cccc(Cl)c1",
            },
            halflife5=HALFLIFE7,
            halflife7=HALFLIFE7,
            halflife9=HALFLIFE7,
        )


if __name__ == "__main__":
    reaction_type = AcidHalideHydrolysis()
    print(reaction_type.name)
    for reactant, product in reaction_type.examples.items():
        print(f"  Reactant: {Chem.MolToSmiles(Chem.MolFromSmiles(reactant))}")
        products = reaction_type.react(reactant)
        if products is None:
            print("  No products")
        else:
            print(f"  Products: {[Chem.MolToSmiles(product) for product in products]}")
