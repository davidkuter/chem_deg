"""
This code is based on the Abiotic Hydrolysis Reaction Library from the EPA. The reaction smarts were determined from the
schemas provided here: https://qed.epa.gov/static_qed/cts_app/docs/Hydrolysis%20Lib%20HTML/HydrolysisRxnLib_ver1-8.htm
"""

from rdkit import Chem

from chem_deg.reactions.base import Reaction

halogenated_aliphatics_substitution_a = Reaction(
    name="Halogenated Aliphatics Substitution A",
    reaction_smarts="[C;X4;!$(C([F,Cl,Br,I])([F,Cl,Br,I])):1][Cl,Br,I:2]>>[C:1][OH:2]",
    examples={
        # Examples from the EPA
        "CBr": "COH",

        # Example to test reaction occurs only on the terminal halogen
        "C(CC(Cl)(Cl)Cl)Cl": "C(CC(Cl)(Cl)Cl)OH"
    }
)

halogenated_aliphatics_substitution_c = Reaction(
    name="Halogenated Aliphatics Substitution C",
    reaction_smarts="[C;X4;$(C([F,Cl,Br,I])([F,Cl,Br,I]));!$(C([F,Cl,Br,I])([F,Cl,Br,I])([F,Cl,Br,I])):1]([*:2])([*:3])([*:4])[Cl,Br,I:5]>>[C:1]([*:2])([*:3])([*:4])[OH:5]",
    examples={
        "C(C)(C)(F)Cl": "C(C)(C)(F)OH"
    }
)


if __name__ == "__main__":

    reaction_type = halogenated_aliphatics_substitution_a
    print(reaction_type.name)
    for reactant, product in reaction_type.examples.items():
        print(f"  Reactant: {reactant}")
        products = reaction_type.react(reactant)
        if products is None:
            print("  No products")
        else:
            print(f"  Products: {[Chem.MolToSmiles(product) for product in products]}")
