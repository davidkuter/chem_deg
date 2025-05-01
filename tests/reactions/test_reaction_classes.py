from rdkit import Chem

from chem_deg.reactions.reaction_classes import Hydrolysis

def test_hydrolysis():
    """
    Test the hydrolysis reactions.
    """
    hydrolysis = Hydrolysis()
    for reaction in hydrolysis.reactions:
        for reactant, expected_product in reaction.examples.items():
            products = reaction.react(reactant)
            products = [Chem.MolToSmiles(product) for product in products if product is not None]

            assert expected_product in products, (
                f"Expected {expected_product} in {products} for {reaction.name} reaction"
            )