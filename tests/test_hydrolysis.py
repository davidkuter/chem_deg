import pytest

from rdkit import Chem

from chem_deg.reactions.base import Reaction
from chem_deg.reactions.hydrolysis import (
    HalogenatedAliphaticsSubstitutionA,
    HalogenatedAliphaticsSubstitutionC,
    HalogenatedAliphaticsElimination,
    EpoxideHydrolysis,
    PhosphorusEsterHydrolysisAcid,
    PhosphorusEsterHydrolysisBase,
)


@pytest.mark.parametrize(
    "reaction",
    [
        HalogenatedAliphaticsSubstitutionA(),
        HalogenatedAliphaticsSubstitutionC(),
        HalogenatedAliphaticsElimination(),
        EpoxideHydrolysis(),
        PhosphorusEsterHydrolysisAcid(),
        PhosphorusEsterHydrolysisBase(),
    ],
)
def test_hydrolysis(reaction: Reaction):
    for reactant, expected_product in reaction.examples.items():
        products = reaction.react(reactant)
        products = [Chem.MolToSmiles(product) for product in products if product is not None]

        assert expected_product in products, (
            f"Expected {expected_product} in {products} for {reaction.name} reaction"
        )
