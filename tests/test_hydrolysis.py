import pytest

from rdkit import Chem

from chem_deg.reactions.base import Reaction
from chem_deg.reactions import (
    HalogenatedAliphaticsSubstitutionA,
    HalogenatedAliphaticsSubstitutionC,
    HalogenatedAliphaticsElimination,
    EpoxideHydrolysis,
    PhosphorusEsterHydrolysisAcid,
    PhosphorusEsterHydrolysisBase,
    CarboxylateEsterHydrolysis,
    LactoneHydrolysisFour,
    LactoneHydrolysisFive,
    LactoneHydrolysisSix,
    CarbonateHydrolysisAcyclic,
    CarbonateHydrolysisCyclic,
    AnhydrideHydrolysisAcyclic,
    AnhydrideHydrolysisCyclicFive,
    AnhydrideHydrolysisCyclicSix,
    AmideHydrolysis,
    LactamHydrolysisFour,
    LactamHydrolysisFive,
    LactamHydrolysisSix,
    CarbamateHydrolysis,
    ThiocarbamateHydrolysis,
    UreaHydrolysisAcyclic,
    UreaHydrolysisCyclicFive,
    UreaHydrolysisCyclicSix,
    SulfonylureaHydrolysis,
    NitrileHydrolysis,
    NSHydrolysis,
    ImideHydrolysisFive,
    ImideHydrolysisSix,
    AcidHalideHydrolysis,
)


@pytest.mark.parametrize(
    "reaction",
    [
        HalogenatedAliphaticsSubstitutionA(),  # 0
        HalogenatedAliphaticsSubstitutionC(),  # 1
        HalogenatedAliphaticsElimination(),  # 2
        EpoxideHydrolysis(),  # 3
        PhosphorusEsterHydrolysisAcid(),  # 4
        PhosphorusEsterHydrolysisBase(),  # 5
        CarboxylateEsterHydrolysis(),  # 6
        LactoneHydrolysisFour(),  # 7
        LactoneHydrolysisFive(),  # 8
        LactoneHydrolysisSix(),  # 9
        CarbonateHydrolysisAcyclic(),  # 10
        CarbonateHydrolysisCyclic(),  # 11
        AnhydrideHydrolysisAcyclic(),  # 12
        AnhydrideHydrolysisCyclicFive(),  # 13
        AnhydrideHydrolysisCyclicSix(),  # 14
        AmideHydrolysis(),  # 15
        LactamHydrolysisFour(),  # 16
        LactamHydrolysisFive(),  # 17
        LactamHydrolysisSix(),  # 18
        CarbamateHydrolysis(),  # 19
        ThiocarbamateHydrolysis(),  # 20
        UreaHydrolysisAcyclic(),  # 21
        UreaHydrolysisCyclicFive(),  # 22
        UreaHydrolysisCyclicSix(),  # 23
        SulfonylureaHydrolysis(),  # 24
        NitrileHydrolysis(),  # 25
        NSHydrolysis(),  # 26
        ImideHydrolysisFive(),  # 27
        ImideHydrolysisSix(),  # 28
        AcidHalideHydrolysis(),  # 29
    ],
)
def test_hydrolysis(reaction: Reaction):
    for reactant, expected_product in reaction.examples.items():
        products = reaction.react(reactant)
        products = [Chem.MolToSmiles(product) for product in products if product is not None]

        assert expected_product in products, (
            f"Expected {expected_product} in {products} for {reaction.name} reaction"
        )
