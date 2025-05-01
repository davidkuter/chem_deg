from chem_deg.kinetics.halflife import HALFLIFE1, HALFLIFE3, HALFLIFE5
from chem_deg.reactions.base import Reaction


class CarbonateHydrolysisAcyclic(Reaction):
    """
    Hydrolysis of acyclic carbonates.
    """

    def __init__(self):
        super().__init__(
            name="Carbonate Hydrolysis (Acyclic)",
            reaction_smarts="[C:2][O:1][#6;!R,!a](=[O])[O:3][C:4]>>[C:2][OH:1].[OH:3][C:4]",
            examples={
                # Examples from the EPA
                "CCOC(=O)OC1=C(c2cc(C)ccc2C)C(=O)NC12CCC(OC)CC2": "CCO.COC1CCC2(CC1)NC(=O)C(c1cc(C)ccc1C)=C2O",  # noqa: E501
            },
            halflife5=HALFLIFE1,
            halflife7=HALFLIFE3,
            halflife9=HALFLIFE5,
        )


class CarbonateHydrolysisCyclic(Reaction):
    """
    Hydrolysis of cyclic carbonates. Currently, only 5-membered cyclic carbonates are supported
    since these are apparently the most common.
    """

    def __init__(self):
        super().__init__(
            name="Carbonate Hydrolysis (Cyclic)",
            reaction_smarts="[C;R:2]1[O;R:1][#6;R](=[O])[O;R:3][C;R:4]~1>>[OH:1][C:2]~[C:4][OH:3]",
            examples={
                # No EPA examples
                "CC1COC(=O)O1": "CC(O)CO",
            },
            halflife5=HALFLIFE1,
            halflife7=HALFLIFE3,
            halflife9=HALFLIFE5,
        )
