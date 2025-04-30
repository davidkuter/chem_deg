from chem_deg.kinetics.halflife import HALFLIFE4
from chem_deg.reactions.base import Reaction


class LactamHydrolysisFour(Reaction):
    """
    Hydrolysis of four-membered ring lactams.
    """

    def __init__(self):
        super().__init__(
            name="Lactam Hydrolysis (Four-membered ring)",
            reaction_smarts="[#6;R:2]1[#7:1][#6;R:4](=[O:5])[#6;R:3]~1>>[#7:1][#6:2]~[#6:3][#6:4](=[O:5])[OH]",
            examples={
                # Examples from the EPA
                "O=C1CCN1c1ccc([N+](=O)[O-])cc1": "O=C(O)CCNc1ccc([N+](=O)[O-])cc1",
            },
            halflife5=HALFLIFE4,
            halflife7=HALFLIFE4,
            halflife9=HALFLIFE4,
        )


class LactamHydrolysisFive(Reaction):
    """
    Hydrolysis of five-membered ring lactams.
    """

    def __init__(self):
        super().__init__(
            name="Lactam Hydrolysis (Five-membered ring)",
            reaction_smarts="[#6;R:2]1[#7:1][#6;R:4](=[O:5])[#6;R:6][#6;R:3]~1>>[#7:1][#6:2]~[#6:3][#6:6][#6:4](=[O:5])[OH]",
            examples={
                # Examples from the EPA
                "O=C1CCCN1c1cccc([N+](=O)[O-])c1": "O=C(O)CCCNc1cccc([N+](=O)[O-])c1",
            },
            halflife5=HALFLIFE4,
            halflife7=HALFLIFE4,
            halflife9=HALFLIFE4,
        )


class LactamHydrolysisSix(Reaction):
    """
    Hydrolysis of six-membered ring lactams.
    """

    def __init__(self):
        super().__init__(
            name="Lactam Hydrolysis (Six-membered ring)",
            reaction_smarts="[#6;R:2]1[#7:1][#6;R:4](=[O:5])[#6;R:6][#6;R:7][#6;R:3]~1>>[#7:1][#6:2]~[#6:3][#6;R:7][#6:6][#6:4](=[O:5])[OH]",
            examples={
                # Examples from the EPA
                "O=C1C[C@@H]2OCC=C3CN4CC[C@]56c7ccccc7N1[C@H]5[C@H]2[C@H]3C[C@H]46": "O=C(O)C[C@@H]1OCC=C2CN3CC[C@]45c6ccccc6N[C@H]4[C@H]1[C@H]2C[C@H]35",  # noqa: E501
            },
            halflife5=HALFLIFE4,
            halflife7=HALFLIFE4,
            halflife9=HALFLIFE4,
        )
