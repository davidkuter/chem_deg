from chem_deg.kinetics.halflife import HALFLIFE2, HALFLIFE5
from chem_deg.reactions.base import Reaction


class ImideHydrolysisFive(Reaction):
    """
    Hydrolysis of five-membered ring imides.
    """

    def __init__(self):
        super().__init__(
            name="Imide Hydrolysis (Five-membered ring)",
            reaction_smarts="[#6:6]1[#6:7](=[#8:8])[#7:2][#6:3](=[#8:4])[#6,#7,#8:5]~1>>[OH][#6:7](=[#8:8])[#6:6]~[#6,#7,#8:5][#6:3](=[#8:4])[#7:2]",
            examples={
                # Examples from the EPA
                "CC(C)NC(=O)N1CC(=O)N(c2cc(Cl)cc(Cl)c2)C1=O": "CC(C)NC(=O)N(CC(=O)O)C(=O)Nc1cc(Cl)cc(Cl)c1",  # noqa: E501
                "O=C(O)c1ccccc1N1C(=O)c2ccccc2C1=O": "O=C(O)c1ccccc1NC(=O)c1ccccc1C(=O)O",
                "C=CC1(C)OC(=O)N(c2cc(Cl)cc(Cl)c2)C1=O": "C=CC(C)(OC(=O)Nc1cc(Cl)cc(Cl)c1)C(=O)O",
            },
            halflife5=HALFLIFE2,
            halflife7=HALFLIFE5,
            halflife9=HALFLIFE5,
        )


class ImideHydrolysisSix(Reaction):
    """
    Hydrolysis of six-membered ring imides.
    """

    def __init__(self):
        super().__init__(
            name="Imide Hydrolysis (Six-membered ring)",
            reaction_smarts="[#6:6]1[#6:7](=[#8:8])[#7:2][#6:3](=[#8:4])[#6,#7,#8:5][#6,#7,#8:9]~1>>[OH][#6:7](=[#8:8])[#6:6]~[#6,#7,#8:9][#6,#7,#8:5][#6:3](=[#8:4])[#7:2]",
            examples={
                # Examples from the EPA
                "O=C1c2cccc3cccc(c23)C(=O)N1c1ccccc1": "O=C(Nc1ccccc1)c1cccc2cccc(C(=O)O)c2-1",
            },
            halflife5=HALFLIFE2,
            halflife7=HALFLIFE5,
            halflife9=HALFLIFE5,
        )
