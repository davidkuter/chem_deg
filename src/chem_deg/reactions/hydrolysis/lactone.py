from chem_deg.kinetics.halflife import HALFLIFE1, HALFLIFE5
from chem_deg.reactions.base import Reaction


class LactoneHydrolysisFour(Reaction):
    """
    Hydrolysis of four-membered ring lactones.
    """

    def __init__(self):
        super().__init__(
            name="Lactone Hydrolysis (Four-membered ring)",
            reaction_smarts="[C;R:3]1[C;R:4](=[O:5])[O;R;!$(O-[C;!R](=O)):1][C,N,O;R:2]~1>>[OH][C:4](=[O:5])[C:3]~[C,N,O:2][OH:1]",
            examples={
                # Examples from the EPA
                "CC1CC(=O)O1": "CC(O)CC(=O)O",
            },
            halflife5=HALFLIFE1,
            halflife7=HALFLIFE5,
            halflife9=HALFLIFE5,
        )


class LactoneHydrolysisFive(Reaction):
    """
    Hydrolysis of five-membered ring lactones.
    """

    def __init__(self):
        super().__init__(
            name="Lactone Hydrolysis (Five-membered ring)",
            reaction_smarts="[#6,#7,#8;R:6]1[#6,#7,#8;R:3][#6;R:4](=[O:5])[#8;R;!$(O-[C;!R](=O)):1][#6,#7,#8:2]~1>>[OH][#6:4](=[O:5])[#6,#7,#8:3][#6,#7,#8:6]~[#6,#7,#8:2][OH:1]",
            examples={
                # Examples from the EPA
                "COC1(c2cccc([N+](=O)[O-])c2)OC(=O)c2ccccc21": "COC(O)(c1ccccc-1C(=O)O)c1cccc([N+](=O)[O-])c1",  # noqa: E501
                "Cc1ccc(C)c(SC=C2OC(=O)c3ccccc32)c1": "Cc1ccc(C)c(SC=C(O)c2ccccc-2C(=O)O)c1",
                # This produces SyntaxWarning: invalid escape sequence '\O'
                # We cannot convert it to raw string (r"") because when we match the normal string
                # produced by Chem.MolToSmiles to the raw string in the tests, it doesn't match.
                # I.e. Do not convert to raw string!
                "COc1ccc(O/C=C2\OC(=O)c3ccccc32)cc1": "COc1ccc(O/C=C(\\O)c2ccccc-2C(=O)O)cc1",
            },
            halflife5=HALFLIFE1,
            halflife7=HALFLIFE5,
            halflife9=HALFLIFE5,
        )


class LactoneHydrolysisSix(Reaction):
    """
    Hydrolysis of six-membered ring lactones.
    """

    def __init__(self):
        super().__init__(
            name="Lactone Hydrolysis (Six-membered ring)",
            reaction_smarts="[#6,#7,#8;R:7]1[#6,#7,#8;R:6][#6,#7,#8;R:3][#6;R:4](=[O:5])[#8;R;!$(O-[C;!R](=O)):1][#6,#7,#8;R:2]~1>>[OH][#6:4](=[O:5])[#6,#7,#8:3][#6,#7,#8:6][#6,#7,#8:7]~[#6,#7,#8:2][OH:1]",
            examples={
                # Examples from the EPA
                "O=C1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O": "O=C(O)[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO",  # noqa: E501
                # The double bond in the ring converts the carbonyl carbon and oxygen atom from
                # aliphatic to aromatc. This can be seen in the SMILES where c1 and o1 are lower
                # case. Compared to above where C1 and O1 are upper case.
                "O=c1ccc2ccccc2o1": "O=C(O)c-c-c1ccccc1O",
            },
            halflife5=HALFLIFE1,
            halflife7=HALFLIFE5,
            halflife9=HALFLIFE5,
        )
