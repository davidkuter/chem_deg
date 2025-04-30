from chem_deg.kinetics.halflife import HALFLIFE7
from chem_deg.reactions.base import Reaction


class AnhydrideHydrolysisAcyclic(Reaction):
    """
    Hydrolysis of acyclic anhydrides.
    """

    def __init__(self):
        super().__init__(
            name="Anhydride Hydrolysis (Acyclic)",
            reaction_smarts="[#6:3][#6;!R:2](=[O:6])[O:1][#6;!R:4](=[O:7])[#6:5]>>[#6:3][#6:2](=[O:6])[OH:1].[OH][#6:4](=[O:7])[C:5]",
            examples={
                # Examples from the EPA
                "CC(=O)OC(C)=O": "CC(=O)O.CC(=O)O",
                "CC(C)(C)C(=O)OC(=O)C(C)(C)C": "CC(C)(C)C(=O)O.CC(C)(C)C(=O)O",
            },
            halflife5=HALFLIFE7,
            halflife7=HALFLIFE7,
            halflife9=HALFLIFE7,
        )


class AnhydrideHydrolysisCyclicFive(Reaction):
    """
    Hydrolysis of five-membered cyclic anhydrides.
    """

    def __init__(self):
        super().__init__(
            name="Anhydride Hydrolysis (Cyclic - Five-membered)",
            # The ~1 in the reactant is import to allow any bond between the two carbon atoms
            reaction_smarts="[#6;R:3]1[#6;R:2](=[#8:6])[#8;R:1][#6;R:4](=[#8:7])[#6;R:5]~1>>[OH:1]-[#6:2](=[#8:6])-[#6:3]~[#6:5]-[#6:4](=[#8:7])[OH]",
            examples={
                # Examples from the EPA
                "O=C1C=CC(=O)O1": "O=C(O)C=CC(=O)O",
                "O=C1CCC(=O)O1": "O=C(O)CCC(=O)O",
                "CC1(C)C(=O)OC(=O)C1(C)C": "CC(C)(C(=O)O)C(C)(C)C(=O)O",
                "Cc1ccc(C)c2c1C(=O)OC2=O": "Cc1ccc(C)c(C(=O)O)c1C(=O)O",
            },
            halflife5=HALFLIFE7,
            halflife7=HALFLIFE7,
            halflife9=HALFLIFE7,
        )


class AnhydrideHydrolysisCyclicSix(Reaction):
    """
    Hydrolysis of six-membered cyclic anhydrides.
    """

    def __init__(self):
        super().__init__(
            name="Anhydride Hydrolysis (Cyclic - Six-membered)",
            # The ~1 in the reactant is import to allow any bond between the two carbon atoms
            reaction_smarts="[#6;R:3]1[#6;R:2](=[#8:6])[#8;R:1][#6;R:4](=[#8:7])[#6;R:5][#6;R:8]~1>>[OH:1]-[#6:2](=[#8:6])-[#6:3]~[#6:8][#6:5]-[#6:4](=[#8:7])[OH]",
            examples={
                # Examples from the EPA
                "O=C1CCCC(=O)O1": "O=C(O)CCCC(=O)O",
            },
            halflife5=HALFLIFE7,
            halflife7=HALFLIFE7,
            halflife9=HALFLIFE7,
        )
