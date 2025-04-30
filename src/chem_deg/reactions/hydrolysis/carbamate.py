from chem_deg.kinetics.halflife import HALFLIFE1, HALFLIFE3, HALFLIFE5
from chem_deg.reactions.base import Reaction


class CarbamateHydrolysis(Reaction):
    """
    Hydrolysis of carbamates. N,N-disubstituted carbamates are resistant to hydrolysis.
    """

    def __init__(self):
        super().__init__(
            name="Carbamate Hydrolysis",
            reaction_smarts="[#6,#7:2][NH:1][C](=[O])[#8:3][#6,#7:4]>>[#6,#7:2][NH2:1].[#8:3][#6,#7:4]",
            examples={
                # Examples from the EPA
                "CNC(=O)Oc1ccc([N+](=O)[O-])cc1": "CN.O=[N+]([O-])c1ccc(O)cc1",
                "CNC(=O)Oc1cccc2ccccc12": "CN.Oc1cccc2ccccc12",
                "CCNC(=O)[C@@H](C)OC(=O)Nc1ccccc1": "CCNC(=O)[C@@H](C)O.Nc1ccccc1",
            },
            halflife5=HALFLIFE3,
            halflife7=HALFLIFE3,
            halflife9=HALFLIFE5,
        )


class ThiocarbamateHydrolysis(Reaction):
    """
    Hydrolysis of thiocarbamates.
    """

    def __init__(self):
        super().__init__(
            name="Thiocarbamate Hydrolysis",
            # [H;C;N] attached to [N:1] is not specified because that would require an explicit
            # H atom to be present in the SMILES. I tried addingHs to the molecule but that caused
            # issues with other reactions so for now we will leave it out.
            reaction_smarts="[#6,#7:2][N:1][C](=[O])[S:4][#6:5]>>[#6,#7:2][NH:1].[SH:4][#6:5]",
            examples={
                # Examples from the EPA
                "CC(C)N(C(=O)SC/C(Cl)=C/Cl)C(C)C": "CC(C)NC(C)C.SC/C(Cl)=C/Cl",
            },
            halflife5=HALFLIFE1,
            halflife7=HALFLIFE1,
            halflife9=HALFLIFE1,
        )
