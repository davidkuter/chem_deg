from chem_deg.kinetics.halflife import HALFLIFE2, HALFLIFE3
from chem_deg.reactions.base import Reaction


class UreaHydrolysisAcyclic(Reaction):
    """
    Hydrolysis of acyclic ureas.
    """

    def __init__(self):
        super().__init__(
            name="Urea Hydrolysis (Acyclic)",
            # [H;C] attached to [N:1]/[N:4] is not specified because that would require an explicit
            # H atom to be present in the SMILES. I tried addingHs to the molecule but that caused
            # issues with other reactions so for now we will leave it out.
            # This will not catch unsubstituted ureas.
            reaction_smarts="[#6:2][N:1][C](=[O])[N:4][#6:5]>>[#6:2][NH:1].[NH:4][#6:5]",
            examples={
                # Examples from the EPA
                "C[C@H]1[C@H](c2ccc(Cl)cc2)SC(=O)N1C(=O)NC1CCCCC1": "C[C@@H]1NC(=O)S[C@H]1c1ccc(Cl)cc1.[NH]C1CCCCC1",  # noqa: E501
                "CC(C)c1ccc(NC(=O)N(C)C)cc1": "CC(C)c1ccc([NH])cc1.CNC",
                "O=C(Nc1ccccc1)N(Cc1ccc(Cl)cc1)C1CCCC1": "Clc1ccc(CNC2CCCC2)cc1.[NH]c1ccccc1",
            },
            halflife5=HALFLIFE3,
            halflife7=HALFLIFE2,
            halflife9=HALFLIFE2,
        )


class UreaHydrolysisCyclicFive(Reaction):
    """
    Hydrolysis of five-membered cyclic ureas.
    """

    def __init__(self):
        super().__init__(
            name="Urea Hydrolysis (Cyclic - Five-membered)",
            # [H;C] attached to [N:1]/[N:4] is not specified because that would require an explicit
            # H atom to be present in the SMILES. I tried addingHs to the molecule but that caused
            # issues with other reactions so for now we will leave it out.
            # This will not catch unsubstituted ureas.
            reaction_smarts="[#6;R:2]1[N;R:1][C,c;R](=[O])[N;R:4][#6;R:5]~1>>[NH:1][#6:2]~[#6:5][NH:4]",
            examples={
                # No examples from the EPA
                "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1": "[NH]C(=O)C([NH])(c1ccccc1)c1ccccc1",
            },
            halflife5=HALFLIFE3,
            halflife7=HALFLIFE2,
            halflife9=HALFLIFE2,
        )


class UreaHydrolysisCyclicSix(Reaction):
    """
    Hydrolysis of six-membered cyclic ureas.
    """

    def __init__(self):
        super().__init__(
            name="Urea Hydrolysis (Cyclic - Six-membered)",
            # [H;C] attached to [N:1]/[N:4] is not specified because that would require an explicit
            # H atom to be present in the SMILES. I tried addingHs to the molecule but that caused
            # issues with other reactions so for now we will leave it out.
            # This will not catch unsubstituted ureas.
            reaction_smarts="[#6;R:3]1[#6;R:2][N;R:1][C,c;R](=[O])[N;R:4][#6;R:5]~1>>[NH:1][#6:2][#6:3]~[#6:5][NH:4]",
            examples={
                # No examples from the EPA
                "CCC1(c2ccccc2)C(=O)NC(=O)NC1=O": "CCC(C([NH])=O)(C([NH])=O)c1ccccc1",
            },
            halflife5=HALFLIFE3,
            halflife7=HALFLIFE2,
            halflife9=HALFLIFE2,
        )


class SulfonylureaHydrolysis(Reaction):
    """
    Hydrolysis of sulfonylureas.
    """

    def __init__(self):
        super().__init__(
            name="Sulfonylurea Hydrolysis",
            reaction_smarts="[#6,#7:3][S:2](=[O])(=[O])[NH:1][C](=[O])[N,n:4][#6,#7:5]>>[#6,#7:3][S:2](=[O])(=[O])[NH2:1].[NH,nh:4][#6,#7:5]",
            examples={
                # Examples from the EPA
                "COC(=O)c1ccccc1S(=O)(=O)NC(=O)Nc1nc(C)nc(OC)n1 ": "COC(=O)c1ccccc1S(N)(=O)=O.COc1nc(C)nc([NH])n1",  # noqa: E501
                "COC(=O)c1c(Cl)nn(C)c1S(=O)(=O)NC(=O)Nc1nc(OC)cc(OC)n1": "COC(=O)c1c(Cl)nn(C)c1S(N)(=O)=O.COc1cc(OC)nc([NH])n1",  # noqa: E501
                "COC(=O)c1cccc(C)c1S(=O)(=O)NC(=O)Nc1nc(OCC(F)(F)F)nc(N(C)C)n1": "CN(C)c1nc([NH])nc(OCC(F)(F)F)n1.COC(=O)c1cccc(C)c1S(N)(=O)=O",  # noqa: E501
                "CCS(=O)(=O)c1cccnc1S(=O)(=O)NC(=O)Nc1nc(OC)cc(OC)n1": "CCS(=O)(=O)c1cccnc1S(N)(=O)=O.COc1cc(OC)nc([NH])n1",  # noqa: E501
                "COC(=O)c1csc(C)c1S(=O)(=O)NC(=O)n1nc(OC)n(C)c1=O": "COC(=O)c1csc(C)c1S(N)(=O)=O.COc1n-[nH]c(=O)n1C",  # noqa: E501
                "COc1cc(OC)nc(NC(=O)NS(=O)(=O)N(C)S(C)(=O)=O)n1": "CN(S(C)(=O)=O)S(N)(=O)=O.COc1cc(OC)nc([NH])n1",  # noqa: E501
            },
            halflife5=HALFLIFE3,
            halflife7=HALFLIFE2,
            halflife9=HALFLIFE2,
        )
