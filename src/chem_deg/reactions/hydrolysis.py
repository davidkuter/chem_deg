"""
This code is based on the Abiotic Hydrolysis Reaction Library from the EPA. The reaction smarts were
determined from the schemas provided here:

https://qed.epa.gov/static_qed/cts_app/docs/Hydrolysis%20Lib%20HTML/HydrolysisRxnLib_ver1-8.htm
"""

from rdkit import Chem

from chem_deg.reactions.base import Reaction
from chem_deg.util import annotate_partial_charges


class HalogenatedAliphaticsSubstitutionA(Reaction):
    def __init__(self):
        super().__init__(
            name="Halogenated Aliphatics Substitution A",
            reaction_smarts="[C;X4;!$(C([F,Cl,Br,I])([F,Cl,Br,I])):1][Cl,Br,I:2]>>[C:1][OH:2]",
            examples={
                # Examples from the EPA
                "CBr": "CO",
                # Example to test reaction occurs only on the terminal halogen
                "ClCCC(Cl)(Cl)Cl": "OCCC(Cl)(Cl)Cl",
            },
        )


class HalogenatedAliphaticsSubstitutionC(Reaction):
    def __init__(self):
        super().__init__(
            name="Halogenated Aliphatics Substitution C",
            reaction_smarts="[C;X4;$(C([F,Cl,Br,I])([F,Cl,Br,I]));!$(C([F,Cl,Br,I])([F,Cl,Br,I])([F,Cl,Br,I])):1]([*:2])([*:3])([*:4])[Cl,Br,I:5]>>[C:1]([*:2])([*:3])([*:4])[OH:5]",
            examples={
                # Examples from the EPA
                "CC(C)(Cl)Cl": "CC(C)(O)Cl",
                # Example to test reaction occurs only on the Chlorine atom
                "CC(C)(F)Cl": "CC(C)(O)F",
            },
        )


class HalogenatedAliphaticsElimination(Reaction):
    def __init__(self):
        super().__init__(
            name="Halogenated Aliphatics Elimination",
            reaction_smarts="[C;$(C([#6,#7,#8,#9,#15,#16,#17,#35,#53])[#6,#7,#8,#9,#15,#16,#17,#35,#53]):1][C;$(C([#6,#7,#8,#9,#15,#16,#17,#35,#53])[#6,#7,#8,#9,#15,#16,#17,#35,#53]):2][I,Br,Cl]>>[C:1]=[C:2]",
            examples={
                # Examples from the EPA
                "CC(C)C(C)(C)Br": "CC(C)=C(C)C",
                "CCC(C)Br": "CC=CC",
                "CC1(Cl)CCCCC1": "CC1=CCCCC1",
                "C1(=CC=C(C=C1)Cl)C(C2=CC=C(C=C2)Cl)C(Cl)Cl": "ClC=C(c1ccc(Cl)cc1)c1ccc(Cl)cc1",
                "ClCCCl": "C=CCl",
                "C(C(Cl)Cl)(Cl)Cl": "ClC=C(Cl)Cl",
                "C(CBr)(CCl)Br": "C=C(Br)CCl",
            },
        )

    def _select_preferred_product(
        self, reactant: Chem.Mol, products: list[Chem.Mol]
    ) -> Chem.Mol | None:
        """
        Select the preferred elimination product based on:
        1. Heaviest halogen eliminated
        2. Most substituted carbon

        Parameters
        ----------
        reactant : Chem.Mol
            The reactant molecule.
        products : list[Chem.Mol]
            The products of the reaction.

        Returns
        -------
        Chem.Mol | None
            The preferred product of the reaction.
        """
        # Atomic numbers for halogens (heaviest first)
        HALOGENS = {53: "I", 35: "Br", 17: "Cl", 9: "F"}

        best_product = None
        best_score = -1

        # Get reactant SMILES for comparison
        reactant_smiles = Chem.MolToSmiles(reactant)

        for product in products:
            product_smiles = Chem.MolToSmiles(product)

            # Find which halogen was eliminated
            eliminated_halogen = None
            for atom in reactant.GetAtoms():
                atomic_num = atom.GetAtomicNum()
                if atomic_num in HALOGENS:
                    halogen_symbol = HALOGENS[atomic_num]
                    # Count occurrences of this halogen in reactant and product
                    reactant_count = reactant_smiles.count(halogen_symbol)
                    product_count = product_smiles.count(halogen_symbol)

                    if product_count < reactant_count:
                        eliminated_halogen = atomic_num
                        break

            if eliminated_halogen:
                # Score based on halogen weight
                halogen_score = eliminated_halogen
                total_score = halogen_score

                if total_score > best_score:
                    best_score = total_score
                    best_product = product

        return best_product

    def react(self, reactant: str) -> list[Chem.Mol] | None:
        """
        Override the base class method to handle the specific case of halogenated aliphatics
        elimination.

        Parameters
        ----------
        reactant : str
            The reactant SMILES.

        Returns
        -------
        list[Chem.Mol] | None
            A list of the products of the reaction.
        """
        # Convert reactant to molecule
        mol = Chem.MolFromSmiles(reactant)

        # Get all products from the base class
        products = self._react(mol)

        # If there are products, select the preferred product
        if products:
            return [self._select_preferred_product(reactant=mol, products=products)]


class EpoxideHydrolysis(Reaction):
    def __init__(self):
        super().__init__(
            name="Epoxide Hydrolysis",
            reaction_smarts="[C:1]1[O:2][C:3]1>>[C:1]([OH:2])-[C:3]-[OH]",
            examples={
                # Examples from the EPA
                "C1CCC2OC2C1": "OC1CCCCC1O",
                "ClCC1CO1": "OCC(O)CCl",
                "ClC1=C(Cl)C2(Cl)C3C4CC(C5OC45)C3C1(Cl)C2(Cl)Cl": "OC1C(O)C2CC1C1C2C2(Cl)C(Cl)=C(Cl)C1(Cl)C2(Cl)Cl",  # noqa: E501
                "c1ccc2c(c1)CCC1OC21": "OC1CCc2ccccc2C1O",
            },
        )


class PhosphorusEsterHydrolysis(Reaction):
    """
    Base class for hydrolysis of phosphorus esters. Child classes are for base- and acid-catalysed
    hydrolysis. A special routine is required to distinguish which reaction site is preferred for
    base- and acid-catalysed hydrolysis.
    """

    def __init__(self):
        super().__init__(
            name="Phosphorus Ester Hydrolysis",
            reaction_smarts="[P:1](=[O,S:2])([N,O,S:5])([N,O,S:6])[N,O,S:3]-[#6:4]>>[P:1](=[O,S:2])([N,O,S:5])([N,O,S:6])[OH].[N,O,S:3]-[#6:4]",
            # Intentionally left blank for child classes to fill in
            examples={},
        )

    @staticmethod
    def _determine_cleavage(reactant: Chem.Mol, product: Chem.Mol) -> tuple[int, int]:
        """
        Determine the atom indexes of the leaving atom and phosphorus atom in the reactant molecule.
        This is done by comparing the bonds in the reactant and product.
        The leaving atom is the one that is involved in a bond that is broken in the product but not
        in the reactant.

        Parameters
        ----------
        reactant : Chem.Mol
            The reactant molecule.
        product : Chem.Mol
            The product molecule(s).
        product_reactant_map : dict[int, int]
            A mapping of the product atom indexes to the reactant atom indexes. This is used to
            determine which atoms in the product correspond to which atoms in the reactant.

        Returns
        -------
        int
            The index of the leaving atom in the reactant molecule.
        """
        # Map the atoms in the product to the reactant
        product_reactant_map = {
            int(atom.GetIdx()): int(atom.GetProp("react_atom_idx"))
            for atom in product.GetAtoms()
            if atom.HasProp("react_atom_idx")
        }

        # Find atoms involved in bonds within the reactant. We only consider atoms that exist
        # in the product.
        reactant_bonds = set()
        for bond in reactant.GetBonds():
            # Get the atom indexes of the bond, sort them and add to the set. We have to sort
            # so that when we compare the bond in the product, we can compare the same atoms.
            bond_atoms = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            reactant_bonds.add(tuple(sorted(bond_atoms)))

        # Find atoms involved in bonds within the product. We only consider atoms that exist
        # in the reactant.
        product_bonds = set()
        for bond in product.GetBonds():
            start_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            if (
                start_atom in product_reactant_map.keys()
                and end_atom in product_reactant_map.keys()
            ):
                bond_atoms = (product_reactant_map[start_atom], product_reactant_map[end_atom])
                product_bonds.add(tuple(sorted(bond_atoms)))

        # Find the leaving atom by comparing the bonds in the reactant and product
        broken_bond = reactant_bonds - product_bonds

        if len(broken_bond) != 1:
            raise ValueError(
                f"Unable to determine leaving atom. More than one bond broken: {broken_bond}"
            )

        # Determine which atom is the leaving group and which is the Phosphorus atom
        broken_bond = list(broken_bond)[0]
        atom = reactant.GetAtomWithIdx(broken_bond[0])
        if atom.GetAtomicNum() == 15:
            phosphorus = broken_bond[0]
            leaving_atom = broken_bond[1]
        else:
            phosphorus = broken_bond[1]
            leaving_atom = broken_bond[0]

        return phosphorus, leaving_atom

    def _select_preferred_product(
        self, reactant: Chem.Mol, products: list[Chem.Mol], highest_electrophilicity: bool = True
    ) -> Chem.Mol | None:
        """
        Select the preferred product based electrophilicity of the carbon attached to the leaving
        group. We use GasteigerCharge partial charges as a proxy for electrophilicity.

        This method will be used by the child classes to determine the preferred product,
        setting the `highest_electrophilicity` parameter to the required value.

        Parameters
        ----------
        reactant : Chem.Mol
            The reactant molecule.
        products : list[Chem.Mol]
            The products of the reaction.
        highest_electrophilicity : bool
            If True, select the product with the highest electrophilicity. If False, select the
            product with the lowest electrophilicity.
            This is used to distinguish between base- and acid-catalysed hydrolysis.

        Returns
        -------
        Chem.Mol | None
            The preferred product of the reaction.
        """
        # Compute partial charges for the reactant.
        charged_reactant = annotate_partial_charges(reactant)

        # Set the results dictionary that stores the electrophilicity charge of the carbon attached
        # to the leaving atom as key, and product as value
        leaving_electrophilicity = {}

        # Iterate over all products to get the electrophilicity value of the product
        for product in products:
            # Find the leaving atom in the reactant
            phosphorus, leaving_atom = self._determine_cleavage(
                reactant=reactant,
                product=product,
            )

            # Find the partial charge of the carbon atom bonded to the leaving atom in the reactant
            for bond in reactant.GetBonds():
                bond_atoms = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]

                # We are only interested in the bond that contains the leaving atom
                if leaving_atom in bond_atoms:
                    # Ignore the leaving atom and phosphorus atom bond
                    if phosphorus not in bond_atoms:
                        # Get the atom index of the carbon atom
                        bond_atoms.remove(leaving_atom)
                        carbon_atom = charged_reactant.GetAtomWithIdx(bond_atoms[0])

                        # Get the Gasteiger charge of the carbon atom
                        charge = float(carbon_atom.GetProp("_GasteigerCharge"))

                        # Store the product in the dictionary with the charge as key
                        leaving_electrophilicity[charge] = product
                        break

        # Select the product with the highest or lowest electrophilicity
        if highest_electrophilicity:
            preferred_product = max(leaving_electrophilicity.keys())
        else:
            preferred_product = min(leaving_electrophilicity.keys())

        return leaving_electrophilicity[preferred_product]


class PhosphorusEsterHydrolysisBase(PhosphorusEsterHydrolysis):
    """
    Base-catalysed hydrolysis of organophosphorus esters.
    """

    def __init__(self):
        super().__init__()
        self.name = self.name + " (Base-catalysed)"
        self.examples = {
            # Examples from the EPA
            "CCOP(=S)(OCC)Oc1nc(Cl)c(Cl)cc1Cl": "CCOP(O)(=S)OCC.Oc1nc(Cl)c(Cl)cc1Cl",
            "CNC(=O)CSP(=S)(OC)OC": "CNC(=O)CS.COP(O)(=S)OC",
            "CCOP(=O)(NC(C)C)Oc1ccc(SC)c(C)c1": "CCOP(=O)(O)NC(C)C.CSc1ccc(O)cc1C",
            "COP(=S)(OC)Oc1ccc([N+](=O)[O-])c(C)c1": "COP(O)(=S)OC.Cc1cc(O)ccc1[N+](=O)[O-]",
        }

    def react(self, reactant: str) -> list[Chem.Mol] | None:
        # Convert reactant to molecule
        mol = Chem.MolFromSmiles(reactant)

        # Get all products from the base class
        products = self._react(mol)

        # If there are products, select the preferred product
        if products:
            return [
                self._select_preferred_product(
                    reactant=mol,
                    products=products,
                    # Set to True for base-catalysed hydrolysis
                    highest_electrophilicity=True,
                )
            ]


class PhosphorusEsterHydrolysisAcid(PhosphorusEsterHydrolysis):
    """
    Acid-catalysed hydrolysis of organophosphorus esters.
    """

    def __init__(self):
        super().__init__()
        self.name = self.name + " (Acid-catalysed)"
        self.examples = {
            # Examples from the EPA
            "CCOP(=S)(OCC)Oc1nc(Cl)c(Cl)cc1Cl": "CCO.CCOP(O)(=S)Oc1nc(Cl)c(Cl)cc1Cl",
            "CNC(=O)CSP(=S)(OC)OC": "CNC(=O)CSP(O)(=S)OC.CO",
            "COP(=S)(OC)Oc1ccc([N+](=O)[O-])c(C)c1": "CO.COP(O)(=S)Oc1ccc([N+](=O)[O-])c(C)c1",
        }

    def react(self, reactant: str) -> list[Chem.Mol] | None:
        # Convert reactant to molecule
        mol = Chem.MolFromSmiles(reactant)

        # Get all products from the base class
        products = self._react(mol)

        # If there are products, select the preferred product
        if products:
            return [
                self._select_preferred_product(
                    reactant=mol,
                    products=products,
                    # Set to True for base-catalysed hydrolysis
                    highest_electrophilicity=False,
                )
            ]


class CarboxylateEsterHydrolysis(Reaction):
    """
    Hydrolysis of carboxylate esters.
    """

    def __init__(self):
        super().__init__(
            name="Carboxylate Ester Hydrolysis",
            # I've used [#6:4] but EPA specifies that it should be [!N,!O]. This didn't work so I
            # used [#6:4] instead. Not sure if I will be missing any reactions because of this.
            reaction_smarts="[#6:4][C:1](=[O:2])[O:3][CX4:5]>>[#6:4][C:1](=[O:2])[OH].[CX4:5][OH:3]",
            examples={
                # Examples from the EPA
                "CCC(=O)OCC": "CCC(=O)O.CCO",
                "CCCCC(CC)COC(=O)c1ccccc1C(=O)OCC(CC)CCCC": "CCCCC(CC)CO.CCCCC(CC)COC(=O)c1ccccc1C(=O)O",  # noqa: E501
                "CC1(C)C(C(=O)OC(C#N)c2cccc(Oc3ccccc3)c2)C1(C)C": "CC1(C)C(C(=O)O)C1(C)C.N#CC(O)c1cccc(Oc2ccccc2)c1",  # noqa: E501
                "CCOC(=O)C(O)(c1ccc(Cl)cc1)c1ccc(Cl)cc1": "CCO.O=C(O)C(O)(c1ccc(Cl)cc1)c1ccc(Cl)cc1",  # noqa: E501
                "CC(C)=NOCCOC(=O)[C@@H](C)Oc1ccc(Oc2cnc3cc(Cl)ccc3n2)cc1": "CC(C)=NOCCO.C[C@@H](Oc1ccc(Oc2cnc3cc(Cl)ccc3n2)cc1)C(=O)O",  # noqa: E501
                "COC(=O)CC(NC(=O)[C@@H](NC(=O)OC(C)C)C(C)C)c1ccc(Cl)cc1": "CC(C)OC(=O)N[C@H](C(=O)NC(CC(=O)O)c1ccc(Cl)cc1)C(C)C.CO",  # noqa: E501
            },
        )


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
        )


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
        )


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
        )


class AmideHydrolysis(Reaction):
    """
    Hydrolysis of amides.
    """

    def __init__(self):
        super().__init__(
            name="Amide Hydrolysis",
            # [H;C;N] attached to [#7:4] is not specified because that would require an explicit
            # H atom to be present in the SMILES. I tried addingHs to the molecule but that caused
            # issues with other reactions so for now we will leave it out.
            reaction_smarts="[#6:3][#6:1](=[#8:2])[#7:4]>>[#6:3][#6:1](=[#8:2])[OH].[#7:4]",
            examples={
                # Examples from the EPA
                "CCNC(=O)[C@@H](C)OC(=O)Nc1ccccc1": "CCN.C[C@@H](OC(=O)Nc1ccccc1)C(=O)O",
                "C#CC(C)(C)NC(=O)c1cc(Cl)cc(Cl)c1": "C#CC(C)(C)N.O=C(O)c1cc(Cl)cc(Cl)c1",
                "CCC(=O)Nc1ccc(Cl)c(Cl)c1": "CCC(=O)O.Nc1ccc(Cl)c(Cl)c1",
                "O=C(NC(=O)c1c(F)cccc1F)Nc1ccc(Cl)cc1": "NC(=O)Nc1ccc(Cl)cc1.O=C(O)c1c(F)cccc1F",
            },
        )


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
        )


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
        )


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
        )


class NitrileHydrolysis(Reaction):
    """
    Hydrolysis of nitriles.
    """
    def __init__(self):
        super().__init__(
            name="Nitrile Hydrolysis",
            reaction_smarts="[C:2](#[N:1])[#6,#7:3]>>[NH2:1][C:2](=[O])[#6,#7:3]",
            examples={
                # Examples from the EPA
                "CC#N": "CC(N)=O",
                "N#Cc1ccccc1": "NC(=O)c1ccccc1",
                "N#CC(Cl)(Cl)Cl": "NC(=O)C(Cl)(Cl)Cl",
                "N#CNC#N": "N#CNC(N)=O",
                "N#Cc1nn(-c2c(Cl)cc(C(F)(F)F)cc2Cl)c(N)c1S(=O)C(F)(F)F": "NC(=O)c1nn(-c2c(Cl)cc(C(F)(F)F)cc2Cl)c(N)c1S(=O)C(F)(F)F",  # noqa: E501
            },
        )


class NSHydrolysis(Reaction):
    """
    Hydrolysis of N-S bonds.
    """

    def __init__(self):
        super().__init__(
            name="N-S Hydrolysis",
            reaction_smarts="[#6:2][#7:1]([#6:3])[#16:4][#6,#7,#8:5]>>[#6:2][#7:1]([#6:3]).[OH][#16:4][#6,#7,#8:5]",
            examples={
                # Examples from the EPA
                # A number of examples have R1-N-S-N-R2 bonds which can cleave to give
                # R1-NH2 or R2-NH2. The code only expects 1 product so these examples have been
                # omitted. In the end, the product will undergo both N-S hydrolysis so it's not a
                # big deal.
                "O=C1C2CC=CCC2C(=O)N1SC(Cl)(Cl)Cl": "O=C1NC(=O)C2CC=CCC12.OSC(Cl)(Cl)Cl",
                "O=C1c2ccccc2C(=O)N1SC(Cl)(Cl)Cl": "O=C1NC(=O)c2ccccc21.OSC(Cl)(Cl)Cl",
            },
        )


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
        )


class AcidHalideHydrolysis(Reaction):
    """
    Hydrolysis of acid halides.
    """

    def __init__(self):
        super().__init__(
            name="Acid Halidee Hydrolysis",
            reaction_smarts="[#6:2](=[#8:3])[F,Cl,Br,I]>>[#6:2](=[#8:3])[OH]",
            examples={
                # Examples from the EPA
                "O=CCl": "O=CO",
                "CC(=O)Cl": "CC(=O)O",
                "CC(C)OC(=O)Cl": "CC(C)OC(=O)O",
                "CSC(=O)Cl": "CSC(=O)O",
                "O=C(Cl)Cl": "O=C(O)Cl",
                "O=C(Cl)c1ccccc1": "O=C(O)c1ccccc1",
                "O=C(F)c1cccc(Cl)c1": "O=C(O)c1cccc(Cl)c1",
            },
        )



if __name__ == "__main__":
    reaction_type = AcidHalideHydrolysis()
    print(reaction_type.name)
    for reactant, product in reaction_type.examples.items():
        print(f"  Reactant: {Chem.MolToSmiles(Chem.MolFromSmiles(reactant))}")
        products = reaction_type.react(reactant)
        if products is None:
            print("  No products")
        else:
            print(f"  Products: {[Chem.MolToSmiles(product) for product in products]}")
