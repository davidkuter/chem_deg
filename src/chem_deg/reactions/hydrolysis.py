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


if __name__ == "__main__":
    reaction_type = CarboxylateEsterHydrolysis()
    print(reaction_type.name)
    for reactant, product in reaction_type.examples.items():
        print(f"  Reactant: {Chem.MolToSmiles(Chem.MolFromSmiles(reactant))}")
        products = reaction_type.react(reactant)
        if products is None:
            print("  No products")
        else:
            print(f"  Products: {[Chem.MolToSmiles(product) for product in products]}")
