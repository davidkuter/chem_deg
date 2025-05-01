from rdkit import Chem

from chem_deg.kinetics.halflife import HalfLife, HALFLIFE2, HALFLIFE3, HALFLIFE4, HALFLIFE5
from chem_deg.reactions.base import Reaction
from chem_deg.util import annotate_partial_charges


class PhosphorusEsterHydrolysis(Reaction):
    """
    Base class for hydrolysis of phosphorus esters. Child classes are for base- and acid-catalysed
    hydrolysis. A special routine is required to distinguish which reaction site is preferred for
    base- and acid-catalysed hydrolysis.
    """

    def __init__(self, halflife5: HalfLife, halflife7: HalfLife, halflife9: HalfLife):
        super().__init__(
            name="Phosphorus Ester Hydrolysis",
            reaction_smarts="[P:1](=[O,S:2])([N,O,S:5])([N,O,S:6])[N,O,S:3]-[#6:4]>>[P:1](=[O,S:2])([N,O,S:5])([N,O,S:6])[OH].[N,O,S:3]-[#6:4]",
            # Intentionally left blank for child classes to fill in
            examples={},
            halflife5=halflife5,
            halflife7=halflife7,
            halflife9=halflife9,
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
        super().__init__(halflife5=HALFLIFE2, halflife7=HALFLIFE3, halflife9=HALFLIFE4)
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
        super().__init__(halflife5=HALFLIFE2, halflife7=HALFLIFE3, halflife9=HALFLIFE3)
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
            reaction_smarts="[#6:4][C:1](=[O:2])[O:3][#6:5]>>[#6:4][C:1](=[O:2])[OH].[#6:5][OH:3]",
            examples={
                # Examples from the EPA
                "CCC(=O)OCC": "CCC(=O)O.CCO",
                "CCCCC(CC)COC(=O)c1ccccc1C(=O)OCC(CC)CCCC": "CCCCC(CC)CO.CCCCC(CC)COC(=O)c1ccccc1C(=O)O",  # noqa: E501
                "CC1(C)C(C(=O)OC(C#N)c2cccc(Oc3ccccc3)c2)C1(C)C": "CC1(C)C(C(=O)O)C1(C)C.N#CC(O)c1cccc(Oc2ccccc2)c1",  # noqa: E501
                "CCOC(=O)C(O)(c1ccc(Cl)cc1)c1ccc(Cl)cc1": "CCO.O=C(O)C(O)(c1ccc(Cl)cc1)c1ccc(Cl)cc1",  # noqa: E501
                "CC(C)=NOCCOC(=O)[C@@H](C)Oc1ccc(Oc2cnc3cc(Cl)ccc3n2)cc1": "CC(C)=NOCCO.C[C@@H](Oc1ccc(Oc2cnc3cc(Cl)ccc3n2)cc1)C(=O)O",  # noqa: E501
                "COC(=O)CC(NC(=O)[C@@H](NC(=O)OC(C)C)C(C)C)c1ccc(Cl)cc1": "CC(C)OC(=O)N[C@H](C(=O)NC(CC(=O)O)c1ccc(Cl)cc1)C(C)C.CO",  # noqa: E501
            },
            halflife5=HALFLIFE2,
            halflife7=HALFLIFE2,
            halflife9=HALFLIFE5,
        )


if __name__ == "__main__":
    ester_hydrolysis = CarboxylateEsterHydrolysis()
    aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    aspirin_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(aspirin_smiles))
    products = ester_hydrolysis.react(aspirin_smiles)
    print(f"Products of hydrolysis of {aspirin_smiles}:\n{products}")