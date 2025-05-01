from rdkit import Chem

from chem_deg.kinetics.halflife import HALFLIFE1, HALFLIFE2, HALFLIFE3
from chem_deg.reactions.base import Reaction


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
            halflife5=HALFLIFE3,
            halflife7=HALFLIFE3,
            halflife9=HALFLIFE3,
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
            halflife5=HALFLIFE1,
            halflife7=HALFLIFE1,
            halflife9=HALFLIFE1,
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
            halflife5=HALFLIFE1,
            halflife7=HALFLIFE1,
            halflife9=HALFLIFE2,
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
