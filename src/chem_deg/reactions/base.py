from rdkit import Chem
from rdkit.Chem import AllChem

from chem_deg.kinetics.halflife import HalfLife


class Reaction:
    def __init__(
        self,
        name: str,
        reaction_smarts: str,
        examples: dict[str, str] = None,
        halflife5: HalfLife = None,
        halflife7: HalfLife = None,
        halflife9: HalfLife = None,
    ):
        """
        Initialize a reaction with a name, reactant SMARTS, and reaction SMARTS.

        Parameters
        ----------
        name : str
            The name of the reaction.
        reaction_smarts : str
            The reaction SMARTS of the reaction.
        examples : dict[str, str]
            A dictionary of examples of the reaction. The keys and values are the reactant and
            product SMILES respectively.
        """
        self.name = name
        self.reaction_smarts = reaction_smarts
        self.examples = examples or {}
        self._rxn = AllChem.ReactionFromSmarts(self.reaction_smarts)

        # Halflife
        self.halflife = {5: halflife5, 7: halflife7, 9: halflife9}

    def _react(self, mol: Chem.Mol | str) -> list[Chem.Mol] | None:
        """
        Attempt to react a molecule to product a degradation product

        Parameters
        ----------
        mol : Chem.Mol | str
            The molecule to react.

        Returns
        -------
        list[Chem.Mol] | None
            A list of the products of the reaction.
        """
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)

        # ToDo: This might be useful but causes issues later on when we strip hydrogens
        # Try add hydrogens to the molecule. We need to this to add explicit hydrogens to the
        # molecule for reactions to work.
        # mol = Chem.AddHs(mol)

        # Run the reaction
        products = self._rxn.RunReactants((mol,))
        if not products:
            return None

        # Use a set to collect unique product SMILES
        unique_products = set()
        valid_products = []

        # Iterate through all products
        for product_tuple in products:
            # If there is only one product in the tuple, extract it
            if len(product_tuple) == 1:
                product = product_tuple[0]
            # If there are multiple products, convert them to SMILES and join them with "."
            else:
                product = Chem.CombineMols(*product_tuple)

            # ToDo: This is needed if we add hydrogen to the molecule but I got some failures with
            # tests because non-ring atoms where assigned as aromatic. We'd need to do some serious
            # error handling here to make it work. For the time being, I am not doing it.
            # Remove hydrogens from the product
            # print(Chem.MolToSmiles(product))
            # product = Chem.RemoveHs(product)

            # Convert to SMILES to check for duplicates
            product_smiles = Chem.MolToSmiles(product)

            # Sometimes SMILES contain "-" which indicates single bonds. E.g.
            # "COc1n-[nH]c(=O)n1C" vs "COc1n[nH]-c(=O)n1C" are the same but have different SMILES
            # To account for this, we strip "-" from the SMILES because single bonds are implied.
            product_smiles = product_smiles.replace("-", "")

            if product_smiles not in unique_products:
                unique_products.add(product_smiles)
                valid_products.append(product)

        return valid_products

    def react(self, reactant: str) -> list[Chem.Mol] | None:
        """
        Entry point for the reaction. Some reactions may need to override this method to handle
        specific cases. This would mostly occur when there is a preferential product formation over
        others.

        Parameters
        ----------
        reactant : str
            The reactant SMILES.

        Returns
        -------
        list[Chem.Mol] | None
            A list of the products of the reaction as rdkit Mol objects.
        """
        try:
            mol = Chem.MolFromSmiles(reactant)
        except Exception:
            raise TypeError(
                f"Invalid SMILES string provided: {reactant}."
            )

        return self._react(mol)

    def __str__(self):
        return f"{self.name}: {self.reaction_smarts}"


class ReactionClass:
    def __init__(self, name: str, reactions: list[Reaction]):
        """
        Initialize a reaction class with a name and a list of reactions.

        Parameters
        ----------
        name : str
            The name of the reaction class.
        reactions : list[Reaction]
            A list of reactions in the reaction class.
        """
        self.name = name
        self.reactions = reactions

    def _cleaned_smiles(self, smiles) -> str:
        """
        Clean the SMILES string by removing any unwanted characters.

        Parameters
        ----------
        smiles : str
            The SMILES string to clean.

        Returns
        -------
        str
            The cleaned SMILES string.
        """
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)

    def react(
        self, reactant: str, return_mol: bool = False
    ) -> list[tuple[Reaction, str | Chem.Mol]]:
        """
        React a molecule with all reactions in the reaction class.

        Parameters
        ----------
        reactant : str
            The reactant SMILES.
        return_mol : bool, optional
            If True, return the products as rdkit Mol objects. If False, return the products as
            SMILES strings. The default is False.

        Returns
        -------
        list[tuple[Reaction, str | Chem.Mol]]
            A list of tuples containing the reaction and products the products of the reaction.
        """
        # React the reactant with all reactions in the reaction class
        products = []
        for reaction in self.reactions:
            product = reaction.react(reactant)
            if product:
                for p in product:
                    products.append((reaction, p))

        # Convert products to SMILES if return_mol is False
        if return_mol is False:
            products = [
                (reaction, self._cleaned_smiles(Chem.MolToSmiles(product)))
                for reaction, product in products
            ]
        return products
