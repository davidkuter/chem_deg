from rdkit import Chem
from rdkit.Chem import AllChem


class Reaction:
    def __init__(self, name: str, reaction_smarts: str, examples: dict[str, str] = None):
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

            # Convert to SMILES to check for duplicates
            product_smiles = Chem.MolToSmiles(product)
            if product_smiles not in unique_products:
                unique_products.add(product_smiles)
                valid_products.append(product)

        return valid_products

    def react(self, reactant: str) -> list[str] | None:
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
        list[str]
            A list of the products of the reaction.
        """
        return self._react(Chem.MolFromSmiles(reactant))

    def __str__(self):
        return f"{self.name}: {self.reaction_smarts}"
