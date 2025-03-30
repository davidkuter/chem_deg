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
            A dictionary of examples of the reaction. The keys and values are the reactant and product SMILES respectively.
        """
        self.name = name
        self.reaction_smarts = reaction_smarts
        self.examples = examples or {}
        self._rxn = AllChem.ReactionFromSmarts(self.reaction_smarts)

    # def react(self, mol: Chem.Mol | str) -> list[Chem.Mol] | None:
    #     """ 
    #     React a molecule.

    #     Parameters
    #     ----------
    #     mol : Chem.Mol | str
    #         The molecule to react.

    #     Returns
    #     -------
    #     list[Chem.Mol] | None
    #         The products of the reaction.
    #     """

    #     # If the molecule is a string, convert it to a molecule
    #     if isinstance(mol, str):
    #         mol = Chem.MolFromSmiles(mol)

    #     # Run the reaction
    #     products = self._rxn.RunReactants((mol,))

    #     # If the reaction does not produce any products, return None
    #     if len(products) == 0:
    #         return None

    #     # Return the products
    #     return [product[0] for product in products]

    def react(self, mol: Chem.Mol | str) -> list[Chem.Mol] | None:
        """React a molecule."""
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
            product = product_tuple[0]
            # Convert to SMILES to check for duplicates
            product_smiles = Chem.MolToSmiles(product)
            if product_smiles not in unique_products:
                unique_products.add(product_smiles)
                valid_products.append(product)
        
        return valid_products

    def __str__(self):
        return f"{self.name}: {self.reaction_smarts}"
