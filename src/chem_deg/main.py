import matplotlib.pyplot as plt
import networkx as nx

from itertools import accumulate
from rdkit import Chem

from chem_deg.reactions.base import ReactionClass, Reaction
from chem_deg.reactions.reaction_classes import Hydrolysis


def draw_graph(graph: nx.MultiDiGraph, filename: str = "graph.png"):
    """
    Draw the graph and save it to a file.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        The graph to draw.
    filename : str, optional
        The name of the file to save the graph to (default is "graph.png").
    """
    connectionstyle = [f"arc3,rad={r}" for r in accumulate([0.15] * 4)]
  
    pos = nx.shell_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_labels(graph, pos, font_size=10)
    nx.draw_networkx_edges(graph, pos, edge_color="grey", connectionstyle=connectionstyle)

    labels = {
        tuple(edge): f"gen={attrs['generation']}"
        for *edge, attrs in graph.edges(keys=True, data=True)
    }
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        labels,
        connectionstyle=connectionstyle,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0},
    )

    plt.savefig(filename)
    plt.close()


def _add_products_to_graph(
    graph: nx.MultiDiGraph,
    reactant: Chem.Mol,
    products: list[tuple[Reaction, str]],
    generation: int,
) -> nx.MultiDiGraph:

    for reaction, product in products:
        # Add the reactant to the graph if it doesn't exist
        if graph.has_node(product) is False:
            graph.add_node(product)

        # Add the edge from the reactant to the product
        graph.add_edge(
            reactant,
            product,
            reaction=reaction,
            generation=generation,
        )

    return graph


def _compute_graph(
    reactants: list[str],
    reaction_classes: list[ReactionClass],
    generation: int = 1,
    max_generation: int = 10_000,
    graph=None,
):
    # Initialize the graph if not provided
    if graph is None:
        graph = nx.MultiDiGraph()
        # There will only be one reactant at the start
        graph.add_node(reactants[0])

    # Get number of edges in the graph - used to determine stop condition
    nodes = set(graph.nodes())

    # Compute the products for each reactant and add them to the graph
    for reactant in reactants:
        for reaction_class in reaction_classes:
            # Determine the products for the reactant for this reaction class
            products: list[tuple[Reaction, str]] = reaction_class.react(reactant, return_mol=False)

            # Some products are salts, represented as a single string containing "."
            # We want to split these into separate products and add each molecule to the graph
            flat_products = []
            for reaction, product in products:
                if "." in product:
                    flattened = [(reaction, p) for p in product.split(".")]
                    flat_products.extend(flattened)
                else:
                    flat_products.append((reaction, product))

            # Add the products to the graph
            graph = _add_products_to_graph(
                graph=graph, reactant=reactant, products=flat_products, generation=generation
            )

    # Determine stop criteria

    # - Exit if the max generation has been reached
    generation += 1
    if generation >= max_generation:
        return graph

    # - Exit if no new nodes were added
    new_products = set(graph.nodes()) - nodes
    if len(new_products) == 0:
        return graph

    # Recursively compute products for the new products if stop criteria are not met
    return _compute_graph(
        reactants=list(new_products),
        reaction_classes=reaction_classes,
        generation=generation,
        max_generation=max_generation,
        graph=graph,
    )


def chemical_degradation(compound: str | Chem.Mol, max_generation: int = 10_000):
    # Validate the input compound
    if isinstance(compound, Chem.Mol):
        try:
            compound = Chem.MolToSmiles(compound)
        except Exception:
            raise ValueError("Invalid Mol object string provided.")

    # Standarize the SMILES
    compound = Chem.MolToSmiles(Chem.MolFromSmiles(compound))

    # Initialize the reaction class
    reaction_classes = [Hydrolysis()]

    # Determine degradation products
    deg_graph = _compute_graph(
        [compound], reaction_classes, generation=1, max_generation=max_generation
    )

    return deg_graph


if __name__ == "__main__":
    # Example usage
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    max_gen = 3
    graph = chemical_degradation(compound=smiles, max_generation=max_gen)
    print("Products:", graph.nodes())
    draw_graph(graph, "chemical_degradation.png")