import matplotlib.pyplot as plt
import networkx as nx

from io import BytesIO
from itertools import accumulate
from rdkit import Chem

from chem_deg.reactions.base import ReactionClass, Reaction
from chem_deg.reactions.reaction_classes import Hydrolysis
from chem_deg.util import draw_image


def draw_degradation_graph(graph: nx.MultiDiGraph, filename: str = None) -> plt.Figure:
    """
    Draw the graph and save it to a file.
    Adapted from:
    https://networkx.org/documentation/stable/auto_examples/drawing/plot_multigraphs.html

    Parameters
    ----------
    graph : nx.MultiDiGraph
        The graph to draw.
    filename : str, optional
        The name of the file to save the graph to. If None, the graph will not be saved.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax1 = axes[0]  # ax1 is graph
    ax2 = axes[1]  # ax2 is SMILES visualization

    # Draw the graph
    connectionstyle = [f"arc3,rad={r}" for r in accumulate([0.15] * 5)]

    pos = nx.shell_layout(graph)
    nx.draw_networkx_nodes(graph, pos, ax=ax1)
    nx.draw_networkx_labels(
        graph, pos, font_size=10, labels={node: n for n, node in enumerate(graph.nodes())}, ax=ax1
    )
    nx.draw_networkx_edges(graph, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax1)

    # Visualize SMILES
    png = draw_image(
        compound=[Chem.MolFromSmiles(node) for node in graph.nodes()],
        out_file=None,
        labels=[str(n) for n in range(len(graph.nodes()))],
        mols_per_row=3,
    )
    image = plt.imread(BytesIO(png))
    ax2.imshow(image, aspect="auto")
    ax2.axis("off")

    # Save the figure
    fig.tight_layout()
    if filename:
        fig.savefig(filename)

    return fig


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

    # Get unique nodes in the graph - used to determine stop condition
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
                    # Sort by length of the product string
                    flattened = sorted(flattened, key=lambda item: len(item[1]), reverse=True)
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


def chemical_degradation(compound: str | Chem.Mol, max_generation: int = 10_000) -> nx.MultiDiGraph:
    # Validate the input compound
    if isinstance(compound, str):
        compound = Chem.MolFromSmiles(compound)
        if compound is None:
            raise ValueError(f"Invalid SMILES provided: {compound}")

    # Standardize the SMILES
    compound = Chem.MolToSmiles(compound)

    # Initialize the reaction class
    reaction_classes = [Hydrolysis()]

    # Determine degradation products
    deg_graph = _compute_graph(
        [compound], reaction_classes, generation=1, max_generation=max_generation
    )

    return deg_graph


if __name__ == "__main__":
    # Example usage
    # Penicillin G
    smiles = "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"
    max_gen = 3
    graph = chemical_degradation(compound=smiles, max_generation=max_gen)
    print("Products:", graph.nodes())
    draw_degradation_graph(graph, "chemical_degradation.png")
