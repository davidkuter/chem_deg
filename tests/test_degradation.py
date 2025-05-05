import networkx as nx
import pytest

from chem_deg.degradation import chemical_degradation, draw_degradation_graph


def test_chemical_degradation():
    """
    Test the chemical_degradation function.
    """
    # Example SMILES for Penicillin G
    smiles = "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"

    # Compute the degradation graph
    deg_graph = chemical_degradation(compound=smiles, max_generation=10_000)

    # Check if the graph has the expected number of nodes
    expected_nodes = 8
    num_nodes = len(deg_graph.nodes())
    assert num_nodes == expected_nodes, (
        f"Number of nodes ({num_nodes}) does not match expected number ({expected_nodes})."
    )

    # Check if the graph has the expected number of edges
    expected_edges = 12
    edges = [e for e in deg_graph.edges(data=True)]
    num_edges = len(edges)
    assert num_edges == expected_edges, (
        f"Number of edges ({num_edges}) does not match expected number ({expected_edges})."
    )

    # Check if the first edge has the expected attributes
    assert edges[0][2].get("reaction", None) is not None, (
        "Reaction attribute is missing in the edge."
    )
    assert edges[0][2].get("generation", None) is not None, (
        "Generation attribute is missing in the edge."
    )


def test_chemical_degradation_invalid_smiles():
    """
    Test the chemical_degradation function with invalid
    SMILES input.
    """

    invalid_smiles = "BAD_SMILES"

    with pytest.raises(ValueError, match="Invalid SMILES provided."):
        chemical_degradation(compound=invalid_smiles)


def test_draw_graph():
    """
    Test the draw_graph function.
    """
    # Mock graph
    graph = nx.MultiDiGraph()
    graph.add_node("CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O")
    graph.add_nodes_from(
        [
            "CC(C)([CH]C(=O)O)S[CH][C@H](NC(=O)Cc1ccccc1)C(=O)O",
            "CC1(C)S[CH]N[C@H]1C(=O)O",
            "CC1(C)S[C@@H]2[C@H](N)C(=O)N2[C@H]1C(=O)O",
        ]
    )
    graph.add_edge(
        "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
        "CC(C)([CH]C(=O)O)S[CH][C@H](NC(=O)Cc1ccccc1)C(=O)O",
    )
    graph.add_edge(
        "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O", "CC1(C)S[CH]N[C@H]1C(=O)O"
    )
    graph.add_edge(
        "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
        "CC1(C)S[C@@H]2[C@H](N)C(=O)N2[C@H]1C(=O)O",
    )

    # Draw the graph and save it to a file
    fig = draw_degradation_graph(graph, filename=None)

    assert fig is not None, "Figure object should not be None."
