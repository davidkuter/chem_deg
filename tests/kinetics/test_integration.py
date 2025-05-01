import networkx as nx

from chem_deg.kinetics.integration import degradation_kinetics
from chem_deg.reactions.reaction_classes import Hydrolysis


def test_degradation_kinetics():
    """
    Test the degradation kinetics function with a fake graph.
    This test is not exhaustive and should be expanded with real data.
    """
    hydrolysis = Hydrolysis()

    # Prepare fake output from chemical_degradation
    graph = nx.MultiDiGraph()
    graph.add_node("Parent")
    graph.add_nodes_from(["Deg1", "Deg2", "Deg3", "Deg4", "Deg5", "Deg6", "Deg7"])
    graph.add_edge("Parent", "Deg1", reaction=hydrolysis.reactions[0])
    graph.add_edge("Parent", "Deg2", reaction=hydrolysis.reactions[0])
    graph.add_edge("Parent", "Deg3", reaction=hydrolysis.reactions[0])
    graph.add_edge("Deg1", "Deg4", reaction=hydrolysis.reactions[2])
    graph.add_edge("Deg1", "Deg5", reaction=hydrolysis.reactions[2])
    graph.add_edge("Deg2", "Deg5", reaction=hydrolysis.reactions[3])
    graph.add_edge("Deg3", "Deg6", reaction=hydrolysis.reactions[4])
    graph.add_edge("Deg6", "Deg4", reaction=hydrolysis.reactions[5])
    graph.add_edge("Deg4", "Deg7", reaction=hydrolysis.reactions[6])

    num_points = 10
    df = degradation_kinetics(graph, time_points=num_points, time_log=False)

    assert len(df.columns) == len(graph.nodes())
    assert len(df.index) == num_points
