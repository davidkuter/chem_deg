import networkx as nx
import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp

from chem_deg.kinetics.halflife import HalfLife
from chem_deg.reactions.base import Reaction
from chem_deg.reactions.reaction_classes import Hydrolysis


def _formation_degradation(
    edges: list[tuple[str, str, dict]],
    conc_dict: dict[str, float],
    ph: int = 5,
) -> float:
    results = []
    for reactant, _, attributes in edges:
        # Get the rate
        reaction: Reaction = attributes["reaction"]
        halflife: HalfLife = reaction.halflife[ph]
        rate = halflife.rate(halflife.midpoint)

        # Get the concentration of the reactant
        concentration = conc_dict[reactant]

        # Calculate the rate of formation
        results.append(rate * concentration)
    return sum(results)


def ode_equations(graph: nx.MultiDiGraph, concentrations: list[float], ph: int = 5) -> list[float]:
    """
    Define the ODEs for the degradation kinetics based on the graph structure.
    """
    # Create a dictionary to map node names to their concentrations
    conc_dict = {node: conc for node, conc in zip(graph.nodes, concentrations)}

    equations = []
    for node in graph.nodes:
        # Determine rate of formation using in edges
        # graph.in_edges("e") # => [('a', 'e'), ('d', 'e')] (note: e is second)
        in_edges = graph.in_edges(node, data=True)
        formation = _formation_degradation(in_edges, conc_dict, ph=ph)

        # Determine rate of degradation using out edges
        # graph.out_edges("b") # => [('b', 'c'), ('b', 'd')] (note: b is first)
        out_edges = graph.out_edges(node, data=True)
        degradation = _formation_degradation(out_edges, conc_dict, ph=ph)

        equations.append(formation - degradation)

    return equations


def _integrate(t, conc, graph: nx.MultiDiGraph, ph: int = 5) -> list[float]:
    """
    Integrate the ODEs over time.
    """
    # Update the concentrations based on the ODE equations
    return ode_equations(graph=graph, concentrations=conc, ph=ph)


def degradation_kinetics(
    degradation_graph: nx.MultiDiGraph,
    ph: int = 5,
    init_conc: float = 1.0,
    time_span: tuple[int, int] = (0, 1000),
    time_log: bool = False,
    time_points: int = 100,
) -> pd.DataFrame:
    
    # Initialize the concentrations of all nodes to 0, except for the reactant
    concentrations = [0.0] * len(degradation_graph.nodes)
    concentrations[0] = init_conc

    # Set the time span for the integration
    if time_log:
        time_min = 0 if time_span[0] == 0 else np.log(time_span[0])
        t_eval = np.logspace(time_min, np.log10(time_span[1]), num=time_points)
    else:
        t_eval = np.linspace(time_span[0], time_span[1], num=time_points)

    # Solve the ODEs
    solution = solve_ivp(
        fun=_integrate,
        t_span=time_span,
        y0=concentrations,
        args=(degradation_graph, ph),
        method="RK45",
        dense_output=True,
        t_eval=t_eval,
    )

    # Format the results into a DataFrame
    results = pd.DataFrame(solution.y.T, columns=degradation_graph.nodes, index=solution.t)
    results = results.rename_axis("Time (min)", axis=0)

    results = results * 100 / init_conc  # Convert to percentage
    results = results.round(2)  # Round to 2 decimal places

    return results


if __name__ == "__main__":
    # Example usage
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

    df = degradation_kinetics(graph)
    print(df)
