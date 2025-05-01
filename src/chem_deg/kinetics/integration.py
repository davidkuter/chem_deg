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
    degradation: bool = True,
) -> float:
    """
    Calculate the rate of formation or degradation based on the edges and concentrations.

    Parameters
    ----------
    edges : list[tuple[str, str, dict]]
        List of edges in the graph, where each edge is a tuple (source, target, attributes).
    conc_dict : dict[str, float]
        Dictionary mapping node names to their concentrations.
    ph : int, optional
        The pH value to use for the calculations, by default 5.
    degradation : bool, optional
        Whether to calculate the rate of degradation (True) or formation (False), by default True.

    Returns
    -------
    float
        The rate of formation or degradation.
    """
    results = []
    for reactant, _, attributes in edges:
        # Get the rate
        reaction: Reaction = attributes["reaction"]

        # We currently don't support reactions with multiple reactants.
        # This is a limitation of the current implementation.
        # If the reaction has multiple reactants, raise an error.
        if "." in reaction.reaction_smarts.split(">>")[0]:
            raise ValueError(
                f"Edge ({reactant}, {attributes}) has multiple reactants. This is not supported."
            )

        halflife: HalfLife = reaction.halflife[ph]
        rate = halflife.rate(halflife.midpoint)

        # Important: account for double counting.
        # When a reaction produces multiple products, the rate is divided by the number of products.
        # We should only do this, however, for degradation reactions.
        if degradation:
            num_products = len(reaction.reaction_smarts.split(">>")[1].split("."))
            rate /= num_products

        # Get the concentration of the reactant
        concentration = conc_dict[reactant]

        # Calculate the rate of formation
        results.append(rate * concentration)
    return sum(results)


# NOTE: This will not work for reactions with multiple reactants.
# ToDo: Make compatible with multiple reactants.
def ode_equations(graph: nx.MultiDiGraph, concentrations: list[float], ph: int = 5) -> list[float]:
    """
    Define the ODEs for the degradation kinetics based on the graph structure.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        The directed graph representing the degradation pathways.
    concentrations : list[float]
        The current concentrations of the nodes in the graph.
    ph : int, optional
        The pH value to use for the calculations, by default 5.

    Returns
    -------
    list[float]
        The rates of change of the concentrations for each node in the graph.
    """
    # Create a dictionary to map node names to their concentrations
    conc_dict = {node: conc for node, conc in zip(graph.nodes, concentrations)}

    equations = []
    for node in graph.nodes:
        # Determine rate of formation using in edges
        # graph.in_edges("e") # => [('a', 'e'), ('d', 'e')] (note: e is second)
        in_edges = graph.in_edges(node, data=True)
        formation = _formation_degradation(in_edges, conc_dict, ph=ph, degradation=False)

        # Determine rate of degradation using out edges
        # graph.out_edges("b") # => [('b', 'c'), ('b', 'd')] (note: b is first)
        out_edges = graph.out_edges(node, data=True)
        degradation = _formation_degradation(out_edges, conc_dict, ph=ph, degradation=True)

        equations.append(formation - degradation)

    return equations


def _integrate(t, conc, graph: nx.MultiDiGraph, ph: int = 5) -> list[float]:
    """
    Integrate the ODEs over time. Wrapped for use with solve_ivp.
    """
    # Update the concentrations based on the ODE equations
    return ode_equations(graph=graph, concentrations=conc, ph=ph)


def degradation_kinetics(
    degradation_graph: nx.MultiDiGraph,
    ph: int = 5,
    init_conc: float = 1.0,
    time_span: tuple[int, int] = (0, 720),  # 0 to 720 hours (30 days)
    time_log: bool = False,
    time_points: int = 100,
) -> pd.DataFrame:
    """
    Calculate the degradation kinetics of a compound over time based on the degradation graph.

    Parameters
    ----------
    degradation_graph : nx.MultiDiGraph
        The directed graph representing the degradation pathways.
    ph : int, optional
        The pH value to use for the calculations, by default 5.
    init_conc : float, optional
        The initial concentration of the reactant, by default 1.0.
    time_span : tuple[int, int], optional
        The time span (in hours) for the integration, by default (0, 720).
    time_log : bool, optional
        Whether to use logarithmic time points, by default False.
    time_points : int, optional
        The number of time points to generate, by default 100.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the concentrations of each node over time.
    """

    # Initialize the concentrations of all nodes to 0, except for the reactant
    concentrations = [0.0] * len(degradation_graph.nodes)
    concentrations[0] = init_conc

    # Set the time span for the integration
    if time_log:
        time_min = np.log(0.5) if time_span[0] == 0 else np.log(time_span[0])
        t_eval = np.logspace(time_min, np.log10(time_span[1]), num=time_points)
        t_eval = np.insert(t_eval, 0, 0)  # Add the initial time point
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
        # Rounding is required to avoid sensitivity where solve_ivp thinks t_eval is not in t_span
        t_eval=np.round(t_eval, 4),
    )

    # Format the results into a DataFrame
    results = pd.DataFrame(solution.y.T, columns=degradation_graph.nodes, index=solution.t)
    results = results.rename_axis("Time (hours)", axis=0)
    results = results.round(2)  # Round to 2 decimal places
    results = results.replace(-0.0, 0.0)  # Convert -0.0 to 0.0

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
