import pandas as pd

from rdkit import Chem

from chem_deg.degradation import chemical_degradation, draw_degradation_graph
from chem_deg.kinetics.integration import degradation_kinetics
from chem_deg.kinetics.graph import draw_kinetics_graph


def simulate_degradation(
    compound: str | Chem.Mol,
    max_generation: int = 10_000,
    ph: int = 5,
    plot_degradation: bool = False,
    plot_kinetics: bool = False,
    time_log: bool = False,
) -> pd.DataFrame:
    """
    Simulate the degradation kinetics of a compound.

    Parameters
    ----------
    compound : str | Chem.Mol
        The compound to compute the degradation kinetics for.
    max_generation : int, optional
        The maximum number of generations to compute, by default 10_000.
    ph : int, optional
        The pH value to use for the calculations, by default 5.
    plot_degradation : bool, optional
        Whether to plot the degradation graph, by default False.
    plot_kinetics : bool, optional
        Whether to plot the degradation kinetics, by default False.
    time_log : bool, optional
        Whether to use logarithmic time points, by default False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the degradation kinetics results.
    """
    # Compute the degradation graph
    deg_graph = chemical_degradation(compound=compound, max_generation=max_generation)

    # Draw the degradation graph if requested
    if plot_degradation:
        draw_degradation_graph(deg_graph, filename="degradation_graph.png")

    # Compute the degradation kinetics
    results = degradation_kinetics(degradation_graph=deg_graph, ph=ph, time_log=time_log)

    # Draw the degradation kinetics plot if requested
    if plot_kinetics:
        draw_kinetics_graph(results, ph=ph, filename="degradation_kinetics.png")

    return results


if __name__ == "__main__":
    # Example usage
    # Penicillin G
    smiles = "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"
    results = simulate_degradation(
        compound=smiles, ph=5, plot_degradation=True, plot_kinetics=True, time_log=True
    )
    results.to_csv("degradation_kinetics.tsv", sep="\t", index=True)
    print(results)
