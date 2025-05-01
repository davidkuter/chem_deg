from rdkit import Chem

from chem_deg.degradation import chemical_degradation, draw_graph
from chem_deg.kinetics.integration import degradation_kinetics


def main(
    compound: str | Chem.Mol,
    max_generation: int = 10_000,
    ph: int = 5,
    plot_degradation: bool = False,
    time_log: bool = False
):
    """
    Main function to compute the degradation kinetics of a compound.

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
    """
    # Compute the degradation graph
    deg_graph = chemical_degradation(compound=compound, max_generation=max_generation)

    # Draw the degradation graph if requested
    if plot_degradation:
        draw_graph(deg_graph, filename="degradation_graph.png")

    # Compute the degradation kinetics
    results = degradation_kinetics(degradation_graph=deg_graph, ph=ph, time_log=time_log)

    return results


if __name__ == "__main__":
    # Example usage
    compound = "CCC(=O)N(c1ccccc1)C1(C(=O)OC)CCN(CCC(=O)OC)CC1"
    results = main(compound=compound, ph=9, plot_degradation=True, time_log=True)
    results.to_csv("degradation_kinetics.tsv", sep="\t", index=True)
    print(results)
