import pandas as pd
import seaborn as sns

from io import BytesIO
from matplotlib import pyplot as plt
from rdkit import Chem

from chem_deg.util import draw_image


def draw_kinetics_graph(df: pd.DataFrame, ph: int |  str, filename: str = None) -> plt.Figure:
    """
    Draw the degradation kinetics graph.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the degradation kinetics data. 
        Index should be time points; columns should be compounds SMILES; values should be 
        concentrations.
    ph : int | str
        The pH value used for the calculations.
    filename : str, optional
        The name of the file to save the graph to. If None, the graph will not be saved.

    Returns
    -------
    plt.Figure
        The figure containing the degradation kinetics graph.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax1 = axes[0]  # ax1 is graph
    ax2 = axes[1]  # ax2 is SMILES visualization

    # Convert SMILES to numerical labels
    smiles = df.columns
    rename = {smiles: n for n, smiles in enumerate(df.columns)}
    df = df.rename(columns=rename)

    sns.lineplot(df, ax=ax1)
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Arbitary Concentration")
    ax1.set_title(f"Degradation Kinetics (pH {ph})")

    # Visualize SMILES
    png = draw_image(
        compound=[Chem.MolFromSmiles(smi) for smi in smiles],
        out_file=None,
        labels=[str(n) for n in range(len(df.columns))],
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