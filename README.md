# Chemical Degradation Reaction
This repository provides a framework for simulating and analyzing chemical degradation pathways of organic compounds. It leverages the RDKit cheminformatics library to predict degradation products based on a set of predefined reaction rules and allows for the kinetic modeling of these degradation processes.

Reaction Libraries have been sourced from the EPA's [Chemical Transformation Simulator](https://qed.epa.gov/cts/about/reactionlibs/).

![Degradation Graph](src/chem_deg/assets/degradation_graph.png)
![Kinetics Graph](src/chem_deg/assets/degradation_kinetics.png)

## Dependencies

*   Python (>=3.12)
*   RDKit (>=2024.9.6)
*   NetworkX (>=3.4)
*   Matplotlib (>=3.10)
*   Seaborn (>=0.13)
*   Pandas (>=2.2)
*   SciPy (>=1.15)

## Installation

Create and active virtual environment
```bash
python3 -m venv chem_deg
source chem_deg/bin/activate
```

Clone and install the repository
```bash
git clone git@github.com:davidkuter/chem_deg.git
pip install -e .
```

## Usage

```python
from chem_deg import simulate_degradation

# Simulate degradation starting from a SMILES string
smiles = "CCC(=O)N(c1ccccc1)C1(C(=O)OC)CCN(CCC(=O)OC)CC1"
df_results = simulate_degradation(
    compound=smiles,    # The compound to simulate degradation of
    ph=5,               # The pH to simulate degradation kinetics of
    max_generation=2,   # The maximum number of degradation steps to simulate.
    plot_degradation=True,  # If `True`, generates a visualization of the degradation graph (saved as `degradation_graph.png`).
    plot_kinetics=True,     # If `True`, generates a plot of the degradation kinetics (saved as `degradation_kinetics.png`).
    time_log=True,      # If `True`, uses a logarithmic time scale for the kinetics simulation.
)
```

## Development

To install development dependencies:

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License - see the [LICENSE](LICENSE) file for details.

For commercial use, please contact David Kuter.