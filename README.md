# Quantum Computing Homework

This repository contains Python/Jupyter solutions for the MIS 68D quantum-like modeling homework PDF.

## Structure

- `quantum_lib.py`: shared matrix, projector, probability, density operator, POVM, rotation, and dynamics utilities.
- `notebooks/01_HW2_SG_App.ipynb`: HW2 Stern-Gerlach application, basic matrix operations, Born rule, projectors, conditional states, order effects, unitary matrices, spectral decomposition, and the A-B-C example.
- `notebooks/02_HW3_Advanced.ipynb`: HW3 tensor products, reciprocity, LTP/interference, order effects, density operators, sequential density measurements, and POVM checks.
- `notebooks/03_HW4_Paradoxes.ipynb`: HW4 elementary applications, real rotations, conjunction/order effects, question order effects, QQ-style computation, and Prisoner's Dilemma disjunction example.
- `notebooks/04_HW6_Dynamics.ipynb`: HW6 Markov and quantum dynamics, Fig. 6.2-style curves, Table 6.1-style dynamic interference, parameter experiments, and Temporal Bell violation.
- `data/`: small CSV inputs with the PDF/slides parameters used by the notebooks.

## Setup

```bash
pip3 install -r requirements.txt
```

## Run

Open the notebooks in Jupyter, or execute all of them from the project root:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_HW2_SG_App.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_HW3_Advanced.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_HW4_Paradoxes.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_HW6_Dynamics.ipynb
```

The notebooks have already been executed once, so saved outputs are visible when opened.
