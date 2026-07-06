# TACA: Topology-Aware Contrastive Adaptation for Influenza Evolution Learning

This repository runs a graph transfer learning pipeline on H1N1 data.

## Data

The `dataset/` directory contains yearly H1N1 files:

- `H1N1_graph_YYYY.pt`: graph data for a given year
- `H1N1_pl_YYYY.pt`: persistence-landscape features for the same year

## Environment

Python version:

```bash
python==3.11.5
```

Main packages:

```text
torch==2.1.0+cu121
torch-geometric==2.4.0
torch-scatter==2.1.2+pt21cu121
torch-sparse==0.6.18+pt21cu121
torch-cluster==1.6.3+pt21cu121
torch-spline-conv==1.2.2+pt21cu121
pyg-lib==0.4.0+pt21cu121

numpy==1.26.4
scipy==1.17.1
scikit-learn==1.9.0
networkx==3.6.1
PyYAML==6.0.3
gudhi==3.13.0
torch-topological==0.1.9
matplotlib==3.11.0
```

## How to Run

From the project root:

```bash
bash run.sh
```


