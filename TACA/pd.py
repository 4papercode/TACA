from typing import Tuple
import numpy as np
import gudhi as gd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_topological.nn import WassersteinDistance, PersistenceInformation
import matplotlib.pyplot as plt


def pairs_from_gudhi(
    edge_index: np.ndarray,
    f_v: np.ndarray
):

    N = f_v.shape[0]

    max_node_idx = int(np.argmax(f_v))
    unpaired_vertex_index = max_node_idx

    st = gd.SimplexTree()

    for v in range(N):
        st.insert([int(v)], filtration=float(f_v[v]))

    for u, v in edge_index:
        u, v = int(u), int(v)
        fu, fv = f_v[u], f_v[v]
        st.insert([u, v], filtration=float(max(fu, fv)))

    st.persistence(min_persistence = -1.0)
    pairs = st.persistence_pairs()
    
    pers_indices = -np.ones((N, 2), dtype=int)   # H0: [birth_vertex, death_vertex]

    for v in range(N):
        pers_indices[v, 0] = v

    for birth_simplex, death_simplex in pairs:
        dim = len(birth_simplex) - 1
        if dim == 0:
            i = int(birth_simplex[0])
            if len(death_simplex) < 1:
                pers_indices[i, 1] = unpaired_vertex_index
            else:
                u, v = int(death_simplex[0]), int(death_simplex[1])
                death_vertex = v if f_v[v] >= f_v[u] else u
                pers_indices[i, 1] = int(death_vertex)

    return pers_indices


def compute_pd(edge_index: torch.LongTensor,
               f_v: torch.Tensor) -> torch.Tensor:

    device = f_v.device
    edge_index_np = edge_index.detach().cpu().numpy().T
    pers_ind_np = pairs_from_gudhi(
       edge_index_np, f_v.detach().cpu().numpy()
    )
    pers_ind = torch.from_numpy(pers_ind_np).to(device)
    pd0 = torch.stack([f_v[pers_ind[:, 0]], f_v[pers_ind[:, 1]]], dim=-1)  # [N,2]

    return pd0

def plot_losses(losses: np.ndarray, savepath: str | None = None):
    plt.figure()
    plt.plot(np.arange(len(losses)), losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", dpi=150)
    plt.show()


def plot_dgm(dgm: np.ndarray, savepath: str | None = None):
    plt.figure()
    mask = ~(np.isclose(dgm[:, 0], 0.0) & np.isclose(dgm[:, 1], 0.0))
    dgm_valid = dgm[mask]
    if dgm_valid.size > 0:
        plt.scatter(dgm_valid[:, 0], dgm_valid[:, 1], s=16)
    lo = float(np.min(dgm_valid)) if dgm_valid.size > 0 else 0.0
    hi = float(np.max(dgm_valid)) if dgm_valid.size > 0 else 1.0
    xs = np.linspace(lo, hi, 100)
    plt.plot(xs, xs)
    plt.title("Persistence Diagram (birth vs. death)")
    plt.xlabel("birth")
    plt.ylabel("death")
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", dpi=150)
    plt.show()