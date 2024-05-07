"""
Module generating double integrator data for the CVAE.
"""

import numpy as np
import torch
from datetime import datetime
from pathlib import Path

DT = 0.1
NUM_TRAJS = 2500
TRAJ_LENGTH = 2.5
NUM_STEPS = int(TRAJ_LENGTH / DT)

DATASET_PATH = (
    Path(__file__).parent / "data" / f"DI_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt"
)


def f_nom(x: torch.tensor):
    p, v = torch.chunk(x, 2, dim=-1)

    return torch.cat([p + DT * v, v], dim=-1)


def d_mean(x: torch.tensor):
    p, v = torch.chunk(x, 2, dim=-1)

    return DT * torch.cat([torch.zeros_like(p), torch.sin(p)], dim=-1)


def d_var(x: torch.tensor):
    p, v = torch.chunk(x, 2, dim=-1)

    return (
        (DT**2)
        * 0.5
        * torch.cat(
            [
                torch.stack([2 + torch.cos(p), torch.exp(-torch.abs(p))], dim=-1),
                torch.stack([torch.exp(-torch.abs(p)), 2 + torch.sin(p)], dim=-1),
            ],
            dim=-2,
        )
    )


def generate_trajectory(x0: torch.tensor):
    X = torch.zeros(NUM_STEPS, 2)
    X[0] = x0
    for ii in range(1, NUM_STEPS):
        X[ii] = (
            f_nom(X[ii - 1])
            + torch.linalg.cholesky(d_var(X[ii - 1])) @ torch.randn_like(X[ii - 1])
            + d_mean(X[ii - 1])
        )

    return X


if __name__ == "__main__":
    X = torch.zeros(NUM_TRAJS, NUM_STEPS, 2)
    for ii in range(NUM_TRAJS):
        X[ii] = generate_trajectory(torch.zeros(2))

    torch.save(X, DATASET_PATH)
