import torch
from torch.utils.data import Dataset
import pandas as pd
from safety_filter.minicvae.cvae_utils import planar_rotation, wrap_angle
import numpy as np


class ADAMDataset(Dataset):
    def __init__(self, path: str):
        """
        Initializes the ADAM dataset.

        Args:
            path: str; path to the dataset.
        """
        # Load the data.
        self.data = pd.read_csv(path)

        # Extract the preimpact states.
        stance_foot = np.array(self.data["stance_foot"])

        # Find preimpact indices i.e., index of all leading/trailing edges in the data.
        self.preimpact_indices = np.where(np.diff(stance_foot) != 0)[0]

        self.data = self.data.iloc[self.preimpact_indices]

        # Extract the states and controls.
        self.states = torch.tensor(
            self.data[["p_x_world", "p_y_world", "yaw_world"]].to_numpy()
        ).float()
        self.x = self.states[:-1]

        self.xp = self.states[1:]
        self.u = torch.tensor(
            self.data[["v_x_body_ctrl", "v_y_body_ctrl", "w_z_body_ctrl"]].to_numpy()
        )[:-1].float()

        self.training_fields = [
            "com_pos_x",
            "com_pos_y",
            "v_x_body",
            "v_y_body",
            "w_z_body",
            "stance_foot",
            # "swing_px",
            # "swing_py",
            # "v_x_body_ctrl",
            # "v_y_body_ctrl",
            # "w_z_body_ctrl",
        ]

        # Extract the time and conditioning data.
        self.dt = torch.tensor(self.data["time"].diff().values[1:]).float()
        self.conditioning = torch.tensor(self.data[self.training_fields].to_numpy())[
            :-1
        ].float()

        # Compute disturbances in world_frame.
        R = planar_rotation(self.x[:, 2])
        self.u_world = (R @ self.u.unsqueeze(-1)).squeeze(-1)
        self.d = self.xp - (self.x + self.u_world * self.dt.unsqueeze(-1))

        # Put disturbances into body frame.
        R_inv = R.inverse()
        self.d = (R_inv @ self.d.unsqueeze(-1)).squeeze(-1)

        # Wrap angles to [-pi, pi].
        self.d = torch.cat(
            [self.d[:, :2], wrap_angle(self.d[:, 2]).unsqueeze(-1)], dim=1
        )

        bad_inds = (torch.abs(self.d) > 0.15).any(dim=1)
        self.x = self.x[~bad_inds]
        self.u = self.u[~bad_inds]
        self.dt = self.dt[~bad_inds]
        self.d = self.d[~bad_inds]
        self.conditioning = self.conditioning[~bad_inds]

        # Compute statistics for normalization.
        self.d_mean = self.d.mean(dim=0)
        self.d_var = self.d.var(dim=0)
        self.cond_mean = self.conditioning.mean(dim=0)
        self.cond_var = self.conditioning.var(dim=0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "u": self.u[idx],
            "dt": self.dt[idx],
            "d": self.d[idx],
            "cond": self.conditioning[idx],
        }
