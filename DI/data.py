import torch
from torch.utils.data import Dataset
from safety_filter.learning.DI.gen_di_data import f_nom


class DIDataset(Dataset):
    def __init__(self, path: str):
        """
        Initializes the double integrator dataset.

        Args:
            path: str; path to the dataset.
        """
        # Load the data.
        self.data = torch.load(path)

        self.x = self.data[:, :-1]
        self.xp = self.data[:, 1:]

        self.d = self.xp - f_nom(self.x)

        self.x = self.x.reshape(-1, 2)
        self.xp = self.xp.reshape(-1, 2)
        self.d = self.d.reshape(-1, 2)
        # self.cond = self.x[:, 0].reshape(-1, 1)
        self.cond = self.x.reshape(-1, 2)

        self.cond_mean = self.cond.mean(dim=0)
        self.cond_var = self.cond.var(dim=0)
        self.d_mean = self.d.mean(dim=0)
        self.d_var = self.d.var(dim=0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "d": self.d[idx],
            "x": self.x[idx],
            "xp": self.xp[idx],
            "cond": self.cond[idx],
        }
