import torch
import numpy as np

# local imports parent directory
import sys
sys.path.append('../')

from simulate_von_neumann import Xeuler_sim


def create_dataloader_sim(N_data, data_split, N, g, a0=1/np.sqrt(2), b0=None):
    X, a, b = Xeuler_sim(N_data, N, g, a0, b0)
    X = torch.tensor(X).float()
    a = torch.tensor(a).float()
    b = torch.tensor(b).float()
    
    return X, a, b