import torch
import numpy as np

# local imports parent directory
import sys
sys.path.append('../')

from simulate_von_neumann import Xeuler_sim


def create_datasets_sim(N_data: int, data_split: float, N: int, g: float, a0: float=1/np.sqrt(2), b0=None):
    '''
    Create val and train dataloaders with Von Neumann simulation data. Keep m measurements per simulation. If you need test data set data_split to 1

    Input:
            N_data: number of data points in total
            data_split: prcentage of data used for training, the rest is split into validation 
            N: number of measurements per simulation (assumes there is enough to reach steady state)
            g: interaction strength
            a0: initial state coefficient plus
            b0: initial state coefficient minus
    Output:
            train_loader: dataloader with training data
            val_loader: dataloader with validation data
            test_loader: dataloader with test data

            
    '''
    assert data_split <= 1 and data_split > 0, 'data_split must be between 0 and 1'


    X, _, _ = Xeuler_sim(N_data, N, g, a0, b0)
    X = torch.tensor(X).float()
    X = X.unsqueeze(1)
    

    # find label
    label = (X.sum(axis=-1) > 0).ravel().long()

    if data_split != 1:
        # split data index
        idx = np.arange(N_data)
        np.random.shuffle(idx)
        split_idx = int(data_split*N_data)
        idx_train = idx[:split_idx]
        idx_val = idx[split_idx:]
    
        # create dataset
        train_dataset = torch.utils.data.TensorDataset(X[idx_train], label[idx_train])
        val_dataset = torch.utils.data.TensorDataset(X[idx_val], label[idx_val])

        return train_dataset, val_dataset

    else: # test data
        test_dataset = torch.utils.data.TensorDataset(X, label)
        return test_dataset
    
    
    






    
