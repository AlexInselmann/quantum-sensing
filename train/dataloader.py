import torch
import numpy as np

# local imports parent directory
import sys
sys.path.append('../')

from simulate_von_neumann import Xeuler_sim


def create_datasets_sim(N_data: int, data_split: float, N: int, g: float, a0: float=1/np.sqrt(2), b0=None, meter_error = None):
    '''
    Create val and train dataloaders with Von Neumann simulation data. Keep m measurements per simulation. If you need test data set data_split to 1

    Input:
            N_data: number of data points in total
            data_split: prcentage of data used for training, the rest is split into validation 
            N: number of measurements per simulation (assumes there is enough to reach steady state)
            g: interaction strength
            a0: initial state coefficient plus
            b0: initial state coefficient minus
            meter_error: None for a perfect meter, otherwise a number between 0 and 1 for the percential of missed measurements in the meausurement record
    Output:
            train_loader: dataloader with training data
            val_loader: dataloader with validation data
            test_loader: dataloader with test data

            
    '''
    assert data_split <= 1 and data_split > 0, 'data_split must be between 0 and 1'


    X, a, b = Xeuler_sim(N_data, N, g, a0, b0)
    a = torch.tensor(a).float()
    b = torch.tensor(b).float()

    X = torch.tensor(X).float()

    if meter_error is not None:
        # sample random measurements to be missed in the measurement record
        N_missed = int(meter_error*N)
        idx_keep = np.random.choice(N, N - N_missed, replace=False)

        # remove these measurements
        X = X[:,idx_keep]

        
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
        train_dataset = torch.utils.data.TensorDataset(X[idx_train], label[idx_train], a[idx_train], b[idx_train])
        val_dataset = torch.utils.data.TensorDataset(X[idx_val], label[idx_val], a[idx_val], b[idx_val])

        if meter_error is not None:
            return train_dataset, val_dataset, idx_keep
       
        return train_dataset, val_dataset

    else: # test data
        test_dataset = torch.utils.data.TensorDataset(X, label, a,b)

        if meter_error is not None:
            return test_dataset, idx_keep
        
        return test_dataset
    
    
    

if __name__ == '__main__':

    dataset = create_datasets_sim(1000, 1, 100, 0.1, meter_error=0.1)

    # check shapes
    sample_X, sample_y, sample_a, sample_b = dataset[0]
    print("X shape:", sample_X.shape)
    print("y shape:", sample_y.shape)
    print("a shape:", sample_a.shape)
    print("b shape:", sample_b.shape)


  



    
