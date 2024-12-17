import torch
import numpy as np

# local imports parent directory
import sys
sys.path.append('../')

from simulate_von_neumann import Xeuler_sim

# 

def create_datasets_sim(N_data: int, data_split: float, N: int, g: float,U_s: np.ndarray, a0: float=1/np.sqrt(2), b0=None, delta_t: float = 0.05,r: int =  None,  meter_error = None, verbose = False):
    '''
    Create val and train dataloaders with Von Neumann simulation data. Keep m measurements per simulation. If you need test data set data_split to 1

    Input:
            N_data: number of data points in total
            data_split: prcentage of data used for training, the rest is split into validation 
            N: number of measurements per simulation (assumes there is enough to reach steady state)
            g: interaction strength
            a0: initial state coefficient plus
            b0: initial state coefficient minus
            r: successrate for a measurement to happen.
            meter_error: None for a perfect meter, otherwise a number between 0 and 1 for the percential of missed measurements in the meausurement record
    Output:
            train_loader: dataloader with training data
            val_loader: dataloader with validation data
            test_loader: dataloader with test data

            
    '''
    assert data_split <= 1 and data_split > 0, 'data_split must be between 0 and 1'


    measurements, a, b = Xeuler_sim(N_sim = N_data,N = N, g = g,U_s = U_s,a0= a0,b0= b0,delta_t=delta_t, r=r, verbose=verbose)
    #X = np.array([m['X'] for m in measurements])
    a = torch.tensor(a).float()
    b = torch.tensor(b).float()

    # unpack measurements
    X = [m['X'] for m in measurements]
    t = [m['t'] for m in measurements]

    # zero pad measurements
    max_length = max([len(x) for x in X])
    min_length = min([len(x) for x in X])
    if verbose:
        print('Max length:', max_length)
        print('Min length:', min_length)
    X = [np.pad(x, (0, max_length - len(x))) for x in X]
    t = [np.pad(x, (0, max_length - len(x))) for x in t]   

    X = torch.tensor(X).float()
    t = torch.tensor(t).float()
    
    
    #X = torch.tensor(X).float()

    if meter_error is not None:
        # sample random measurements to be missed in the measurement record
        N_missed = int(meter_error*N)
        idx_keep = np.random.choice(N, N - N_missed, replace=False)

        # remove these measurements
        X = X[:,idx_keep]

        
    X = X.unsqueeze(1)
    

    # find label
    #label = (X.sum(axis=-1) > 0).ravel().long()
    

    if data_split != 1:
        # split data index
        idx = np.arange(N_data)
        np.random.shuffle(idx)
        split_idx = int(data_split*N_data)
        idx_train = idx[:split_idx]
        idx_val = idx[split_idx:]
    
        # create dataset
        train_dataset = torch.utils.data.TensorDataset(X[idx_train], t[idx_train], a[idx_train], b[idx_train])
        val_dataset = torch.utils.data.TensorDataset(X[idx_val], t[idx_val], a[idx_val], b[idx_val])

        if meter_error is not None:
            return train_dataset, val_dataset, idx_keep
       
        return train_dataset, val_dataset

    else: # test data
        test_dataset = torch.utils.data.TensorDataset(X, t, a,b)

        if meter_error is not None:
            return test_dataset, idx_keep
        
        return test_dataset
    
    
    

if __name__ == '__main__':
    Us = np.array([[1,0],[0,-1]])
    N = 100 # NOTE: must be fixed for the CNN
    g = 0.25
    a0 = 1/np.sqrt(2)
    delta_t = 0.05
    U_s = np.array([[1,0],[0,-1]])

    # data
    N_data = 1000
    dataset =  create_datasets_sim(N_data = 100,data_split=1, N=N, g=g, U_s = U_s, a0=a0, delta_t = delta_t, verbose=True)
    #dataset = create_datasets_sim(N_data = 1000,data_split = 1,N =  100, g = 0.1,U_s = Us, r = 10, meter_error=None)
  
    # check shapes
   
    sample_X, sample_t, sample_a, sample_b = dataset[0]
    
    print("X shape:", sample_X.shape)
    print("t shape:", sample_t.shape)
    print("a shape:", sample_a.shape)
    print("b shape:", sample_b.shape)
    breakpoint()


  



    
