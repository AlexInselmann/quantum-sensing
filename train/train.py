import torch
import torch.nn.functional as F



def train_model(model, train_loader, val_loader, m, epochs, loss_fn, optimizer, device, steady_state = True, verbose=True):
    '''
    Train a model with a given train_loader and validate it with val_loader.

    Input:
            model: model to be trained
            train_loader: dataloader with training data
            val_loader: dataloader with validation data
            m: number of measurements per simulation
            epochs: number of epochs to train
            loss_fn: loss function to be optimized
            optimizer: optimizer to be used
            device: device to run the model
            steady_state: For the classification task is the label end coefficients of a or coefficients at m.
            verbose: print loss and accuracy during training
    Output:
            model: trained model
            history: dictionary with loss and accuracy for each epoch
    '''
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    find_label_q = lambda am,bm: (abs(am)**2 > abs(bm)**2).ravel().long()
    find_label_ss = lambda x: (x.sum(axis=-1) > 0).ravel().long()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for X, t, a,b  in train_loader:
            X, t = X.to(device), t.to(device)
            Xm = X[:,:, :m] # keep first m measurements


            # Model prediction, loss and optimization
            optimizer.zero_grad()
            pred = model(Xm)

            # label
            if steady_state:
                y = find_label_ss(X)
            else:   
                y = find_label_q(a[:,m], b[:,m]) #(abs(a[:,m])**2 > abs(b[:,m])**2).float() #torch.tensor(a[:,m]**2,  b[:,m]**2).long()

            # loss and optimization
            loss = loss_fn(pred, y) 
            loss.backward()
            optimizer.step()

            # Metrics calculation
            train_loss += loss.item()
            train_acc += (pred.argmax(1) == y).sum().item()
    

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for X, t, a,b in val_loader:
                X, t = X.to(device), t.to(device)
                Xm = X[:,:, :m]

                # Model prediction and loss calculation
                pred = model(Xm)

                # label
                if steady_state:
                    y = find_label_ss(X)
                else:   
                    y = find_label_q(a[:,m], b[:,m]) #(abs(a[:,m])**2 > abs(b[:,m])**2).float() #torch.tensor(a[:,m]**2,  b[:,m]**2).long()

                # Metrics calculation
                val_loss += loss.item()
                val_acc += (pred.argmax(1) == y).sum().item()

            val_loss /= len(val_loader.dataset)
            val_acc /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if verbose:
            print(f'Epoch {epoch+1}/{epochs}:')
            if steady_state:
                print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
                print(f'Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n')
            else:
                print(f'Train - Loss: {train_loss:.4f}')
                print(f'Val - Loss: {val_loss:.4f}\n')


    return model, history

def train_model_reg(model, train_loader, val_loader, m, epochs, loss_fn, optimizer, device, verbose=True):
    '''
    Train a model with a given train_loader and validate it with val_loader.

    Input:
            model: model to be trained
            train_loader: dataloader with training data
            val_loader: dataloader with validation data
            m: number of measurements per simulation
            epochs: number of epochs to train
            loss_fn: loss function to be optimized
            optimizer: optimizer to be used
            device: device to run the model
            steady_state: is the task classification set to true else false for regression on the states. 
            verbose: print loss and accuracy during training
    Output:
            model: trained model
            history: dictionary with loss and accuracy for each epoch
    '''
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    find_label = lambda am,bm: (abs(am)**2 > abs(bm)**2).ravel().long()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for X, t, a,b  in train_loader:
            X, t = X.to(device), t.to(device)
            X = X[:,:, :m] # keep first m measurements


            # Model prediction, loss and optimization
            optimizer.zero_grad()
            pred = model(X)
           
            
            a = a[:,1:] # remove a0
            loss = loss_fn(pred, a)
            loss.backward()
            optimizer.step()

            # Metrics calculation
            train_loss += loss.item()
            
    

        train_loss /= len(train_loader.dataset)
        if steady_state:
            train_acc /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for X, t, a,b in val_loader:
                X, t = X.to(device), t.to(device)
                X = X[:,:, :m]

                # Model prediction and loss calculation
                pred = model(X)
                
                a = a[:,1:]
                loss = loss_fn(pred, a)

                # Metrics calculation
                val_loss += loss.item()
        
            val_loss /= len(val_loader.dataset)
            if steady_state:
                val_acc /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if verbose:
            print(f'Epoch {epoch+1}/{epochs}:')
            if steady_state:
                print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
                print(f'Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n')
            else:
                print(f'Train - Loss: {train_loss:.4f}')
                print(f'Val - Loss: {val_loss:.4f}\n')


    return model, history

if __name__ == '__main__':
    from model import *
    from dataloader import *
    # psysical model parameters
    N = 100 # NOTE: must be fixed for the CNN
    g = 0.25
    a0 = 1/np.sqrt(2)
    U_s = np.array([[1,0],[0,1]])
    delta_t = 1#0.05

    find_probs = lambda a, b: np.maximum(abs(a)**2, abs(b)**2).mean(axis=0)

    # data
    N_data = 1000


    # ML parameters
    epochs = 10 
    lr = 1e-4
    batch_size = 128
    folds = 10

    m = 90
    steady_state = True
    verbose = True

    # create dataset
    train_dataset, val_dataset =  create_datasets_sim(N_data,0.8, N=N, g=g, U_s = U_s, a0=a0, delta_t = delta_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    # model
    model = model = vggSmall(num_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train
    model, stat = train_model(model, train_loader, val_loader, m, epochs, loss_fn, optimizer, device , verbose=False)
    print(stat)
