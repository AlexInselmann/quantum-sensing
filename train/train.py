import torch

def train_model(model, train_loader, val_loader, m, epochs, loss_fn, optimizer, device, verbose=True):
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
            verbose: print loss and accuracy during training
    Output:
            model: trained model
            history: dictionary with loss and accuracy for each epoch
    '''
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            X = X[:,:, :m] # keep first m measurements

            # Model prediction, loss and optimization
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            # Metrics calculation
            train_loss += loss.item()
            train_acc += (y_pred.argmax(1) == y).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                X = X[:,:, :m]

                # Model prediction and loss calculation
                y_pred = model(X)
                loss = loss_fn(y_pred, y)

                # Metrics calculation
                val_loss += loss.item()
                val_acc += (y_pred.argmax(1) == y).sum().item()
            val_loss /= len(val_loader.dataset)
            val_acc /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if verbose:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
            print(f'Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n')


    return model, history