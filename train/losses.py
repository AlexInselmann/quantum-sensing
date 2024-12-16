import torch

class weighted_MSEloss(torch.nn.Module):
    def __init__(self, weights, reduction='mean'):
        super(weighted_MSEloss, self).__init__()
        self.weights = weights
        self.reduction = reduction

      
        # make sure the weight sum to 1
        self.weights = self.weights / self.weights.sum()
        
    def forward(self, outputs, targets):
        if self.reduction == 'mean':
            return torch.mean(self.weights * (outputs - targets)**2)
        elif self.reduction == 'sum':
            return torch.sum(self.weights * (outputs - targets)**2)
        else:
            raise ValueError('reduction method not supported')

    

if __name__ == "__main__":

    loss_fn = weighted_MSEloss(weights=torch.tensor([1,2,3], dtype=torch.float32))
    outputs = torch.tensor([1, 2, 4], dtype=torch.float32)
    targets = torch.tensor([1, 2, 3], dtype=torch.float32)

    loss = loss_fn(outputs, targets)
    print(loss)

    # test with batch size
    print('Test with batch size')
    loss_fn = weighted_MSEloss(weights=torch.tensor([1,2,3], dtype=torch.float32))
    outputs = torch.tensor([[0, 2, 4], [0, 2, 4]], dtype=torch.float32)
    targets = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float32)
    print(f"outputs: {outputs.shape}, targets: {targets.shape}")
    loss = loss_fn(outputs, targets)
    print(loss)
