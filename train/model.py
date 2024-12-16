import torch.nn as nn
import torch
import numpy as np

# simple LSTM model
class simpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(simpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=1)
        )
    
   
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[-1])
        return
    

# simple CNN 1d model
class VGG(nn.Module):

    def __init__(self, features, num_classes, arch='vgg'):
        super(VGG, self).__init__()
        self.arch = arch
        self.features = features
        self.pool = nn.AdaptiveAvgPool1d(4)

        num_param = 256 * 4 if arch == 'vggsmall' else 512 * 4
       
        self.classifier = nn.Sequential(
            nn.Linear(num_param, 512), # if you change to another VGG, change 256 to 512
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, num_classes),
        )
        self.sm = nn.Softmax(dim=1)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.sm(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'small': [64, 'M', 128, 256, 'M'],
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vggSmall(**kwargs):
    model = VGG(make_layers(cfg['small']), arch='vggsmall', **kwargs)
    return model

def vgg11(**kwargs):
    model = VGG(make_layers(cfg['A']), arch='vgg11', **kwargs)
    return model

def vgg13bn(**kwargs):
    model = VGG(make_layers(cfg['B'], batch_norm=True), arch='vgg13bn', **kwargs)
    return model

def vgg16bn(**kwargs):
    model = VGG(make_layers(cfg['D'], batch_norm=True), arch='vgg16bn', **kwargs)
    return model

def vgg19bn(**kwargs):
    model = VGG(make_layers(cfg['E'], batch_norm=True), arch='vgg19bn', **kwargs)
    return model

if __name__ == '__main__':
    print('testing models')