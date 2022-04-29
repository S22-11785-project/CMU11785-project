import torch
import torch.nn as nn

class MLP(torch.nn.Module):
    """ A toy baseline model
    """
    def __init__(
        self, 
        in_size,
        output_size=1,
        hidden_sizes = [64, 256, 512, 256],
        batchnorm = [True, True, True, True],
        activation = ['ReLU', 'ReLU', 'ReLU', 'ReLU'],
        dropout = [False, False, False, True],
    ):
        super(MLP, self).__init__()
        assert len(hidden_sizes) == len(batchnorm) == len(activation) == len(dropout)

        hidden_layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                input_size = in_size
            else:
                input_size = hidden_sizes[i-1]
            hidden_layers.append(nn.Linear(input_size, hidden_sizes[i]))
            if batchnorm[i]:
                hidden_layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            
            hidden_layers.append(self.decide_activation(activation[i]))
            if dropout[i]:
                hidden_layers.append(nn.Dropout(p=0.2))
        output_layer = [nn.Linear(hidden_sizes[-1], output_size)]
        layers = hidden_layers + output_layer
        self.laysers = nn.Sequential(*layers)

    def forward(self, x):
        return self.laysers(x)

    @staticmethod
    def decide_activation(str_activation):
        assert str_activation in ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh', 'Softplus', 'SiLU']
        if str_activation == 'ReLU':
            return nn.ReLU()
        elif str_activation == 'LeakyReLU':
            return nn.LeakyReLU()
        elif str_activation == 'Sigmoid':
            return nn.Sigmoid()
        elif str_activation == 'Tanh':
            return nn.Tanh()
        elif str_activation == 'Softplus':
            return nn.Softplus()
        elif str_activation == 'SiLU':
            return nn.SiLU()

class MLPwCNN(torch.nn.Module):
    """ MLP regressor with CNN feature extractor
    """
    def __init__(self, in_size):
        super(MLPwCNN, self).__init__()
        self.emb = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.AdaptiveAvgPool2d((1, 1))
        )
        self.mlp = MLP(in_size*32)


    def forward(self, x):
        # original x shape: (B, in_features)

        # Convert x to (B, 1, in_features) for Conv1d
        x = x[:, None, :]
        out = self.emb.forward(x)

        # Flatten the output channels to (B, in_features*emb_out_ch)
        out = torch.flatten(out, start_dim=1)
        out = self.mlp(out)
        return out


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, batch_first, dropout_prob=0):
        super(SimpleLSTM, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.is_cuda:
            h0 = torch.cuda.FloatTensor(self.layer_dim, x.size(0), self.hidden_dim).fill_(0).requires_grad_()
            c0 = torch.cuda.FloatTensor(self.layer_dim, x.size(0), self.hidden_dim).fill_(0).requires_grad_()
        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = out[:, -1, :]
        out = self.fc(out)
        return out