import torch
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, output_dim=512,
                 num_layer=2, bias=True):
        super(FCN, self).__init__()

        layer = list()
        for i in range(0, num_layer):
            ini = hidden_dim
            fin = hidden_dim

            if i == 0:
                ini = input_dim
            if i == num_layer - 1:
                fin = output_dim
            w_tmp = nn.Linear(ini, fin, bias=bias)
            nn.init.xavier_normal_(w_tmp.weight.data)
#            nn.init.kaiming_normal_(w_tmp.weight.data)
            nn.init.normal_(w_tmp.bias.data)

            layer.append(w_tmp)
            if i != num_layer-1:
                layer.append(nn.ReLU())
        self.layer = nn.Sequential(*layer)

    def forward(self, X):

        Y = self.layer(X)
        return Y


class Net(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim1=1024, latent_dim=512,
                 hidden_dim2=1024, output_dim=1, num_layer=2, bias=True):
        super(Net, self).__init__()

        self.fc_i = FCN(input_dim=input_dim, hidden_dim=hidden_dim1,
                        output_dim=latent_dim, num_layer=2)
        self.fc_o = FCN(input_dim=latent_dim*2, hidden_dim=hidden_dim2,
                        output_dim=output_dim, num_layer=2)

    def fo_i(self, x):
        z = self.fc_i(x)
        return z

    def fo_f(self, z1, z2):
        z = torch.concatenate([z1, z2], axis=1)
        z = torch.relu(z)
        y = self.fc_o(z)
        return y

    def forward(self, x1, x2):
        z1 = self.fo_i(x1)
        z2 = self.fo_i(x2)
        y = self.fo_f(z1, z2)

        return y
