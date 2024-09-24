import torch

from torch import nn

class Transformer_Encoder(nn.Module):
    def __init__(self, n_layers, n_heads):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(190,n_heads,batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,num_layers=n_layers)
        self.fc =  nn.Sequential(nn.Flatten(),
                                 nn.Linear(7600,128),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Linear(128, 16),
                                 nn.BatchNorm1d(16),
                                 nn.ReLU(),
                                 nn.Linear(16,1)
                                 )
    def forward(self,X):
        if len(X.shape) != 3:
            X = X.reshape(-1,40,190)
        X = self.encoder(X)
        logits = self.fc(X)
        return logits