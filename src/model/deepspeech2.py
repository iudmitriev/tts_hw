import torch
from torch import nn

from einops import rearrange 

from src.base import BaseModel


class DeepSpeech2Model(BaseModel):
    def __init__(self, n_feats, n_class, hidden_channels=32, gru_hidden_size=800, gru_num_layers=5, bidirectional=True, linear_hidden_size=1600, **batch):
        super().__init__(n_feats, n_class, **batch)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=(11, 41), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(11, 21), stride=1, padding='same')
        )
        self.rnn = nn.GRU(input_size=hidden_channels*n_feats, hidden_size=gru_hidden_size, num_layers=gru_num_layers, 
                          bidirectional=bidirectional, batch_first=True)

        if bidirectional:
            linear_in_features = 2 * gru_hidden_size
        else:
            linear_in_features = gru_hidden_size
        self.full = nn.Sequential(
            nn.Linear(in_features=linear_in_features, out_features=linear_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=linear_hidden_size, out_features=n_class),
        )

    def forward(self, spectrogram, **batch):
        x = spectrogram.unsqueeze(dim=1)
        x = self.conv(x)
        x = rearrange(x, 'b c f l -> b l (c f)')
        x = self.rnn(x)[0]
        x = self.full(x)
        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
