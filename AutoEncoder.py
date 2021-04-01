#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm

from tqdm import trange

class AutoEncoder(nn.Module):
    def __init__(self, num_features, resolution, batch_size, lr_encoder=1e-3, lr_decoder=1e-3):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, 3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, num_features, 3, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU()
        )
        self.shape_encoded = self.encoder(torch.zeros(batch_size, 1, *resolution)).shape

        prod_shape = np.array(self.shape_encoded).prod().astype(int)

        self.decoder_shaper = nn.Sequential(
            nn.Linear(num_features, prod_shape),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_features, 16, 3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, 3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, 3, bias=False),
            nn.ReLU()
        )

        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=lr_encoder)
        self.opt_decoder = optim.Adam(self.decoder.parameters(), lr=lr_decoder)

        self.crit = nn.MSELoss()

    def encode(self, x):
        x = self.encoder(x)
        x = x.mean(dim=(2, 3))

        return x

    def decode(self, x):
        x = self.decoder_shaper(x)
        print(tuple(self.shape_encoded))
        x = x.view(self.shape_encoded)
        x = self.decoder(x)

        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)

        return x

    def train(self, x):
        self.opt_encoder.zero_grad()
        self.opt_decoder.zero_grad()

        output = self.forward(x)
        loss = self.crit(output, x)

        loss.backward()
        self.opt_encoder.step()
        self.opt_decoder.step()

        return loss.item()

