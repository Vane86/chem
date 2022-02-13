from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.optim import Adam


class Model(ABC):

    @abstractmethod
    def fit(self, features, labels, **kwargs):
        pass

    @abstractmethod
    def predict(self, features):
        pass


class FCModel(nn.Module, Model):

    def __init__(self, inputs, outputs, threshold=0.5):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(inputs, 256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(128, outputs)
        self.act3 = nn.Sigmoid()

        self._threshold = threshold

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        return x

    # kwargs:
    # lr - learning rate,
    # epochs - # of epochs,
    # history - list for loss history;
    def fit(self, features, labels, **kwargs):
        optimizer = Adam(self.parameters(), lr=kwargs.get('lr', 1e-4), weight_decay=kwargs.get('weight_decay', 0))
        criterion = nn.BCELoss()
        for epoch in range(kwargs.get('epochs', 10)):
            optimizer.zero_grad()
            loss = criterion(self(features).flatten(), labels)
            loss.backward()
            optimizer.step()
            if 'history' in kwargs:
                kwargs['history'].append(loss.item())

    def predict(self, features):
        return self(features) > self._threshold
