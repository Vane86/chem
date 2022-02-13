from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.optim import Adam


class Model(ABC):

    @abstractmethod
    def fit(self, features, labels, **kwargs):
        pass

    @abstractmethod
    def predict(self, features):
        pass


class FCLayer(nn.Module):

    def __init__(self, inputs, outputs, p):
        super().__init__()
        self.fc1 = nn.Linear(inputs, outputs)
        self.norm1 = nn.BatchNorm1d(outputs)
        self.drop1 = nn.Dropout(p=p)
        self.act1 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.drop1(x)
        x = self.act1(x)
        return x


class FCModel(nn.Module, Model):

    def __init__(self, inputs, outputs, p=0.5, threshold=0.5):
        nn.Module.__init__(self)
        self.l1 = FCLayer(inputs, 256, p=p)
        # self.l2 = FCLayer(256, 256, p=p)
        self.fc3 = nn.Linear(256, outputs)
        self.act3 = nn.Sigmoid()

        self._threshold = threshold

    def forward(self, x):
        x = self.l1(x)
        # x = self.l2(x)
        x = self.act3(self.fc3(x))
        return x

    # kwargs:
    # lr - learning rate,
    # epochs - # of epochs,
    # history - list for loss history;
    def fit(self, features, labels, **kwargs):
        self.train()
        optimizer = Adam(self.parameters(), lr=kwargs.get('lr', 1e-4), weight_decay=kwargs.get('weight_decay', 0))
        criterion = nn.BCELoss()
        for epoch in range(kwargs.get('epochs', 10)):
            optimizer.zero_grad()
            loss = criterion(self(features).flatten(), labels)
            loss.backward()
            optimizer.step()
            if 'history' in kwargs:
                kwargs['history'].append(loss.item())
        self.eval()

    def predict_prob(self, features):
        return self(features)

    def predict(self, features):
        return self(features) >= self._threshold
