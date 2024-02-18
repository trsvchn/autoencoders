import torch
import torch.nn as nn


class FooBar(nn.Module):
    """Dummy Net for testing/debugging.
    """

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(784, 784)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        return x
