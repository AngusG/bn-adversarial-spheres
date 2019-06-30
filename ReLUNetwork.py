import torch
import torch.nn as nn


class ReLUNetwork(nn.Module):
    def __init__(self, input_size, num_layers, num_units, num_classes, do_batch_norm=False):
        super(ReLUNetwork, self).__init__()
        self.do_batch_norm = do_batch_norm
        self.input_layer = nn.Linear(input_size, num_units, bias=True)
        self.features = self._make_layers(num_layers, num_units)
        self.classifier = nn.Linear(num_units, num_classes, bias=True)

    def forward(self, inputs):
        """Forward pass, returns outputs of each layer. Use last out (final) for backprop!"""
        out = self.input_layer(inputs)
        out = self.features(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, num_layers, num_units):
        layers = []
        for i in range(num_layers):
            if self.do_batch_norm:
                layers += [nn.BatchNorm1d(num_units, momentum=None),
                           nn.Linear(num_units, num_units, bias=True),
                           nn.ReLU()]
            else:
                layers += [nn.Linear(num_units, num_units, bias=True),
                           nn.ReLU()]
        return nn.Sequential(*layers)
