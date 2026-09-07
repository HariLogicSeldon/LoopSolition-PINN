# Derived from PINNs-Torch's FCN implementation.
# Copyright (c) 2023, Reza Akbarian Bafghi
# SPDX-License-Identifier: BSD-3-Clause
# See ../../LICENSES/PINNs-Torch-BSD-3-Clause.txt and ../../THIRD_PARTY_NOTICES.md.
from typing import Dict, List

import numpy as np
import torch
from torch import nn


class DIYFCN(nn.Module):
    """A simple fully-connected neural net for solving equations.

    In this model, lower and upper bound will be used for normalization of input data
    """
    output_names: List[str]

    def __init__(self, layers, lb, ub, output_names, discrete: bool = False) -> None:
        """Initialize a `FCN` module.

        :param layers: The list indicating number of neurons in each layer.
        :param lb: Lower bound for the inputs.
        :param ub: Upper bound for the inputs.
        :param output_names: Names of outputs of net.
        :param discrete: If the problem is discrete or not.
        """
        super().__init__()

        self.model = self.initalize_net(layers)
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32, requires_grad=False))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32, requires_grad=False))
        self.output_names = output_names
        self.discrete = discrete

    def initalize_net(self, layers: List):
        """Initialize the layers of the neural network.

        :param layers: The list indicating number of neurons in each layer.
        :return: The initialized neural network.
        """

        initializer = nn.init.xavier_uniform_
        net = nn.Sequential()

        input_layer = nn.Linear(layers[0], layers[1])
        initializer(input_layer.weight)

        net.add_module("input", input_layer)
        net.add_module("activation_1", nn.GELU())

        for i in range(1, len(layers) - 2):
            hidden_layer = nn.Linear(layers[i], layers[i + 1])
            initializer(hidden_layer.weight)
            net.add_module(f"hidden_{i + 1}", hidden_layer)
            net.add_module(f"activation_{i + 1}", nn.GELU())

        output_layer = nn.Linear(layers[-2], layers[-1])
        initializer(output_layer.weight)
        net.add_module("output", output_layer)
        return net

    def forward(self, spatial: List[torch.Tensor], time: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a single forward pass through the network.

        :param spatial: List of input spatial tensors.
        :param time: Input tensor representing time.
        :return: A tensor of solutions.
        """

        # Discrete Mode
        if self.discrete:
            if len(spatial) == 2:
                x, y = spatial
                z = torch.cat((x, y), 1)
            elif len(spatial) == 3:
                x, y, z = spatial
                z = torch.cat((x, y, z), 1)
            else:
                z = spatial[0]
            z = 2.0 * (z - self.lb[:-1]) / (self.ub[:-1] - self.lb[:-1]) - 1.0

        # Continuous Mode
        else:
            if len(spatial) == 1:
                x = spatial[0]
                z = torch.cat((x, time), 1)
            elif len(spatial) == 2:
                x, y = spatial
                z = torch.cat((x, y, time), 1)
            else:
                x, y, z = spatial
                z = torch.cat((x, y, z, time), 1)
            z = 2.0 * (z - self.lb) / (self.ub - self.lb) - 1.0

        z = self.model(z)

        # Discrete Mode
        if self.discrete:
            outputs_dict = {name: z for i, name in enumerate(self.output_names)}

        # Continuous Mode
        else:
            outputs_dict = {name: z[:, i: i + 1] for i, name in enumerate(self.output_names)}
        return outputs_dict
