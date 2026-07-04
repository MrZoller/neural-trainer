"""Track 1 models, written to be read (DESIGN.md §4: "the code itself is a
teaching artifact"). If you understood neural-viz's App.jsx, this is the same
network expressed in PyTorch.
"""

import torch
from torch import nn


class SimpleMLP(nn.Module):
    """A multilayer perceptron for MNIST.

    The forward pass is the same three-step dance neural-viz animates:

        z = W·x + b        (nn.Linear stores W and b; W has shape [out, in],
                            exactly the layout neural-viz exports)
        a = relu(z)        (the nonlinearity — without it, stacking Linear
                            layers would collapse into one big linear map,
                            which is why XOR needed a hidden layer)
        ...repeat per hidden layer, then one final Linear with NO activation.

    The output layer emits raw scores ("logits"), not probabilities. Softmax
    lives inside nn.CrossEntropyLoss during training (log-sum-exp is more
    numerically stable there) and is applied explicitly only at inference —
    the same trick as neural-viz's BCE+sigmoid shortcut, generalized to 10
    classes.
    """

    def __init__(self, in_features: int = 28 * 28, hidden: tuple[int, ...] = (128, 64),
                 classes: int = 10):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            prev = width
        layers.append(nn.Linear(prev, classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Images arrive as [batch, 1, 28, 28]; an MLP sees a flat vector of
        # 784 pixel intensities — it has no notion of 2D neighborhoods.
        # (That blindness is what convolutions fix, later in Track 1.)
        return self.net(x.flatten(start_dim=1))

    def linear_layers(self) -> list[tuple[str, nn.Linear]]:
        """Named Linear layers, for per-layer stats in the UI."""
        return [(f"net.{i}", m) for i, m in enumerate(self.net) if isinstance(m, nn.Linear)]

    def relu_layers(self) -> list[tuple[str, nn.ReLU]]:
        return [(f"net.{i}", m) for i, m in enumerate(self.net) if isinstance(m, nn.ReLU)]
