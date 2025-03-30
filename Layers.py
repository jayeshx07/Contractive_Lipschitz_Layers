import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Union
from Utils import SpectralNormPowerMethod  
from typing import List, Tuple, Union
# --- Constants ---

MAX_ITER = 1  # Default max iterations for training
EVAL_MAX_ITER = 100  # Default max iterations for evaluation


# --- Custom Layers ---

class ConvexPotentialLayerConv(nn.Module):
    """
    A convolutional layer based on convex potential theory.

    Args:
        input_size (Tuple[int, ...]): Input size for spectral norm calculation.
        cin (int): Number of input channels.
        cout (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        epsilon (float): Small value for numerical stability.
    """

    def __init__(self, input_size: Tuple[int, ...], cin: int, cout: int,
                 kernel_size: int = 3, stride: int = 1,
                 epsilon: float = 1e-4):
        super().__init__()

        self.activation = nn.ReLU(inplace=False)  # Use out-of-place ReLU
        self.stride = stride
        self.register_buffer('eval_sv_max',
                             torch.tensor([0.0]))  # Use torch.tensor

        self.kernel = nn.Parameter(torch.zeros(cout, cin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(cout))

        self.pm = SpectralNormPowerMethod(input_size)
        self.train_max_iter = MAX_ITER
        self.eval_max_iter = EVAL_MAX_ITER

        # Kaiming (He) initialization for weights and biases
        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the convex potential layer."""
        res = F.conv2d(x, self.kernel, bias=self.bias, stride=self.stride,
                       padding=1)
        res = self.activation(res)
        res = F.conv_transpose2d(res, self.kernel, stride=self.stride,
                                 padding=1)

        if self.training:
            self.eval_sv_max.zero_()  # Reset eval_sv_max during training
            sv_max = self.pm(self.kernel, self.train_max_iter)
            h = 2 / (sv_max ** 2 + self.epsilon)
        else:
            if self.eval_sv_max == 0:
                self.eval_sv_max.copy_(
                    self.pm(self.kernel, self.eval_max_iter))  # Use .copy_
            h = 2 / (self.eval_sv_max ** 2 + self.epsilon)

        out = (1+ 0.2)*x - h * res
        return out


class ConvexPotentialLayerLinear(nn.Module):
    """
    A linear layer based on convex potential theory.

    Args:
        cin (int): Number of input features.
        cout (int): Number of output features.
        epsilon (float): Small value for numerical stability.
    """

    def __init__(self, cin: int, cout: int, epsilon: float = 1e-4):
        super().__init__()
        self.activation = nn.ReLU(inplace=False)  # Use out-of-place ReLU
        self.register_buffer('eval_sv_max', torch.tensor([0.0])) # Use torch.tensor

        self.weights = nn.Parameter(torch.zeros(cout, cin))
        self.bias = nn.Parameter(torch.zeros(cout))

        self.pm = SpectralNormPowerMethod((1, cin))
        self.train_max_iter = MAX_ITER
        self.eval_max_iter = EVAL_MAX_ITER

        # Kaiming (He) initialization
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.epsilon = epsilon
        #self.alpha = nn.Parameter(torch.zeros(1))  # Removed unused parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the convex potential layer."""
        res = F.linear(x, self.weights, self.bias)
        res = self.activation(res)
        res = F.linear(res, self.weights.t()) # no need to detach here

        if self.training:
            self.eval_sv_max.zero_()  # Reset eval_sv_max during training
            sv_max = self.pm(self.weights, self.train_max_iter)
            h = 2 / (sv_max ** 2 + self.epsilon)
        else:
            if self.eval_sv_max == 0:
                self.eval_sv_max.copy_(self.pm(self.weights,
                                               self.eval_max_iter))  # Use .copy_
            h = 2 / (self.eval_sv_max ** 2 + self.epsilon)

        out = x - h * res
        return out


class Normalize(nn.Module):
    """
    Normalization layer.

    Args:
        mean (list or tuple): Mean for each channel.
        std (list or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean: Union[List, Tuple], std: Union[List, Tuple]):
        super().__init__()
        self.mean = torch.tensor(mean)  # Use torch.tensor
        self.std = torch.tensor(std)  # Use torch.tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes the input tensor."""
        # Reshape mean and std for broadcasting
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[
                                                                None, :, None, None]


class NormalizedModel(nn.Module):
    """
    Combines a model with a normalization layer.

    Args:
        model (nn.Module): The base model.
        mean (list or tuple): Mean for normalization.
        std (list or tuple): Standard deviation for normalization.
    """

    def __init__(self, model: nn.Module, mean: Union[List, Tuple],
                 std: Union[List, Tuple]):
        super().__init__()
        self.model = model
        self.normalize = Normalize(mean, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies normalization and then the base model."""
        return self.model(self.normalize(x))


class PaddingChannels(nn.Module):
    """
    Pads or clones channels of the input tensor.

    Args:
        ncout (int): Desired number of output channels.
        ncin (int): Number of input channels.
        mode (str): 'zero' for zero-padding, 'clone' for channel cloning.
    """

    def __init__(self, ncout: int, ncin: int = 3, mode: str = "zero"):
        super().__init__()
        self.ncout = ncout
        self.ncin = ncin
        if mode not in ("zero", "clone"):
            raise ValueError("mode must be 'zero' or 'clone'")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pads or clones channels."""
        if self.mode == "clone":
            # Clone and scale for unit variance
            repeats = int(self.ncout / self.ncin)
            remainder = self.ncout % self.ncin
            if remainder != 0 :
                raise ValueError("Output channels is not a multiple of input channels when using 'clone' mode")
            return x.repeat(1, repeats, 1, 1) / np.sqrt(repeats)

        elif self.mode == "zero":
            bs, _, size1, size2 = x.shape
            out = torch.zeros(bs, self.ncout, size1, size2, device=x.device)
            out[:, :self.ncin] = x  # Fill initial channels with input
            return out

        return x # Should not reach here, added return for type checker



class PoolingLinear(nn.Module):
    """
    Linear layer with input pooling.

    Args:
        ncin (int): Number of input features.
        ncout (int): Number of output features.
        agg (str): Aggregation method: 'mean', 'max', or 'trunc'.
    """

    def __init__(self, ncin: int, ncout: int, agg: str = "mean"):
        super().__init__()
        self.ncout = ncout
        self.ncin = ncin
        if agg not in ("mean", "max", "trunc"):
            raise ValueError("agg must be 'mean', 'max', or 'trunc'")
        self.agg = agg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies pooling followed by a linear transformation."""
        if self.agg == "trunc":
            return x[:, :self.ncout]

        k = self.ncin / self.ncout
        if not k.is_integer():
            raise ValueError(f"Input features ({self.ncin}) must be divisible by output features ({self.ncout}) for agg='mean' or 'max'")
        k = int(k)
        out = x[:, :self.ncout * k]
        out = out.view(x.size(0), self.ncout, k)

        if self.agg == "mean":
            out = np.sqrt(k) * out.mean(dim=2)  # L2 normalization factor
        elif self.agg == "max":
            out = out.max(dim=2)[0]

        return out


class LinearNormalized(nn.Linear):
    """
    Linear layer with L2-normalized weights.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the linear transformation with normalized weights."""
        Q = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x, Q, self.bias)