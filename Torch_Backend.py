import torch
import torchvision
import numpy as np
import torch.nn as nn
from functools import singledispatch
from typing import Dict, Tuple, List, Any, Callable

# Assuming 'build_graph' from the refactored core.py is available

# --- Device Configuration ---

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Overriding Generic Functions for Torch ---

@singledispatch
def cat(*xs):
    """Generic function to concatenate arrays or tensors."""
    raise NotImplementedError(f"cat not implemented for type {type(xs[0])}")

@cat.register(torch.Tensor)
def _(*xs: torch.Tensor) -> torch.Tensor:
    return torch.cat(xs)

@singledispatch
def to_numpy(x):
    """Generic function to convert a tensor to a NumPy array."""
    raise NotImplementedError(f"to_numpy not implemented for type: {type(x)}")

@to_numpy.register(torch.Tensor)
def _(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


# --- Dataset Loading Functions ---

def cifar10(root: str) -> Dict[str, Dict[str, Any]]:
    """Loads CIFAR-10 dataset, handling different torchvision versions."""
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    # Handle potential attribute name differences in torchvision
    train_data_key = 'train_data' if hasattr(train_set, 'train_data') else 'data'
    train_labels_key = 'train_labels' if hasattr(train_set,
                                                'train_labels') else 'targets'
    test_data_key = 'test_data' if hasattr(test_set, 'test_data') else 'data'
    test_labels_key = 'test_labels' if hasattr(test_set,
                                                'test_labels') else 'targets'
    return {
        'train': {'data': getattr(train_set, train_data_key),
                  'labels': getattr(train_set, train_labels_key)},
        'test': {'data': getattr(test_set, test_data_key),
                 'labels': getattr(test_set, test_labels_key)}
    }


def cifar100(root: str) -> Dict[str, Dict[str, Any]]:
    """Loads CIFAR-100 dataset, handling different torchvision versions."""
    train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True)
    # Handle potential attribute name differences in torchvision
    train_data_key = 'train_data' if hasattr(train_set, 'train_data') else 'data'
    train_labels_key = 'train_labels' if hasattr(train_set,
                                                'train_labels') else 'targets'
    test_data_key = 'test_data' if hasattr(test_set, 'test_data') else 'data'
    test_labels_key = 'test_labels' if hasattr(test_set,
                                                'test_labels') else 'targets'

    return {
        'train': {'data': getattr(train_set, train_data_key),
                  'labels': getattr(train_set, train_labels_key)},
        'test': {'data': getattr(test_set, test_data_key),
                 'labels': getattr(test_set, test_labels_key)}
    }


# --- Batch Loading Class ---

class Batches:
    """
    Provides iterable batches of data from a dataset.

    Args:
        dataset: The dataset to load.
        batch_size: The size of each batch.
        shuffle: Whether to shuffle the data.
        set_random_choices: Whether to set random choices for data augmentation.
        num_workers: Number of worker processes for data loading.
        drop_last: Whether to drop the last incomplete batch.
        return_perturbation: if the dataset has pertrubation
    """

    def __init__(self, dataset, batch_size: int, shuffle: bool,
                 set_random_choices: bool = False, num_workers: int = 0,
                 drop_last: bool = False, return_perturbation: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.return_perturbation = return_perturbation
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        if self.return_perturbation:
            # Modified to return a dictionary
            return ({'input': x.to(device).float(), 'target': y.to(device).long(),
                    'delta': delta.to(device).float(), 'index': idx}  # Include delta and idx
                    for (x, y, delta, idx) in self.dataloader)
        else:
             return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x, y) in self.dataloader)

    def __len__(self) -> int:
        return len(self.dataloader)


# --- PyTorch Module Building Blocks ---

class Identity(nn.Module):
    """Identity module (returns input unchanged)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Mul(nn.Module):
    """Multiplication module (scales input by a constant weight)."""

    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class Flatten(nn.Module):
    """Flattens the input tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)  # Correct flattening


class Add(nn.Module):
    """Addition module (adds two tensors)."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class Concat(nn.Module):
    """Concatenation module (concatenates tensors along the channel dimension)."""

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        return torch.cat(xs, dim=1)  # Specify dimension for concatenation


class Correct(nn.Module):
    """Calculates the number of correct predictions."""

    def forward(self, classifier: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return (classifier.max(dim=1)[1] == target).float()


def batch_norm(num_channels: int, bn_bias_init: float | None = None,
                 bn_bias_freeze: bool = False,
                 bn_weight_init: float | None = None,
                 bn_weight_freeze: bool = False) -> nn.BatchNorm2d:
    """Creates a BatchNorm2d layer with configurable initialization and freezing."""
    m = nn.BatchNorm2d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False
    return m


# --- Network Class ---

class Network(nn.Module):
    """
    Builds a network from a graph representation.

    Args:
        net (dict):  A dictionary representing the network architecture,
            as produced by `build_graph`.
    """

    def __init__(self, net: Dict):
        super().__init__()
        self.graph = build_graph(net)
        for name, (module, _) in self.graph.items():
            setattr(self, name, module)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[
        str, torch.Tensor]:
        """
        Performs a forward pass through the network.

        Args:
            inputs (dict):  A dictionary containing the input tensor(s).

        Returns:
            dict:  A dictionary containing the outputs of all nodes in the graph.
        """
        cache: Dict[str, torch.Tensor] = dict(inputs)
        for node_name, (_, input_names) in self.graph.items():
            module = getattr(self, node_name)
            input_tensors = [cache[input_name] for input_name in input_names]
            cache[node_name] = module(*input_tensors)  # Pass inputs to module
        return cache

    def half(self) -> "Network":
        """
        Converts the network to half-precision, excluding BatchNorm layers.
        Deprecated in favor to FP16 training
        """
        for module in self.children():
            if not isinstance(module, nn.BatchNorm2d):
                module.float()  # Should be .half(), but corrected for consistency
        return self


trainable_params = lambda model: filter(lambda p: p.requires_grad,
                                        model.parameters())


# --- Optimizer Class ---

class TorchOptimiser:
    """
    Wrapper for PyTorch optimizers, allowing for scheduled parameters.

    Args:
        weights: The model parameters to optimize.
        optimizer: The PyTorch optimizer class (e.g., torch.optim.SGD).
        step_number:  The initial step number.
        **opt_params:  Keyword arguments for the optimizer,
            which can be callables (functions of step number).
    """

    def __init__(self, weights: List[torch.nn.Parameter],
                 optimizer: Callable[..., torch.optim.Optimizer],
                 step_number: int = 0, **opt_params):
        self.weights = weights
        self.step_number = step_number
        self.opt_params = opt_params
        self._opt = optimizer(weights, **self.param_values())

    def param_values(self) -> Dict:
        """Evaluates the optimizer parameters, handling callable values."""
        return {k: v(self.step_number) if callable(v) else v for k, v in
                self.opt_params.items()}

    def step(self) -> None:
        """Performs an optimizer step and updates the step number."""
        self.step_number += 1
        self._opt.param_groups[0].update(**self.param_values())
        self._opt.step()

    def __repr__(self) -> str:
        return repr(self._opt)


def SGD(weights, lr: float = 0.0, momentum: float = 0.0,
        weight_decay: float = 0.0, dampening: float = 0.0,
        nesterov: bool = False) -> TorchOptimiser:
    """Creates an SGD optimizer with the specified parameters."""
    return TorchOptimiser(weights, torch.optim.SGD, lr=lr, momentum=momentum,
                          weight_decay=weight_decay, dampening=dampening,
                          nesterov=nesterov)