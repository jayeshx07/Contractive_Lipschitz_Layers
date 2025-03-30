"""
Core utilities and neural network training framework.

This module provides a streamlined and optimized implementation for training 
deep learning models, specifically tailored for CIFAR-10 but adaptable to other datasets.
It includes utilities for data preprocessing, augmentation, model definition, 
training loop management, and logging.  The design emphasizes efficiency, readability,
and extensibility.

Key features:
- Efficient data loading and preprocessing.
- Flexible data augmentation using composable transformations.
- A clean and modular approach to defining neural network architectures.
- A robust training loop with integrated logging and timing.
- Support for standard optimization algorithms.

Based on https://github.com/davidcpage/cifar10-fast
"""

import time
import numpy as np
import torch
from collections import namedtuple
from functools import singledispatch
from typing import NamedTuple, List, Dict, Tuple, Any, Callable, Union
import torch.nn.functional as F
from Torch_Backend import cifar10, cifar100, Batches

# --- Configuration and Constants ---

cifar10_mean = (0.4914, 0.4822, 0.4465)  # CIFAR-10 mean RGB values
cifar10_std = (0.2471, 0.2435, 0.2616)  # CIFAR-10 std dev RGB values
sep = '_'  # Separator for constructing node identifiers


# --- Utility Classes ---

class Timer:
    """Measures elapsed time, accumulating total duration."""

    def __init__(self):
        self._start_times: List[float] = []
        self.total_time: float = 0.0

    def start(self) -> None:
        """Starts a new timing interval."""
        self._start_times.append(time.time())

    def __call__(self, include_in_total: bool = True) -> float:
        """Returns time since last start, optionally adding to total."""
        if not self._start_times:
            raise RuntimeError("Timer not started.")
        duration = time.time() - self._start_times[-1]
        if include_in_total:
            self.total_time += duration
        return duration


def localtime() -> str:
    """Gets current time as 'YYYY-MM-DD HH:MM:SS'."""
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


class TableLogger:
    """Collects and displays training metrics in a structured table."""

    def __init__(self):
        self.keys: List[str] = []  # Set on first use

    def append(self, output: Dict[str, Any]) -> None:
        """Adds a new metric set; prints a formatted row."""
        if not self.keys:
            self.keys = list(output.keys())
            header = '  '.join(f'{name:>12s}' for name in self.keys)
            print(header)

        values = [output[name] for name in self.keys]
        formatted_values = [f'{v:12.4f}' if isinstance(v, float) else f'{v:12}'
                            for v in values]
        print('  '.join(formatted_values))


# --- Data Handling ---

def normalise(x: np.ndarray, mean: Tuple[float, ...] = cifar10_mean,
              std: Tuple[float, ...] = cifar10_std) -> np.ndarray:
    """Normalizes an image to zero mean and unit variance."""
    x_f32 = x.astype(np.float32)
    mean_arr = np.array(mean, dtype=np.float32) * 255
    std_arr = np.array(std, dtype=np.float32) * 255
    return (x_f32 - mean_arr) / std_arr


def dont_normalise(x: np.ndarray, mean: Tuple[float, ...] = cifar10_mean,
                   std: Tuple[float, ...] = cifar10_std) -> np.ndarray:
    """Identity function (no standardization).  API consistency."""
    return x.astype(np.float32)  # Ensure consistent dtype


def pad(x: np.ndarray, border: int = 4) -> np.ndarray:
    """Adds reflection padding around image borders."""
    return np.pad(x, ((0, 0), (border, border),
                         (border, border), (0, 0)), mode='reflect')


def transpose(x: np.ndarray, source: str = 'NHWC',
                 target: str = 'NCHW') -> np.ndarray:
    """Changes the order of image dimensions."""
    return x.transpose([source.index(dim) for dim in target])


# --- Data Augmentation Transforms ---

class Crop(NamedTuple):
    """Crops a fixed-size region, tracking original shape."""
    h: int
    w: int
    _x_shape: Tuple = None  # Stores input shape on first call

    def __call__(self, x: np.ndarray, x0: int, y0: int) -> np.ndarray:
        """Performs the cropping operation."""
        if self._x_shape is None:  # Delayed shape initialization
            self._x_shape = x.shape
        return x[:, y0:y0 + self.h,
               x0:x0 + self.w]

    def undo(self, x: np.ndarray, x0: int,
               y0: int) -> np.ndarray:
        """Reconstructs the original image size by padding."""
        C, H, W = self._x_shape
        padded = np.pad(x,
                        ((0, 0), (y0, H - (y0 + self.h)),
                         (x0, W - (x0 + self.w))),
                        mode='constant', constant_values=0)
        return padded

    def options(self, x_shape: Tuple) -> Dict[str, range]:
        """Calculates all valid offset ranges."""
        C, H, W = x_shape
        x_range = range(W + 1 - self.w)
        y_range = range(H + 1 - self.h)
        return {'x0': x_range, 'y0': y_range}

    def output_shape(self, x_shape: Tuple) -> Tuple:
        """Determines the output dimensions after cropping."""
        C, _, _ = x_shape
        return (C, self.h, self.w)


class FlipLR(NamedTuple):
    """Flips an image horizontally based on a boolean flag."""

    def __call__(self, x: np.ndarray, choice: bool) -> np.ndarray:
        """Applies (or not) the horizontal flip."""
        return x[:, :, ::-1].copy() if choice else x

    def undo(self, x: np.ndarray, choice: bool) -> np.ndarray:
        """Reverses the flip, mirroring __call__()."""
        return self(x, choice)  # Identical operation

    def options(self, x_shape: Tuple) -> Dict[str, List[bool]]:
        """Returns the possible flip choices (True/False)."""
        return {'choice': [True, False]}


class Cutout(NamedTuple):
    """Masks a rectangular region to zero (Cutout)."""
    h: int
    w: int

    def __call__(self, x: np.ndarray, x0: int, y0: int) -> np.ndarray:
        """Applies the masking."""
        x_copy = x.copy()  # Work on a copy to avoid modifying original
        x_copy[:, y0:y0 + self.h,
        x0:x0 + self.w] = 0.0
        return x_copy

    def options(self, x_shape: Tuple) -> Dict[str, range]:
        """Computes valid starting points for the mask."""
        C, H, W = x_shape
        x_range = range(W + 1 - self.w)
        y_range = range(H + 1 - self.h)
        return {'x0': x_range, 'y0': y_range}


class Transform:
    """
    Sequentially applies a list of image transformations with random params.

    Args:
        dataset:  The underlying dataset to transform.
        transforms:  List of transformation objects.
    """

    def __init__(self, dataset, transforms: List[Callable]):
        self.dataset = dataset
        self.transforms = transforms
        self.choices: List[Dict] | None = None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Any]:
        data, labels = self.dataset[index]
        if self.choices is not None:
            for choices, f in zip(self.choices, self.transforms):
                args = {k: v[index] for (k, v) in choices.items()}
                data = f(data, **args)
        return data, labels


    def set_random_choices(self):
        """Generates random parameters for each transform, per image."""
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)

        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append(
                {k: np.random.choice(v, size=N) for (k, v) in options.items()})


# --- Dictionary Operations ---

def union(*dicts: Dict) -> Dict:
    """Combines multiple dictionaries, later keys overwrite earlier ones."""
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged


def path_iter(nested_dict: Dict, pfx: Tuple = ()) -> Tuple[Tuple, Any]:
    """Iterates through all paths in a nested dictionary; yields (path, value)."""
    for name, val in nested_dict.items():
        if isinstance(val, dict):
            yield from path_iter(val, (*pfx, name))
        else:
            yield ((*pfx, name), val)


# --- Graph Construction ---

class RelativePath(NamedTuple):
    """Represents a relative path to a node within the network graph."""
    parts: Tuple[str, ...]


rel_path = lambda *parts: RelativePath(parts)


def build_graph(net: Dict) -> Dict[str, Tuple[Any, List[str]]]:
    """
    Transforms a nested network definition into a flattened graph structure.

    The input `net` is a nested dictionary describing the network's layers
    and their connections.  The function converts this into a flat dictionary
    where keys are unique node identifiers (strings) and values are tuples:
    (node_value, list_of_input_node_ids).

    Args:
        net: Nested dictionary defining the network architecture.

    Returns:
        A dictionary representing the computation graph.
    """
    net = dict(path_iter(net))  # Flatten the nested dictionary
    default_inputs = [[('input',)]] + [[k] for k in net.keys()]

    def _with_default_inputs(values):
        return (val if isinstance(val, tuple) else (val, default_inputs[idx])
                for idx, val in enumerate(values))

    def _parts(path, pfx):
        if isinstance(path, RelativePath):
            return tuple(pfx) + path.parts
        elif isinstance(path, str):
            return (path,)
        else:
            return path
    
    graph = {}
    for (*pfx, name), (val, inputs) in zip(net.keys(),
                                               _with_default_inputs(net.values())):
        node_id = sep.join((*pfx, name))
        input_ids = [sep.join(_parts(x, pfx)) for x in inputs]
        graph[node_id] = (val, input_ids)
    return graph


# --- Training Loop Utilities ---
@singledispatch
def cat(*xs):
    """Generic function to concatenate arrays or tensors."""
    raise NotImplementedError(f"cat not implemented for type {type(xs[0])}")

@cat.register(torch.Tensor)
def _(*xs: torch.Tensor):
    return torch.cat(xs)

@cat.register(np.ndarray)
def _(*xs: np.ndarray):
    return np.concatenate(xs)

@singledispatch
def to_numpy(x):
    """Generic function to convert a tensor to a NumPy array."""
    raise NotImplementedError(f"to_numpy not implemented for type: {type(x)}")

@to_numpy.register(torch.Tensor)
def _(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
class PiecewiseLinear(NamedTuple):
    """Defines a piecewise linear schedule for parameters like learning rate."""
    knots: List[float]  # Independent variable values (e.g., epochs)
    vals: List[float]  # Corresponding dependent variable values

    def __call__(self, t: float) -> float:
        """Interpolates the value at a given point 't'."""
        return np.interp([t], self.knots, self.vals)[0]


class StatsLogger:
    """Accumulates and averages specified training metrics."""

    def __init__(self, keys: List[str]):
        self._stats: Dict[str, List[torch.Tensor]] = {
            k: [] for k in keys}

    def append(self, output: Dict[str, torch.Tensor]):
        """Records metrics from a single batch."""
        for k, v in self._stats.items():
            v.append(output[k])

    def stats(self, key: str) -> torch.Tensor:
        """Retrieves all recorded values for a metric."""
        return cat(*self._stats[key])

    def mean(self, key: str) -> float:
        """Calculates the mean of a metric."""
        return float(np.mean(to_numpy(self.stats(key))))


def run_batches(model: Callable, batches: List[Tuple[torch.Tensor, torch.Tensor]],
                training: bool,
                optimizer_step: Callable | None = None,
                stats: StatsLogger | None = None) -> StatsLogger:
    """Processes a series of batches through the model."""
    stats = stats or StatsLogger(('loss', 'correct'))
    model.train(training)

    for batch in batches:
        output = model(batch)
        stats.append(output)

        if training:
            if not isinstance(output['loss'], torch.Tensor):
                raise TypeError("Model 'loss' output must be a torch.Tensor")
            output['loss'].sum().backward()
            if optimizer_step is None:
                raise ValueError("Optimizer function required during training.")
            optimizer_step()
            model.zero_grad()

    return stats


def train_epoch(model: Callable,
                train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
                test_batches: List[Tuple[torch.Tensor, torch.Tensor]],
                optimizer_step: Callable, timer: Timer,
                test_time_in_total: bool = True) -> Dict[str, float]:
    """Executes one complete training epoch (train + validation)."""
    timer.start()
    train_stats = run_batches(model, train_batches, True, optimizer_step)
    train_time = timer(True)

    timer.start()
    test_stats = run_batches(model, test_batches, False)
    test_time = timer(test_time_in_total)

    return {
        'train time': train_time, 'train loss': train_stats.mean('loss'),
        'train acc': train_stats.mean('correct'),
        'test time': test_time, 'test loss': test_stats.mean('loss'),
        'test acc': test_stats.mean('correct'),
        'total time': timer.total_time,
    }


def train(model: Callable, optimizer,
          train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
          test_batches: List[Tuple[torch.Tensor, torch.Tensor]],
          epochs: int,
          loggers: Tuple[TableLogger, ...] = (),
          test_time_in_total: bool = True,
          timer: Timer | None = None) -> Dict[str, float]:
    """Runs the full training loop for a specified number of epochs."""
    timer = timer or Timer()

    for epoch in range(epochs):
        epoch_stats = train_epoch(model, train_batches, test_batches,
                                  optimizer.step, timer,
                                  test_time_in_total=test_time_in_total)
        summary = union(
            {'epoch': epoch + 1,
             'lr': optimizer.param_values()['lr'] * train_batches.batch_size},
            epoch_stats)
        for logger in loggers:
            logger.append(summary)

    return summary