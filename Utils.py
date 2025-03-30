import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Callable, Any
from advertorch.attacks import L2PGDAttack
from autoattack import AutoAttack



# --- Utility Functions ---

def zero_gradients(i: Any) -> None: # Assuming i has parameters and named_parameters as in nn.Module
    """Zeros out the gradients of all parameters in a model or optimizer ."""
    if hasattr(i, 'parameters'): # For models
      for param in i.parameters():
        if param.grad is not None:
            param.grad.zero_()
    elif hasattr(i, '_opt'): # For TorchOptimizer class
        i._opt.zero_grad()
    else: # for other iterables
        for param in i:
          if param.grad is not None:
            param.grad.zero_()

class Config:
    """A simple configuration class to store hyperparameters."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def calc_l2distsq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculates the squared L2 distance between two tensors."""
    diff = x - y
    return diff.view(diff.size(0), -1).pow(2).sum(dim=1)


def margin_loss(y_pred: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """Calculates the multi-margin loss."""
    return F.multi_margin_loss(y_pred, y, margin=eps, p=1)


def certified_accuracy(test_loader: Any, model: nn.Module,
                       lip_cst: float = 1.0,
                       eps: float = 36.0 / 255.0) -> Tuple[float, ...]:
    """
    Calculates certified accuracy based on margin bounds.

    Args:
        test_loader: Data loader for the test set.
        model: The PyTorch model.
        lip_cst: Lipschitz constant.
        eps: Perturbation budget (epsilon).

    Returns:
        Tuple: (accuracy, cert_right, cert_wrong, insc_right, insc_wrong)
    """
    model.eval()  # Ensure model is in evaluation mode
    cert_right = 0.0
    cert_wrong = 0.0
    insc_right = 0.0
    insc_wrong = 0.0
    acc = 0.0
    n = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch in test_loader:
            x, y = batch['input'], batch['target']
            yhat = model(x)

            correct = (yhat.max(dim=1)[1] == y)
            margins = torch.sort(yhat, dim=1)[0]
            certified = (margins[:, -1] - margins[:, -2]) > (
                        np.sqrt(2.0) * lip_cst * eps)

            n += y.size(0)
            cert_right += torch.sum(correct & certified).item()
            cert_wrong += torch.sum(~correct & certified).item()
            insc_right += torch.sum(correct & ~certified).item()
            insc_wrong += torch.sum(~correct & ~certified).item()
            acc += torch.sum(correct).item()

    # Calculate percentages
    cert_right /= n
    cert_wrong /= n
    insc_right /= n
    insc_wrong /= n
    acc /= n

    return acc, cert_right, cert_wrong, insc_right, insc_wrong


def certified_accuracy_LLN(test_loader: Any, model: nn.Module,
                           lip_cst: float = 3.0,
                           eps: float = 36.0 / 255.0) -> Tuple[float, ...]:
    """
    Calculates certified accuracy using the LLN-based method.

    Args:
        test_loader:  Data loader for test set.
        model: The PyTorch model.
        lip_cst: Lipschitz constant.
        eps: Perturbation budget.

    Returns:
        Tuple: (accuracy, cert_right, cert_wrong, insc_right, insc_wrong)
    """
    model.eval()
    cert_right = 0.0
    cert_wrong = 0.0
    insc_right = 0.0
    insc_wrong = 0.0
    acc = 0.0
    n = 0

    # Assuming the last layer is named 'last_last' and is a linear layer.
    # This part might need adjustment depending on your model structure.
    normalized_weight = F.normalize(model.module.model.last_last.weight, p=2, dim=1)


    with torch.no_grad():
        for batch in test_loader:
            x, y = batch['input'], batch['target']
            yhat = model(x)

            correct = (yhat.max(dim=1)[1] == y)
            margins, indices = torch.sort(yhat, dim=1)
            margins = (margins[:, -1][:, None] - margins[:, :-1])

            # Calculate normalized margin differences
            for idx in range(margins.size(0)):
                diff_weights = normalized_weight[indices[idx, -1]] - \
                               normalized_weight[indices[idx, :-1]]
                margins[idx] /= torch.norm(diff_weights, dim=1, p=2)

            margins = torch.sort(margins, dim=1)[0]
            certified = (margins[:, 0] > eps * lip_cst)

            n += y.size(0)
            cert_right += torch.sum(correct & certified).item()
            cert_wrong += torch.sum(~correct & certified).item()
            insc_right += torch.sum(correct & ~certified).item()
            insc_wrong += torch.sum(~correct & ~certified).item()
            acc += torch.sum(correct).item()
    # Calculate percentages
    cert_right /= n
    cert_wrong /= n
    insc_right /= n
    insc_wrong /= n
    acc /= n
    return acc, cert_right, cert_wrong, insc_right, insc_wrong


def test_pgd_l2(model: nn.Module, test_batches: Any,
                loss_fn: Callable = nn.CrossEntropyLoss(reduction="sum"),
                eps: float = 36.0 / 255.0, nb_iter: int = 10,
                eps_iter: float = 0.2, rand_init: bool = True,
                clip_min: float = 0.0, clip_max: float = 1.0,
                targeted: bool = False) -> float:
    """
    Evaluates model robustness using PGD with L2 norm.

    Args:
        model: The PyTorch model.
        test_batches: Data loader for test set.
        loss_fn: Loss function.
        eps: Perturbation budget.
        nb_iter: Number of PGD iterations.
        eps_iter: Step size for each iteration.
        rand_init: Whether to use random initialization.
        clip_min: Minimum value for clipping.
        clip_max: Maximum value for clipping.
        targeted: Whether to perform targeted attacks.

    Returns:
        float: Adversarial accuracy.
    """
    adversary = L2PGDAttack(model, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
                            eps_iter=eps_iter, rand_init=rand_init,
                            clip_min=clip_min, clip_max=clip_max,
                            targeted=targeted)
    model.eval()
    correct = 0
    total = 0

    for batch in test_batches:
        images, target = batch['input'], batch['target']
        images_adv = adversary.perturb(images, target)
        with torch.no_grad():  # Disable gradient during evaluation
          predictions_adv = model(images_adv)

        predictions_adv = predictions_adv.argmax(dim=1)  # No need for keepdim
        correct += (predictions_adv == target).sum().item()
        total += target.size(0)

    accuracy = 100.0 * correct / total
    print(f'\nTest set adversarial accuracy: {correct}/{total} ({accuracy:.0f}%)\n')
    return accuracy


def test_auto_attack(model: nn.Module, test_batches: Any,
                     eps: float = 36.0 / 255.0) -> float:
    """
    Evaluates model robustness using AutoAttack.

    Args:
        model: The PyTorch model.
        test_batches: Data loader for the test set.
        eps: Perturbation budget.

    Returns:
        float: Adversarial accuracy.
    """
    adversary = AutoAttack(model, norm='L2', eps=eps, version='standard')
    model.eval()
    correct = 0
    total = 0

    for batch in test_batches:
        images, targets = batch['input'], batch['target']
        # AutoAttack expects batches, so no need to iterate within the batch
        images_adv = adversary.run_standard_evaluation(images, targets, bs=images.size(0))

        with torch.no_grad():
            predictions_adv = model(images_adv)
        predictions_adv = predictions_adv.argmax(dim=1)
        correct += (predictions_adv == targets).sum().item()
        total += targets.size(0)

    accuracy = 100.0 * correct / total
    print(f'\nTest set adversarial accuracy: {correct}/{total} ({accuracy:.0f}%)\n')
    return accuracy


class TriangularLRScheduler:
    """
    Implements a triangular learning rate schedule.

    Args:
        optimizer: The optimizer.
        lr_steps: Total number of training steps (epochs).
        lr: Maximum learning rate.
    """

    def __init__(self, optimizer: Any, lr_steps: int, lr: float):
        self.optimizer = optimizer
        self.epochs = lr_steps
        self.lr = lr

    def step(self, t: int) -> None:
        """
        Updates the learning rate based on the current step.

        Args:
            t: Current training step.
        """
        lr = np.interp([t],
                       [0, self.epochs * 2 // 5, self.epochs * 4 // 5,
                        self.epochs],
                       [0, self.lr, self.lr / 20.0, 0])[0]
        # Assuming the optimizer has a 'param_groups' attribute.  This is
        # standard for PyTorch optimizers.
        self.optimizer.param_groups[0]['lr'] = lr  # Directly set 'lr'


class SpectralNormPowerMethod(nn.Module):
    """
    Computes the spectral norm of a matrix using the power method.

    Args:
       input_size: Size of the input.
       eps: Small value for numerical stability.
    """

    def __init__(self, input_size: int, eps: float = 1e-8):
        super().__init__()
        self.input_size = input_size
        self.eps = eps
        self.u = torch.randn(input_size)
        self.u = self.u / self.u.norm(p=2)
        self.u = nn.Parameter(self.u, requires_grad=False)  # u is not trainable

    def normalize(self, arr: torch.Tensor) -> torch.Tensor:
        """Normalizes a tensor to have unit L2 norm."""
        norm = torch.sqrt((arr ** 2).sum())
        return arr / (norm + 1e-12)  # Add small epsilon for stability

    def _compute_dense(self, M: torch.Tensor, max_iter: int) -> torch.Tensor:
        """Computes spectral norm for a dense matrix."""
        u = self.u
        for _ in range(max_iter):
            v = self.normalize(F.linear(u, M))
            u = self.normalize(F.linear(v, M.T))
        self.u.data = u # update u
        z = F.linear(self.u, M)
        sigma = torch.mul(z, v).sum()
        return sigma

    def _compute_conv(self, kernel: torch.Tensor,
                      max_iter: int) -> torch.Tensor:
        """Computes spectral norm for a convolutional kernel."""
        pad = (1, 1, 1, 1)
        pad_ = (-1, -1, -1, -1)  # Corrected padding for conv_transpose2d
        u = self.u
        for _ in range(max_iter):
            v = self.normalize(F.conv2d(F.pad(u, pad), kernel))
            u = self.normalize(F.pad(F.conv_transpose2d(v, kernel), pad_))
        self.u.data = u # update u
        z = F.conv2d(F.pad(self.u, pad), kernel)
        sigma = torch.mul(z, v).sum()
        return sigma

    def forward(self, M: torch.Tensor, max_iter: int) -> torch.Tensor:
        """
        Computes the spectral norm.

        Args:
          M: The matrix (or convolutional kernel).
          max_iter: Maximum number of iterations.

        Returns:
          float: The spectral norm.
        """
        if len(M.shape) == 4:
            return self._compute_conv(M, max_iter)
        elif len(M.shape) == 2:
            return self._compute_dense(M, max_iter)
        else:
            raise ValueError("Input tensor must be 2D (dense) or 4D (conv).")