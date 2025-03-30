import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from data import DataClass
from model import ConvexPotentialLayerNetwork, NormalizedModel
from layers import margin_loss, certified_accuracy, certified_accuracy_LLN
from attacks import test_auto_attack, test_pgd_l2
from lr_scheduler import TriangularLRScheduler

class Trainer:
    """
    Handles training and evaluation of a ConvexPotentialLayerNetwork.

    Args:
        config:  A configuration object containing hyperparameters.  Should
            have attributes for: seed, lr, epochs, batch_size, weight_decay,
            depth, depth_linear, save_dir, conv_size, num_channels, n_features,
            margin, lln, dataset, norm_input.
    """

    def __init__(self, config) -> None:

        self.cuda = True  # Assume CUDA is available
        self.seed = config.seed
        self.lr = config.lr
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.wd = config.weight_decay
        self.depth = config.depth
        self.depth_linear = config.depth_linear
        self.save_dir = config.save_dir
        self.conv_size = config.conv_size
        self.num_channels = config.num_channels
        self.n_features = config.n_features
        self.margin = config.margin
        self.lln = config.lln
        self.dataset = config.dataset
        self.norm_input = config.norm_input

        # Initialize lists to store metrics
        self.train_losses: List[float] = []
        self.accuracies: List[float] = []
        self.certified_accuracies: List[float] = []
        self.time_epoch: float = 0.0


    def _setup_data_and_model(self) -> None:
        """Sets up data loaders, model, optimizer, and LR scheduler."""

        torch.manual_seed(self.seed) # Set seed

        # --- Data Loading ---
        if self.dataset == "c10":
            self.mean = (0.4913997551666284, 0.48215855929893703, 0.4465309133731618)
            self.std = (0.24703225141799082, 0.24348516474564, 0.26158783926049628)
            num_classes = 10
        elif self.dataset == "c100":
            self.mean = (0.5071, 0.4865, 0.4409)
            self.std = (0.2673, 0.2564, 0.2762)
            num_classes = 100
        else:
            raise ValueError("Invalid dataset.  Must be 'c10' or 'c100'.")

        if not self.norm_input:
            self.std = (1.0, 1.0, 1.0)

        self.data = DataClass(self.dataset, batch_size=self.batch_size)
        self.train_batches, self.test_batches = self.data()

        # --- Model Initialization ---
        self.model = ConvexPotentialLayerNetwork(
            depth=self.depth, depth_linear=self.depth_linear,
            num_classes=num_classes, conv_size=self.conv_size,
            num_channels=self.num_channels, n_features=self.n_features,
            use_lln=self.lln
        )
        self.model = NormalizedModel(self.model, self.mean, self.std)
        self.model = nn.DataParallel(self.model) # Use DataParallel
        self.model = self.model.cuda()
        print(self.model)

        # --- Optimizer and LR Scheduler ---
        self.optimizer = optim.Adam(self.model.parameters(),
                                    weight_decay=self.wd, lr=self.lr)
        lr_steps = self.epochs * len(self.train_batches)
        self.lr_scheduler = TriangularLRScheduler(self.optimizer, lr_steps,
                                                 self.lr)

        # --- Loss Function ---
        self.criterion = lambda yhat, y: margin_loss(yhat, y, self.margin)
        print(f"Margin loss with param {self.margin}, lr = {self.lr}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")


    def __call__(self) -> None:
        """Runs the complete training and evaluation loop."""

        self._setup_data_and_model()  # Initialize everything
        best_acc = 0.0

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            self._train_epoch(epoch)  # Train for one epoch
            print(f'Epoch {epoch}: Training complete. Evaluating...')

            self.time_epoch = time.time() - start_time
            print(f"Epoch time: {self.time_epoch:.2f} seconds")

            acc, cert_acc = self._evaluate(epoch)  # Evaluate
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(),
                           f"{self.save_dir}/best_model.pt") # Save best

            self.accuracies.append(acc)
            self.certified_accuracies.append(cert_acc)

        # Save the final model
        torch.save(self.model.state_dict(), f"{self.save_dir}/last_model.pt")

        self._plot_results() # create the plots



    def _train_epoch(self, epoch: int) -> None:
        """Trains the model for a single epoch."""

        self.model.train()  # Set model to training mode
        total_loss = 0.0

        for batch_idx, batch in enumerate(self.train_batches):
            # Advance learning rate schedule
            step = (epoch -1) * len(self.train_batches) + batch_idx + 1
            self.lr_scheduler.step(step)

            images, targets = batch['input'], batch['target']
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs.cpu(), targets.cpu()).cuda()

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Print training progress
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} '
                      f'[{batch_idx * len(images)}/{len(self.train_batches.dataset)} '
                      f'({100. * batch_idx / len(self.train_batches):.0f}%)]'
                      f'\tLoss: {loss.item():.6f} ')

        avg_loss = total_loss / len(self.train_batches)
        self.train_losses.append(avg_loss)


    def _evaluate(self, epoch: int) -> Tuple[float, float]:
        """Evaluates the model on the test set."""

        # Choose certified accuracy function based on 'use_lln'
        cert_acc_fn = certified_accuracy_LLN if self.lln else certified_accuracy
        lip_cst = 1.0 / np.min(self.std)  # Lipschitz constant
        acc, cert_acc, _, _, _ = cert_acc_fn(
            self.test_batches, self.model, lip_cst=lip_cst, eps=36.0 / 255.0
        )
        lr = self.optimizer.param_groups[0]['lr'] # Get current learning rate
        print(f"Epoch {epoch}: Accuracy: {acc:.4f}, "
              f"Certified Accuracy: {cert_acc:.4f}, lr: {lr:.4f}\n")
        return acc, cert_acc

    def _plot_results(self) -> None:
        """
        Plot of training loss, accuracy and certified accuracy.
        """
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epochs + 1), self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.savefig(f"{self.save_dir}/training_loss.png")  # Save the plot
        plt.close()

        # Plot accuracy vs epoch
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epochs + 1), self.accuracies, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend()
        plt.savefig(f"{self.save_dir}/accuracy_vs_epoch.png")  # Save the plot
        plt.close()

        # Plot certified accuracy vs epoch
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.epochs + 1), self.certified_accuracies,
                 label='Certified Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Certified Accuracy')
        plt.title('Certified Accuracy Over Epochs')
        plt.legend()
        plt.savefig(f"{self.save_dir}/certified_accuracy_vs_epoch.png")  # Save
        plt.close()


    def eval_final(self, eps: float = 36.0 / 255.0) -> None:
        """
        Performs a final evaluation, including PGD and AutoAttack.

        Args:
            eps: Perturbation budget for adversarial attacks.
        """
        self.model.eval()  # Ensure model is in evaluation mode

        # Certified accuracy
        lip_cst = 1.0 / np.min(self.std)
        cert_acc_fn = certified_accuracy_LLN if self.lln else certified_accuracy
        acc, cert_acc, _, _, _ = cert_acc_fn(
            self.test_batches, self.model, lip_cst=lip_cst, eps=eps
        )

        # AutoAttack
        with torch.no_grad():  # Disable gradient calculation for AutoAttack
            acc_auto = test_auto_attack(self.model, self.test_batches, eps=eps)

        # PGD attack
        eps_iter = 2.0 * eps / 10.0
        if eps == 0:
            acc_pgd = acc  # If eps=0, PGD accuracy is the same as clean accuracy
        else:
            acc_pgd = test_pgd_l2(
                self.model, self.test_batches,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=eps, nb_iter=10, eps_iter=eps_iter, rand_init=True,
                clip_min=0.0, clip_max=1.0, targeted=False
            )

        print(f"Final Evaluation (epsilon={eps}):\n"
              f"  Clean Accuracy    : {acc:.4f}\n"
              f"  Certified Accuracy: {cert_acc:.4f}\n"
              f"  AutoAttack Accuracy: {acc_auto:.4f}\n"
              f"  PGD Attack Accuracy: {acc_pgd:.4f}")