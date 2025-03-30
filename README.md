# Contractive_Lipschitz_Layers

# Towards Robust Deep Learning: Lipschitz Continuity and Contractive Layers

This repository contains the official PyTorch implementation for the Master of Science (Research) thesis titled "Towards Robust Deep Learning: Lipschitz Continuity and Contractive Layers" conducted at the Indian Institute of Science (IISc), Bengaluru.

**Author:** Jayesh Kumar Jaiswal
## Overview

Deep neural networks are known to be vulnerable to adversarial attacks – small, often imperceptible perturbations to the input that can cause drastic changes in the output. This thesis investigates methods for improving the adversarial robustness of deep learning models, focusing on the principles of Lipschitz continuity.

We explore:
1.  **Lipschitz Neural Networks:** Techniques like spectral normalization (estimated via Power Iteration) to constrain the Lipschitz constant of network layers.
2.  **Convex Potential Layer Networks (CPL-Nets):** A novel architecture incorporating convex potential layers derived from gradient methods to enhance robustness while maintaining competitive accuracy.
3.  **Contractive Layers:** An extension to 1-Lipschitz layers, theoretically analyzed and empirically evaluated for further improvements in certified robustness and standard accuracy.

The code implements CPL-Nets and Contractive Layer variants, trains them on CIFAR-10, and evaluates their standard accuracy, certified robustness (based on Lipschitz constants), and empirical robustness against strong adversarial attacks like Projected Gradient Descent (PGD) and AutoAttack.

## Key Contributions & Features Implemented

*   Implementation of **Convex Potential Layers (CPL)** for both Convolutional and Linear blocks in PyTorch.
*   Integration of **Spectral Normalization** using the Power Iteration method for dynamic Lipschitz constant estimation during training.
*   Implementation of the proposed **Contractive Layers**.
*   Training pipelines for CIFAR-10 using Margin Loss.
*   Evaluation scripts for:
    *   Standard Accuracy
    *   Certified Robustness (using the Lipschitz constant derived from spectral norms)
    *   Empirical Robustness against PGD (L2) attacks
    *   Empirical Robustness against AutoAttack (L2)
*   Comparison framework against a standard ResNet18 baseline.
*   Options for using Last Layer Normalization (LLN) and different model configurations (CPL-S, CPL-M).

## Requirements

*   Python 3.x
*   PyTorch (>= 1.8 recommended, check version compatibility)
*   Torchvision
*   NumPy
*    `autoattack` library (if using the AutoAttack evaluation script directly: `pip install git+https://github.com/fra31/auto-attack`)
Usage
Training

To train a model (e.g., CPL-S without LLN), run the main training script (adjust parameters as needed):

python train.py \
    --model CPLS \ # Or CPLM
    --dataset cifar10 \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.01 \
    --margin 1.0 \
    --use-lln False \ # Set True to use Last Layer Normalization
    --use-contractive False \ # Set True to use Contractive Layers
    --contractive-mu 0.1 \ # Mu value if using contractive layers
    --save-path ./checkpoints/cpls_no_lln

Results Summary

Our experiments on CIFAR-10 demonstrate:

CPL-Nets achieve significantly higher certified and empirical robustness against adversarial attacks compared to standard ResNet architectures, while maintaining competitive standard accuracy. (See Chapter 6 & 7 in the thesis).

Example: CPL-M achieved ~63% certified accuracy and ~72% PGD accuracy at ε=36/255 vs. ~44% PGD accuracy for ResNet18.

Contractive Layers further improve standard accuracy (up to ~86%) and certified accuracy (up to ~83% at ε=36/255) over 1-Lipschitz CPL-Nets, especially at lower perturbation levels. (See Chapter 8 in the thesis).

Please refer to the thesis PDF for detailed results, tables, figures, and analysis.

Citation

If you find this work useful in your research, please consider citing the thesis:

@mastersthesis{jaiswal2024robust,
  author       = {Jaiswal, Jayesh Kumar},
  title        = {Towards Robust Deep Learning: Lipschitz Continuity and Contractive Layers},
  school       = {Indian Institute of Science},
  year         = {2024},
  address      = {Bengaluru, India},
  month        = {April}, % Or the submission month
  note         = {Master of Science (Research) Thesis}
}
