import torch
import torch.nn as nn 
import torch.nn.functional as F
from typing import List
from Layers import ConvexPotentialLayerConv, ConvexPotentialLayerLinear, LinearNormalized, PoolingLinear
from Layers import PaddingChannels


class ConvexPotentialLayerNetwork(nn.Module):
    """
    Deep network employing convex potential layers for robust classification.

    This network architecture uses a combination of convolutional and linear
    layers based on the convex potential theory, aiming for improved
    robustness against adversarial attacks.

    Args:
        depth: Number of convolutional potential layers.
        num_classes: The number of output classes for classification.
        depth_linear: Number of linear potential layers.
        conv_size: Number of output channels in each convolutional layer.
        num_channels: Number of channels after initial input padding.
        n_features: Number of features in the hidden linear layers.
        use_lln:  If True, uses a LinearNormalized layer for the final
                   classification; otherwise, uses a PoolingLinear layer.
    """
    def __init__(self, depth: int, num_classes: int, depth_linear: int = 3,
                 conv_size: int = 20, num_channels: int = 20,
                 n_features: int = 512, use_lln: bool = True) -> None:
        super().__init__()

        # --- Network Parameters ---
        self.num_classes = num_classes
        self.conv_size = conv_size
        self.depth = depth
        self.depth_linear = depth_linear
        self.num_channels = num_channels
        self.use_lln = use_lln
        

        # --- Input Preprocessing ---
        self.conv1 = PaddingChannels(self.num_channels, ncin=3, mode="zero")

        # --- Convolutional Potential Layer Stack ---
        conv_blocks: List[nn.Module] = []
        for _ in range(self.depth):
            conv_blocks.append(
                ConvexPotentialLayerConv((1, self.num_channels, 32, 32),
                                          cin=self.num_channels,
                                          cout=self.conv_size)
            )
        conv_blocks.append(nn.AvgPool2d(4, divisor_override=4))  # Spatial pooling
        self.stable_block = nn.Sequential(*conv_blocks)

        # --- Linear Potential Layer Stack ---
        linear_blocks: List[nn.Module] = [nn.Flatten()]  # Flatten for linear layers
        in_features = self.num_channels * 8 * 8  # Calculate based on conv output
        for _ in range(self.depth_linear):
            linear_blocks.append(
                ConvexPotentialLayerLinear(cin=in_features, cout=n_features)
            )
            in_features = n_features # Update in_features for next layer

        self.layers_linear = nn.Sequential(*linear_blocks)

        # --- Output Layer ---
        if self.use_lln:
            self.last_last = LinearNormalized(in_features=in_features,
                                              out_features=self.num_classes)
        else:
            self.last_last = PoolingLinear(ncin=in_features,
                                           ncout=self.num_classes, agg="trunc")

        # --- Combined Network ---
        self.base = nn.Sequential(self.conv1, self.stable_block,
                                    self.layers_linear)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the entire network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor (logits).
        """
        features = self.base(x)  # Process input through conv and linear stacks
        output = self.last_last(features)   # Final classification layer
        return output