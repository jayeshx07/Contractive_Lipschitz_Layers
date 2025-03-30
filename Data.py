import sys
from typing import Tuple, List, Any
from dataclasses import dataclass
from typing import Dict 
import numpy as np
from data import cifar10, cifar100


class DataClass:
    """
    Provides CIFAR-10/CIFAR-100 data loaders with preprocessing and augmentation.

    Args:
        dataset (str):  'c10' for CIFAR-10, 'c100' for CIFAR-100.
        batch_size (int):  Desired batch size for training.
    """

    def __init__(self, dataset: str = "c10", batch_size: int = 256):
        if dataset not in ("c10", "c100"):
            raise ValueError("Dataset must be 'c10' or 'c100'")

        self.dataset_name = dataset
        self.batch_size = batch_size
        print(f'Using batch size: {batch_size}')
        self.data_dir = './data'  # Fixed data directory

    def __call__(self) -> Tuple[Any, Any]:  # Use Any since Batches type is unknown
        """
        Loads and preprocesses the specified dataset.

        Returns:
            Tuple: Training batches and test batches.
        """
        dataset_fn = cifar10 if self.dataset_name == "c10" else cifar100
        dataset = dataset_fn(self.data_dir)

        timer = Timer()
        timer.start()  # Use .start()
        print('Preprocessing training data...')
        train_data = self._preprocess(dataset['train'], pad_data=True)
        print(f'Finished in {timer(True):.2f} seconds')  # use timer correctly

        timer.start()# Use .start()
        print('Preprocessing test data...')
        test_data = self._preprocess(dataset['test'], pad_data=False)
        print(f'Finished in {timer(True):.2f} seconds')# use timer correctly

        # Data augmentation for training set
        train_transform = Transform(train_data, [Crop(32, 32), FlipLR()])

        # Create batch iterators
        train_batches = Batches(train_transform, self.batch_size, shuffle=True,
                                set_random_choices=True, num_workers=20)
        test_batches = Batches(test_data, 256, shuffle=False, num_workers=20) # Keep the original batch size.

        return train_batches, test_batches

    def _preprocess(self, data_dict: Dict, pad_data: bool) -> List[
        Tuple[np.ndarray, Any]]:
        """
        Helper function to preprocess raw data.

        Args:
            data_dict: Dictionary containing 'data' and 'labels'.
            pad_data: Whether to apply padding.

        Returns:
            List of (image, label) tuples.
        """
        images = data_dict['data']
        labels = data_dict['labels']
        if pad_data:
            images = pad(images, 4)
        images = transpose(images) / 255.0
        return list(zip(images, labels))