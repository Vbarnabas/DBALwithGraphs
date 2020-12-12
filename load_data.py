import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split

class LoadData:
    """Download, split and shuffle dataset into train, validate, test and pool"""
    def __init__(self):
        self.mnist_train, self.mnist_test = self.download_dataset()
        self.X_train_All, self.y_train_All, self.X_val, self.y_val, \
            self.X_pool, self.y_pool, self.X_test, self.y_test = self.split_and_load_dataset()
        self.X_init, self.y_init = self.preprocess_training_data()

    def check_MNIST_folder(self) -> bool:
        """Check whether MNIST folder exists, if yes remove and redownload."""
        if os.path.isfile("MNIST/"):
            return False
        return True

    def download_dataset(self):
        """Load MNIST dataset for training and test set."""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        )
        download = self.check_MNIST_folder()
        mnist_train = MNIST(".", train=True, download=download, transform=transform)
        mnist_test = MNIST(".", train=False, download=download, transform=transform)
        return mnist_train, mnist_test

    def split_and_load_dataset(
        self,
        train_size: int = 10000,
        val_size: int = 5000,
        pool_size: int = 45000,
    ):
        """Split all training datatset into train, validate, pool sets and load them accordingly.

        Attributes:
            all_training_set: MNIST training dataset,
            test_set: MNIST test dataset
            training_size: Training data size,
            val_size: Validation data size,
            pool_size: Pool set size
        """
        train_set, val_set, pool_set = random_split(
            self.mnist_train, [train_size, val_size, pool_size]
        )
        train_loader = DataLoader(dataset=train_set, batch_size=train_size, shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=val_size, shuffle=True)
        pool_loader = DataLoader(dataset=pool_set, batch_size=pool_size, shuffle=True)
        test_loader = DataLoader(dataset=self.mnist_test, batch_size=10000, shuffle=True)
        X_train_All, y_train_All = next(iter(train_loader))
        X_val, y_val = next(iter(val_loader))
        X_pool, y_pool = next(iter(pool_loader))
        X_test, y_test = next(iter(test_loader))
        return X_train_All, y_train_All, X_val, y_val, X_pool, y_pool, X_test, y_test

    def preprocess_training_data(self):
        """Setup a random but balanced initial training set of 20 data points

        Attributes:
            X_train_All: X input of training set,
            y_train_All: y input of training set
        """
        initial_idx = np.array([], dtype=np.int)
        for i in range(10):
            idx = np.random.choice(np.where(self.y_train_All == i)[0], size=2, replace=False)
            initial_idx = np.concatenate((initial_idx, idx))
        X_init = self.X_train_All[initial_idx]
        y_init = self.y_train_All[initial_idx]
        return X_init, y_init

    def load_all(self):
        """Load all data"""
        return self.X_init, self.y_init, self.X_train_All, self.y_train_All, self.X_val, \
            self.y_val, self.X_pool, self.y_pool, self.X_test, self.y_test 