import os
import numpy as np
import torch
import torchvision.transforms as transforms
#from torchvision.datasets import MNIST
#from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from datasets import Dataset
from torch.utils.data import random_split
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader



class LoadData:
    """Download, split and shuffle dataset into train, validate, test and pool"""

    def __init__(self, val_size: int = 10):
        self.train_size = 400
        self.val_size = val_size
        self.test_size = 100
        self.pool_size = 2000 - self.train_size - self.val_size - self.test_size

        #self.mnist_train, self.mnist_test = self.download_dataset()
        self.full=self.download_dataset()
        (
            self.X_train_All,
            self.y_train_All,
            self.X_val,
            self.y_val,
            self.X_pool,
            self.y_pool,
            self.X_test,
            self.y_test,
        ) = self.split_and_load_dataset()
        self.X_init, self.y_init = self.preprocess_training_data()

    def tensor_to_np(self, tensor_data: torch.Tensor) -> np.ndarray:
        """Since Skorch doesn not support dtype of torch.Tensor, we will modify
        the dtype to numpy.ndarray

        Attribute:
            tensor_data: Data of class type=torch.Tensor
        """
        np_data = tensor_data.detach().numpy()
        return np_data

    def check_MNIST_folder(self) -> bool:
        """Check whether MNIST folder exists, skip download if existed"""
        if os.path.exists("MNIST/"):
            return False
        return True

    def download_dataset(self):
        """Load MNIST dataset for training and test set."""
        # transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )

        dataset_hf = load_dataset("graphs-datasets/AIDS")
        dataset_pg_list_full = [Data(graph) for graph in dataset_hf["full"]]
        dataset_pg=DataLoader(dataset_pg_list_full)
       # print(next(iter(dataset_pg)).x['num_nodes'])

        return dataset_pg_list_full

        #download = self.check_MNIST_folder()
        #mnist_train = MNIST(".", train=True, download=download, transform=transform)
        #mnist_test = MNIST(".", train=False, download=download, transform=transform)



    def split_and_load_dataset(self):
        """Split all training datatset into train, validate, pool sets and load them accordingly."""
        train_set, val_set, pool_set, test_set = random_split(
            self.full, [self.train_size, self.val_size, self.pool_size, self.test_size]
        )
        train_loader = DataLoader(
            dataset=train_set, batch_size=self.train_size, shuffle=True
        )
        val_loader = DataLoader(dataset=val_set, batch_size=self.val_size, shuffle=True)
        pool_loader = DataLoader(
            dataset=pool_set, batch_size=self.pool_size, shuffle=True
        )
        test_loader = DataLoader(
            dataset=test_set, batch_size=self.test_size, shuffle=True
        )
        train_batch = next(iter(train_loader))
        X_train_All=train_batch.x
        y_train_All=X_train_All['y']
        del X_train_All['y']

        # Validation set
        val_batch = next(iter(val_loader))
        X_val = val_batch.x
        y_val = X_val['y']
        del X_val['y']


        # Pool set
        pool_batch = next(iter(pool_loader))
        X_pool = pool_batch.x
        y_pool = X_pool['y']
        del X_pool['y']  # Remove 'y' from x

        # Test set
        test_batch = next(iter(test_loader))
        X_test = test_batch.x
        y_test = X_test['y']
        del X_test['y']  # Remove 'y' from x
        return X_train_All, y_train_All, X_val, y_val, X_pool, y_pool, X_test, y_test

    def preprocess_training_data(self):
        """Setup a random but balanced initial training set of 20 data points

        Attributes:
            X_train_All: X input of training set,
            y_train_All: y input of training set
        """
        initial_idx = np.array([], dtype=int)
        self.y_train_All=torch.tensor([label for sublist in self.y_train_All for label in sublist])
        for i in range(2):
            idx = np.random.choice(
                np.where(self.y_train_All == i)[0], size=10, replace=False
            )
            initial_idx = np.concatenate((initial_idx, idx))
        print(initial_idx)
        X_init = [{key: value[initial_idx] for key, value in self.X_train_All.items()}]
        print(X_init)
        y_init = self.y_train_All[initial_idx]
        print(f"Initial training data points: {X_init.shape[0]}")
        print(f"Data distribution for each class: {np.bincount(y_init)}")
        return X_init, y_init

    def load_all(self):
        """Load all data"""
        return (
            self.tensor_to_np(self.X_init),
            self.tensor_to_np(self.y_init),
            self.tensor_to_np(self.X_val),
            self.tensor_to_np(self.y_val),
            self.tensor_to_np(self.X_pool),
            self.tensor_to_np(self.y_pool),
            self.tensor_to_np(self.X_test),
            self.tensor_to_np(self.y_test),
        )
