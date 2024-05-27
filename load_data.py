import os
import numpy as np
import torch
import torchvision.transforms as transforms
#from torchvision.datasets import MNIST
#from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch, Dataset as PyGDataset
from skorch.dataset import Dataset as SkorchDatasetBasic
from skorch import NeuralNetClassifier
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
        self.full, self.y_list = self.download_dataset()

        (self.train,
         self.val,
         self.pool,
         self.test) = self.split_and_load_dataset()

        self.X_init = self.preprocess_training_data()

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
        dataset_pg_list_full = []
        y_list = []
        for graph in dataset_hf['full']:

            node_features = torch.tensor(graph['node_feat'], dtype=torch.float)
            edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
            edge_attributes = torch.tensor(graph['edge_attr'], dtype=torch.float)
            y=torch.tensor(graph['y'], dtype=torch.long)
            num_nodes = graph['num_nodes']

            data_object = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attributes, y=y, num_nodes=num_nodes)
            dataset_pg_list_full.append(data_object)
            y_list.append(y.item())
        return dataset_pg_list_full, y_list



    def split_and_load_dataset(self):
        """Split all training datatset into train, validate, pool sets and load them accordingly."""
        train_set, val_set, pool_set, test_set = random_split(
            self.full, [self.train_size, self.val_size, self.pool_size, self.test_size]
        )
        train_y, val_y, pool_y, test_y = random_split(
            self.y_list, [self.train_size, self.val_size, self.pool_size, self.test_size]
        )
        # train_loader = SkorchDataLoader(dataset=train_set, batch_size=self.train_size, shuffle=True)
        # val_loader = SkorchDataLoader(dataset=val_set, batch_size=self.val_size, shuffle=True)
        # pool_loader = SkorchDataLoader(dataset=pool_set, batch_size=self.pool_size, shuffle=True)
        # test_loader = SkorchDataLoader(dataset=test_set, batch_size=self.test_size, shuffle=True)
        # train_dataset=SkorchDataset(train_set, train_y)
        # val_dataset = SkorchDataset(val_set, val_y)
        # pool_dataset=SkorchDataset(pool_set, pool_y)
        # test_dataset = SkorchDataset(pool_set, test_y)

        return CustomGraphDataset(train_set), CustomGraphDataset(val_set), CustomGraphDataset(pool_set), CustomGraphDataset(test_set)

    def extract_data(self, dataloader):
        #ezt nemtom megtehetem e memoria meg torchgeometric miatt
        #sztem megse tehetem meg mert akk elvesz edge_index, edge_attr és number_of_nodes
        for data in dataloader:
            X = torch.cat([d.x for d in data], dim=0)
            y = torch.cat([d.y for d in data], dim=0)
            return X, y

    def preprocess_training_data(self):
        """Setup a random but balanced initial training set of 20 data points"""
        X_init = []

        count_0, count_1 = 0, 0  # Adjust based on the number of classes you have
        num_samples = 10

        for graph in self.train:
            label = graph.y.item()
            if label == 0 and count_0 < num_samples:
                X_init.append(graph)
                count_0 += 1
            elif label == 1 and count_1 < num_samples:
                X_init.append(graph)
                count_1 += 1
            if count_0 >= num_samples and count_1 >= num_samples:
                break


        return X_init

    def load_all(self):
        datasets={
                "X_init": self.X_init,
                "train_dataset": self.train,
                "val_dataset": self.val,
                "pool_dataset": self.pool,
                "test_dataset": self.test
            }

        return datasets
class CustomGraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        graph = self.data_list[idx]
        # Return data in a format that your model expects; potentially you need to handle batching differently
        return graph


class SkorchDataLoader(torch.utils.data.DataLoader):
    def _collate_fn(self, data_list, follow_batch=[]):
        data = Batch.from_data_list(data_list, follow_batch)
        edge_attr = torch.ones_like(data.edge_index[0], dtype=torch.float) if data.edge_attr is None else data.edge_attr

        # Can't pass a Dataset directly, since it expects tensors.
        # Use dict of tensors instead. Also, use torch.sparse for
        # adjacency matrix to pass skorch's same-dimension check
        return {
            'x': data.x,
            'adj': torch.sparse.FloatTensor(data.edge_index,
                                            edge_attr,
                                            size=[data.num_nodes, data.num_nodes],
                                            device=data.x.device),
            'batch': data.batch
        }, data.y

    def __iter__(self):
        for batch in super().__iter__():
            # Ensure the batch can be indexed like a tuple (input, target)
            yield batch.to_data_list(), batch.y

    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=True,
                 follow_batch=[],
                 **kwargs):
        super(SkorchDataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: self._collate_fn(data_list, follow_batch),
            **kwargs)


class SkorchDataset(SkorchDatasetBasic):
    def __init__(self, X, y):
        # We need to specify `length` to avoid checks
        super(SkorchDataset, self).__init__(X, y, length=len(X))

    def transform(self, X, y):
        return X  # Ignore y, since it is included in X

#
# class SkorchDataLoader(torch.utils.data.DataLoader):
#     def _collate_fn(self, data_list, follow_batch=[]):
#         data = Batch.from_data_list(data_list, follow_batch)
#         if data.edge_attr is None:
#             edge_attr = torch.ones_like(data.edge_index[0], dtype=torch.float)
#         else:
#             edge_attr = data.edge_attr
#
#         print("data edge index: ", data.edge_index, "\n size: ", data.edge_index.size() ,"\nedge attr: ", edge_attr, "\n size: ", edge_attr.size(), "\nnum_nodes: ", data.num_nodes)
#         return {
#             'x': data.x,
#             'adj': torch.sparse_coo_tensor(data.edge_index, edge_attr, size=[data.num_nodes, data.num_nodes], dtype=torch.float, device=data.x.device),
#             'batch': data.batch
#         }, data.y
#
#     def __init__(self, dataset, batch_size=1, shuffle=True, follow_batch=[], **kwargs):
#         super(SkorchDataLoader, self).__init__(
#             dataset, batch_size, shuffle,
#             collate_fn=lambda data_list: self._collate_fn(data_list, follow_batch),
#             **kwargs)
#
# class SkorchDataset(SkorchDatasetBasic):
#     def __init__(self, X, y):
#         super().__init__(X, y, length=len(X))
#
#     def transform(self, X, y):
#         return X
#
#

