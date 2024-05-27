import os
import time
import argparse
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skorch import NeuralNetClassifier
from scipy.ndimage.filters import gaussian_filter1d
from torch_geometric.datasets import TUDataset

from load_data import LoadData, SkorchDataLoader, SkorchDataset
from cnn_model import ConvNN
from gnn_model import GraphNN
from active_learning import select_acq_function, active_learning_procedure


def load_gnn_model(args, device):
    """Load new model each time for different acqusition function
    each experiments"""
    
    model = GraphNN().to(device)
    gnn_classifier = NeuralNetClassifier(
        module=model,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        iterator_train=SkorchDataLoader,
        iterator_valid=SkorchDataLoader,
        dataset=SkorchDataset,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        train_split=None,
        verbose=0,
        device=device,
    )
    return gnn_classifier


def save_as_npy(data: np.ndarray, folder: str, name: str):
    """Save result as npy file

    Attributes:
        data: np array to be saved as npy file,
        folder: result folder name,
        name: npy filename
    """
    file_name = os.path.join(folder, name + ".npy")
    np.save(file_name, data)
    print(f"Saved: {file_name}")


def plot_results(data: dict):
    """Plot results histogram using matplotlib"""
    sns.set()
    for key in data.keys():
        # data[key] = gaussian_filter1d(data[key], sigma=0.9) # for smoother graph
        plt.plot(data[key], label=key)
    plt.show()


def print_elapsed_time(start_time: float, exp: int, acq_func: str):
    """Print elapsed time for each experiment of acquiring

    Attributes:
        start_time: Starting time (in time.time()),
        exp: Experiment iteration
        acq_func: Name of acquisition function
    """
    elp = time.time() - start_time
    print(
        f"********** Experiment {exp} ({acq_func}): {int(elp//3600)}:{int(elp%3600//60)}:{int(elp%60)} **********"
    )


def train_active_learning(args, device, dataloaders: dict, X_init) -> dict:
    """Start training process

    Attributes:
        args: Argparse input,
        estimator: Loaded model, e.g. CNN classifier,
        device: Cpu or gpu,
        dataloaders: Dataset dict that consists of all datasets,
    """
    acq_functions = select_acq_function(args.acq_func)
    results = dict()
    if args.determ:
        state_loop = [True, False]  # dropout VS non-dropout
    else:
        state_loop = [True]  # run dropout only

    for state in state_loop:
        for i, acq_func in enumerate(acq_functions):
            avg_hist = []
            test_scores = []
            acq_func_name = str(acq_func).split(" ")[1] + "-MC_dropout=" + str(state)
            print(f"\n---------- Start {acq_func_name} training! ----------")
            for e in range(args.experiments):
                start_time = time.time()
                estimator = load_gnn_model(args, device)
                print(
                    f"********** Experiment Iterations: {e+1}/{args.experiments} **********"
                )
                training_hist, test_score = active_learning_procedure(
                    query_strategy=acq_func,
                    dataloaders=dataloaders,
                    X_init=X_init,
                    estimator=estimator,
                    T=args.dropout_iter,
                    n_query=args.query,
                    training=state,
                )
                avg_hist.append(training_hist)
                test_scores.append(test_score)
                print_elapsed_time(start_time, e + 1, acq_func_name)
            avg_hist = np.average(np.array(avg_hist), axis=0)
            avg_test = sum(test_scores) / len(test_scores)
            print(f"Average Test score for {acq_func_name}: {avg_test}")
            results[acq_func_name] = avg_hist
            save_as_npy(data=avg_hist, folder=args.result_dir, name=acq_func_name)
    print("--------------- Done Training! ---------------")
    return results

def train_and_evaluate(args, device):
    #aktiv tanulas nelkul

    model = GraphNN().to(device)
    gnn_classifier = NeuralNetClassifier(
        module=model,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        iterator_train=SkorchDataLoader,
        iterator_valid=SkorchDataLoader,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        dataset=SkorchDataset,
        train_split=None,
        verbose=1,
        device=device,
    )

    dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')

    gnn_classifier.fit(list(dataset), dataset.data.y.numpy())


    val_accuracy = gnn_classifier.score(dataset)
    print(f"Validation Accuracy: {val_accuracy:.4f}")


    test_accuracy = gnn_classifier.score(dataset)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return val_accuracy, test_accuracy
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="EP",
        help="number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--seed", type=int, default=369, metavar="S", help="random seed (default: 369)"
    )
    parser.add_argument(
        "--experiments",
        type=int,
        default=3,
        metavar="E",
        help="number of experiments (default: 3)",
    )
    parser.add_argument(
        "--dropout_iter",
        type=int,
        default=100,
        metavar="T",
        help="dropout iterations,T (default: 100)",
    )
    parser.add_argument(
        "--query",
        type=int,
        default=10,
        metavar="Q",
        help="number of query (default: 10)",
    )
    parser.add_argument(
        "--acq_func",
        type=int,
        default=0,
        metavar="AF",
        help="acqusition functions: 0-all, 1-uniform, 2-max_entropy, \
                            3-bald, 4-var_ratios, 5-mean_std (default: 0)",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=100,
        metavar="V",
        help="validation set size (default: 100)",
    )
    parser.add_argument(
        "--determ",
        action="store_true",
        help="Compare with deterministic models (default: False)",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="result_npy",
        metavar="SD",
        help="Save npy file in this folder (default: result_npy)",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize your data loaders
    # DataLoader = LoadData(args.val_size)
    # dataloaders = {
    #     'train_dataset': DataLoader.train_dataset,
    #     'val_loader': DataLoader.val_loader,
    #     'test_loader': DataLoader.test_loader
    # }

    # datasets=DataLoader.load_all()

    # Run the simplified training and evaluation
    val_accuracy, test_accuracy = train_and_evaluate(args, device)
    print(f"Finished Training! Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

    #
    # args = parser.parse_args()
    # torch.manual_seed(args.seed)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    #
    #
    # DataLoader = LoadData(args.val_size)
    # (dataloaders, X_init) = DataLoader.load_all()
    #
    # if not os.path.exists(args.result_dir):
    #     os.mkdir(args.result_dir)
    #
    # results = train_active_learning(args, device, dataloaders, X_init)
    # plot_results(data=results)


if __name__ == "__main__":
    main()


