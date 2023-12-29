from __future__ import annotations
from typing import Callable, Dict, Tuple, Literal, Union

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError as MAPE


import matplotlib.pyplot as plt

from tqdm import tqdm


# url https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks

# random
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


class TrainError(Exception):
    "encoder in ReduceModel isn't trained yet"


def get_cuda():
    """get device 
    Returns:
         torch.device: cuda if available else cpu
    """
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def RMSE(y_recon, y):
    return torch.sqrt(torch.mean((y_recon-y)**2))


class AE(nn.Module):
    losses = {
        "MSE": nn.MSELoss(),
        "RMSE": RMSE,
    }
    def __init__(self, layers: Tuple[int] = (1018, 300, 10), activation: Callable = nn.Sigmoid()):
        """Inner logic of Autoencoder

        Args:
            layers (Tuple[int], optional): number of neurons per layer. Defaults to (1018, 300, 10).
            activation (Callable, optional): torch.nn func. Defaults to nn.Sigmoid().
        """

        super(AE, self).__init__()

        self.layers = layers
        self.encoder = nn.Sequential()
        for i in range(len(layers)-2):
            self.encoder.append(nn.Linear(layers[i], layers[i+1]))
            self.encoder.append(activation)

        self.encoder.append(nn.Linear(layers[-2], layers[-1]))

        self.decoder = nn.Sequential()
        for i in range(len(layers)-1, 1, -1):
            self.decoder.append(nn.Linear(layers[i], layers[i-1]))
            self.decoder.append(activation)
        self.decoder.append(nn.Linear(layers[1], layers[0]))

        print(self.encoder)
        print(self.decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def _train(self, train_set: TensorDataset, test_set: TensorDataset,
               epochs: int, lr: float = 1e-5, batch_size: int = 128, loss_func: Literal["MSE", "RMSE"] = "MSE") -> Dict[str, object]:
        """train self

        Args:
            train_set (TensorDataset): train data
            test_set (TensorDataset): test data
            epochs (int): number of epochs
            lr (float, optional): learning rate. Defaults to 1e-5.
            batch_size (int, optional): batch size for DataLoader. Defaults to 128.
            loss_func (Literal["MSE", "RMSE"], optional): Loss function. Defaults to "MSE".

        Returns:
            Dict[str, object]: dict with hyper params and train/test losses  
        """

        # device = torch.device("cuda")
        device = get_cuda()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = self.losses[loss_func]
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

        train_loader = DataLoader(train_set, batch_size=batch_size, worker_init_fn=seed_worker,
                                  generator=g)
        val_loader = DataLoader(test_set, batch_size=batch_size, worker_init_fn=seed_worker,
                                generator=g)

        # storage for loss
        train_loss_list = [None]*epochs
        test_loss_list = [None]*epochs
        mape_list = [None]*epochs

        for epoch in tqdm(range(epochs)):
            self.train()
            train_loss = 0.
            for data, in train_loader:

                inputs = data.to(device)

                optimizer.zero_grad()

                outputs = self(inputs)

                loss = criterion(outputs, inputs)
                loss.backward()

                optimizer.step()

                train_loss += loss.item()

            train_loss /= train_loader.__len__()

            train_loss_list[epoch] = train_loss
            scheduler.step()
            # Validation loop
            val_loss = 0.
            # mape_ = 0.
            self.eval()
            with torch.no_grad():
                for data in val_loader:
                    inputs, = data
                    inputs = inputs.to(device)
                    outputs = self(inputs)
                    loss = criterion(outputs, inputs)
                    # mp = MAPE(device=device)(outputs, inputs)
                    val_loss += loss.item()
                    # mape_ += mp

                val_loss /= len(val_loader)
                # mape_ /= len(val_loader)

                # mape_list[epoch] = mape_
                test_loss_list[epoch] = val_loss

        train_results = {"model": "AE",
                         "epochs": epochs,
                         "learning_rate": lr,
                         "batch_size": batch_size,
                         "Loss": loss_func,
                         "Latent_space": self.layers[-1],
                         "train_loss": train_loss_list[-1],
                         "test_loss": test_loss_list[-1],
                         "train_loss_list": train_loss_list,
                         "test_loss_list": test_loss_list,
                        #  "mape_list": mape_list
                         }

        print(f'Epoch {epochs}, Train Loss: {train_loss_list[-1]}')
        print(f'Epoch {epochs}, Validation Loss: {test_loss_list[-1]}')
        return train_results

    def _transform(self, x: torch.Tensor) -> np.ndarray:
        """Reduce number of features

        Args:
            x (torch.Tensor): x with shape (input_layer_size, int)

        Returns:
            np.ndarray: reduced x
        """

        return self.encoder(x)


def load_data() -> Tuple[TensorDataset, TensorDataset]:
    """load qmof dataset

    Args:
        scale (Literal["minmax", "normalizer"], optional): scaler. Defaults to "normalizer".

    Returns:
        Tuple[DataLoader, DataLoader]: train and test TensorDatasets 
    """
    

    path_train = "preprocessing/datasets/qmof_train.csv"
    path_test = "preprocessing/datasets/qmof_train.csv"
    train = TensorDataset(torch.tensor(pd.read_csv(
        path_train, index_col=0).values, dtype=torch.float32))
    test = TensorDataset(torch.tensor(pd.read_csv(
        path_test, index_col=0).values, dtype=torch.float32))

    return train, test


class ReduceModel:
    device = get_cuda()
    train_set, test_set = load_data()
    dataset = torch.cat(
        (*train_set.tensors, *test_set.tensors)).to(device=device)

    def check_is_trained(self):
        if not self.trained:
            raise TrainError("model isn't trained yet")

    def __init__(self, **params) -> None:
        """model for reducing number of features

        Args:
            params are params of nn model class

        """

        self.trained = False
        self.device = get_cuda()
        self.model = AE(**params).to(self.device)

    def train(self, epochs: int, lr: float = 1e-3, batch_size: int = 128, loss_func: Literal['MSE', 'RMSE'] = "MSE", **kwargs) -> None:
        """train AE 

        Args:
            epochs (int): number of epochs to train 
            lr (float, optional): learning rate. Defaults to 1e-3.
            batch_size (int, optional): batch size for loader. Defaults to 128.
            loss_func (Literal["MSE", "MAE"], optional): Loss function. Defaults to "MSE".
            **kwargs: For special params 
        """

        if self.trained:
            raise TrainError("model is trained")

        self.train_results = self.model._train(self.train_set,
                                               self.test_set,
                                               epochs=epochs,
                                               lr=lr,
                                               batch_size=batch_size,
                                               loss_func=loss_func,
                                               **kwargs)

        self.trained = True

    # def plot_loss(self):
    #     """plot train/test loss vs epochs
    #     """

    #     self.check_is_trained()

    #     plt.plot(self.train_results["train_loss_list"],
    #              "g", label="train loss")
    #     plt.plot(self.train_results["test_loss_list"], "r", label="test loss")
    #     plt.xlabel("epochs")
    #     plt.ylabel(self.train_results["Loss"])
    #     plt.title(", ".join([f"{key}: {self.train_results[key]}" for key in self.train_results if key not in [
    #               "train_loss", "test_loss", "train_loss_list", "test_loss_list", "mape_list"]]))
    #     plt.legend()
    #     plt.show()

    def transform(self, x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """reducing data

        Args:
            x (Union[np.ndarray, pd.DataFrame]): data to reduce

        Raises:
            ValueError: if self.model isn't trained yet

        Returns:
            np.ndarray: reduced x
        """

        self.check_is_trained()

        x_torch = torch.Tensor(np.array(x)).to(self.device)
        x_reduced = self.model._transform(x_torch)

        return x_reduced.cpu().detach().numpy()