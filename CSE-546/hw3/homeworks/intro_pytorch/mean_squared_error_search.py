# %%

if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
    from .train import plot_model_guesses, train

from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)

# define the networks

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = LinearLayer(2, 2, generator=RNG)
        
    def forward(self, inputs):
        x = self.linear_0(inputs)
        return x

class NN_1_Sig(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = LinearLayer(2, 2, generator=RNG)
        self.linear_1 = LinearLayer(2, 2, generator=RNG)
        self.sig = SigmoidLayer()
        
    def forward(self, inputs):
        x = self.linear_0(inputs)
        x = self.sig.forward(x)
        x = self.linear_1(x)
        return x

class NN_1_ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = LinearLayer(2, 2, generator=RNG)
        self.linear_1 = LinearLayer(2, 2, generator=RNG)
        self.relu = ReLULayer()
        
    def forward(self, inputs):
        x = self.linear_0(inputs)
        x = self.relu.forward(x)
        x = self.linear_1(x)
        return x

class NN_2_Sig_ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = LinearLayer(2, 2, generator=RNG)
        self.linear_1 = LinearLayer(2, 2, generator=RNG)
        self.linear_2 = LinearLayer(2, 2, generator=RNG)
        self.sig = SigmoidLayer()
        self.relu = ReLULayer()
        
    def forward(self, inputs):
        x = self.linear_0(inputs)
        x = self.sig.forward(x)
        x = self.linear_1(x)
        x = self.relu.forward(x)
        x = self.linear_2(x)
        return x

class NN_2_ReLU_Sig(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = LinearLayer(2, 2, generator=RNG)
        self.linear_1 = LinearLayer(2, 2, generator=RNG)
        self.linear_2 = LinearLayer(2, 2, generator=RNG)
        self.sig = SigmoidLayer()
        self.relu = ReLULayer()
        
    def forward(self, inputs):
        x = self.linear_0(inputs)
        x = self.relu.forward(x)
        x = self.linear_1(x)
        x = self.sig.forward(x)
        x = self.linear_2(x)
        return x

@problem.tag("hw3-A")
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is also a 2-d vector of floats, but specifically with one being 1.0, while other is 0.0.
            Index of 1.0 in target corresponds to the true class.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to CrossEntropy accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    #raise NotImplementedError("Your Code Goes Here")
    correct = 0
    total = 0
    for (x,y) in dataloader:
        with torch.no_grad():
            pred = model(x)
            match = pred.argmax(dim=1) == y.argmax(dim=1)
            correct+=torch.sum(match).item()
            total += pred.shape[0]

    return(correct / total)



@problem.tag("hw3-A")
def mse_parameter_search(
    dataloader_train: DataLoader, dataloader_val: DataLoader
) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers

    Args:
        dataloader_train (DataLoader): Dataloader for training dataset.
        dataloader_val (DataLoader): Dataloader for validation dataset.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    #raise NotImplementedError("Your Code Goes Here")

    model_dict = {}
    
    # initialize models
    lin_reg = LinearRegression()
    nn_1_sig = NN_1_Sig()
    nn_1_relu = NN_1_ReLU()
    nn_2_sig_relu = NN_2_Sig_ReLU()
    nn_2_relu_sig = NN_2_ReLU_Sig()

    models = [
        lin_reg, 
        nn_1_sig, 
        nn_1_relu, 
        nn_2_sig_relu, 
        nn_2_relu_sig
    ]

    names = [
        'Linear Regression', 
        '1 Layer NN (Sigmoid)', 
        '1 Layer NN (ReLU)', 
        '2 Layer NN (Sigmoid, ReLU)', 
        '2 Layer NN (ReLU, Sigmoid)'
    ]

    results = {}

    for i, model in enumerate(models):

        print('Training Model: {}'.format(names[i]))

        # initialize loss
        mse = MSELossLayer()

        # initialize optimizer
        sgd = SGDOptimizer(model.parameters(), lr = 1e-2)

        results[names[i]] = train(
            dataloader_train, model, mse, sgd, dataloader_val, epochs = 100
        )

    return(results)



@problem.tag("hw3-A", start_line=21)
def main():
    """
    Main function of the MSE problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    mse_dataloader_train = DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(to_one_hot(y))),
        batch_size=32,
        shuffle=True,
        generator=RNG,
    )
    mse_dataloader_val = DataLoader(
        TensorDataset(
            torch.from_numpy(x_val).float(), torch.from_numpy(to_one_hot(y_val))
        ),
        batch_size=32,
        shuffle=False,
    )
    mse_dataloader_test = DataLoader(
        TensorDataset(
            torch.from_numpy(x_test).float(), torch.from_numpy(to_one_hot(y_test))
        ),
        batch_size=32,
        shuffle=False,
    )
    #raise NotImplementedError("Your Code Goes Here")
    results = mse_parameter_search(mse_dataloader_train, mse_dataloader_val)

    colors = ['r', 'g', 'b', 'y', 'm']

    model_names = list(results.keys())

    for i,key in enumerate(model_names):
        plt.plot(results[key]['train'], '{}--'.format(colors[i]), label = '{}: Train'.format(key))
        plt.plot(results[key]['val'], '{}-'.format(colors[i]), label = '{}: Validation'.format(key))
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('MSE Search Results')
    plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW3/A4b_mse.png')
    plt.close()

    mins = [min(results[x]['val']) for x in model_names]
    min_idx = mins.index(min(mins))
    best_model = results[model_names[min_idx]]['model']

    plot_model_guesses(
        mse_dataloader_test, 
        model = best_model, 
        title = 'MSE: {}'.format(model_names[min_idx]),
        suffix = 'mse'
    )

    print('Best Model Accuracy on Test Set: {}'.format(accuracy_score(best_model, mse_dataloader_test)))





def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Helper function. Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r

# %%
if __name__ == "__main__":
    main()


# %%

# lr = LinearRegression()
# print('Params')
# [print(x) for x in lr.parameters()]
# sgd = SGDOptimizer(lr.parameters(), lr = 1e-1)
# mse = MSELossLayer()

# loss = mse(lr(torch.randn(4).reshape((2,2))), torch.ones(4).reshape((2,2)))
# # mse(lr(torch.randn(4).reshape((2,2))), torch.ones(4).reshape((2,2))).backward()
# loss.backward()
# # print('Initial Loss')
# # print(loss)
# # loss.backward()
# # print('Backwards Loss')
# # print(loss)
# print('New Params')
# #[print(x) for x in lr.parameters()]

# with torch.no_grad():
#     for param in lr.parameters():
#         if param.grad is not None:
#             param.add_(param.grad, alpha = -1)

# # for param in lr.parameters():
# #     param = param - param.grad
# #     print(param)

# [print(x) for x in lr.parameters()]

# # [print(x) for x in lr.parameters()]
# # preds = lr(torch.arange(4.).reshape((2,2)))
# sgd.step()
# #[print(x.grad) for x in lr.parameters()]

# # mse = MSELossLayer()

# # mse(torch.arange(2.), torch.randn(2))

# %%
# a = torch.tensor([2., 3.], requires_grad=True)
# b = torch.tensor([6., 4.], requires_grad=True)

# Q = (3*a**3 - b**2).sum()

# Q.backward()

# print(a.grad)
# %%

(x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

mse_dataloader_train = DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(to_one_hot(y))),
        batch_size=32,
        shuffle=True,
        generator=RNG,
    )

for i, (x,y) in enumerate(iter(mse_dataloader_train)):
    print(i)

for i, (x,y) in enumerate(mse_dataloader_train):
    print(i)
# %%
