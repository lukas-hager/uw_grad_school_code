if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from losses import CrossEntropyLossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from .optimizers import SGDOptimizer
    from .losses import CrossEntropyLossLayer
    from .train import plot_model_guesses, train

from typing import Any, Dict

import numpy as np
import torch
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
        self.softmax = SoftmaxLayer()
        
    def forward(self, inputs):
        x = self.linear_0(inputs)
        x = self.softmax(x)
        return x

class NN_1_Sig(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = LinearLayer(2, 2, generator=RNG)
        self.linear_1 = LinearLayer(2, 2, generator=RNG)
        self.sig = SigmoidLayer()
        self.softmax = SoftmaxLayer()
        
    def forward(self, inputs):
        x = self.linear_0(inputs)
        x = self.sig.forward(x)
        x = self.linear_1(x)
        x = self.softmax(x)
        return x

class NN_1_ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = LinearLayer(2, 2, generator=RNG)
        self.linear_1 = LinearLayer(2, 2, generator=RNG)
        self.relu = ReLULayer()
        self.softmax = SoftmaxLayer()
        
    def forward(self, inputs):
        x = self.linear_0(inputs)
        x = self.relu.forward(x)
        x = self.linear_1(x)
        x = self.softmax(x)
        return x

class NN_2_Sig_ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = LinearLayer(2, 2, generator=RNG)
        self.linear_1 = LinearLayer(2, 2, generator=RNG)
        self.linear_2 = LinearLayer(2, 2, generator=RNG)
        self.sig = SigmoidLayer()
        self.relu = ReLULayer()
        self.softmax = SoftmaxLayer()
        
    def forward(self, inputs):
        x = self.linear_0(inputs)
        x = self.sig.forward(x)
        x = self.linear_1(x)
        x = self.relu.forward(x)
        x = self.linear_2(x)
        x = self.softmax(x)
        return x

class NN_2_ReLU_Sig(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = LinearLayer(2, 2, generator=RNG)
        self.linear_1 = LinearLayer(2, 2, generator=RNG)
        self.linear_2 = LinearLayer(2, 2, generator=RNG)
        self.sig = SigmoidLayer()
        self.relu = ReLULayer()
        self.softmax = SoftmaxLayer()
        
    def forward(self, inputs):
        x = self.linear_0(inputs)
        x = self.relu.forward(x)
        x = self.linear_1(x)
        x = self.sig.forward(x)
        x = self.linear_2(x)
        x = self.softmax(x)
        return x


@problem.tag("hw3-A")
def crossentropy_parameter_search(
    dataloader_train: DataLoader, dataloader_val: DataLoader
) -> Dict[str, Any]:
    """
    Main subroutine of the CrossEntropy problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers
    NOTE: Each model should end with a Softmax layer due to CrossEntropyLossLayer requirement.

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
        ce = CrossEntropyLossLayer()

        # initialize optimizer
        sgd = SGDOptimizer(model.parameters(), lr = 7e-3)

        results[names[i]] = train(
            dataloader_train, model, ce, sgd, dataloader_val, epochs = 100
        )

    return(results)



@problem.tag("hw3-A")
def accuracy_score(model, dataloader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for CrossEntropy.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is an integer representing a correct class to a corresponding observation.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to MSE accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    #raise NotImplementedError("Your Code Goes Here")

    correct = 0
    total = 0
    for (x,y) in dataloader:
        with torch.no_grad():
            pred = model(x)
            match = pred.argmax(dim=1) == y
            correct+=torch.sum(match).item()
            total += pred.shape[0]

    return(correct / total)


@problem.tag("hw3-A", start_line=21)
def main():
    """
    Main function of the Crossentropy problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me Crossentropy loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    ce_dataloader_train = DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y)),
        batch_size=32,
        shuffle=True,
        generator=RNG,
    )
    ce_dataloader_val = DataLoader(
        TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val)),
        batch_size=32,
        shuffle=False,
    )
    ce_dataloader_test = DataLoader(
        TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test)),
        batch_size=32,
        shuffle=False,
    )

    results = crossentropy_parameter_search(ce_dataloader_train, ce_dataloader_val)

    #raise NotImplementedError("Your Code Goes Here")

    colors = ['r', 'g', 'b', 'y', 'm']

    model_names = list(results.keys())

    for i,key in enumerate(model_names):
        plt.plot(results[key]['train'], '{}--'.format(colors[i]), label = '{}: Train'.format(key))
        plt.plot(results[key]['val'], '{}-'.format(colors[i]), label = '{}: Validation'.format(key))
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('CE')
    plt.title('CE Search Results')
    plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW3/A4b_ce.png')
    plt.close()

    mins = [min(results[x]['val']) for x in model_names]
    min_idx = mins.index(min(mins))
    best_model = results[model_names[min_idx]]['model']

    plot_model_guesses(
        ce_dataloader_test, 
        model = best_model, 
        title = 'CE: {}'.format(model_names[min_idx]),
        suffix = 'ce'
    )

    print('Best Model Accuracy on Test Set: {}'.format(accuracy_score(best_model, ce_dataloader_test)))



if __name__ == "__main__":
    main()
