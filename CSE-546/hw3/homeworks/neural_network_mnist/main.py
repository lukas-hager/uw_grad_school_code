# %%

# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
from numpy.core.numeric import cross
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Optional
from torchinfo import summary

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)

class LinearLayer(Module):
    def __init__(
        self, dim_in: int, dim_out: int, generator: Optional[torch.Generator] = None
    ):
        """Linear Layer, which performs calculation of: x @ weight + bias

        In constructor you should initialize weight and bias according to dimensions provided.
        You should use torch.randn function to initialize them by normal distribution, and provide the generator if it's defined.

        Both weight and bias should be of torch's type float.
        Additionally, for optimizer to work properly you will want to wrap both weight and bias in nn.Parameter.

        Args:
            dim_in (int): Number of features in data input.
            dim_out (int): Number of features output data should have.
            generator (Optional[torch.Generator], optional): Generator to use when creating weight and bias.
                If defined it should be passed into torch.randn function.
                Defaults to None.

        Note:
            - YOU ARE NOT ALLOWED to use torch.nn.Linear (or it's functional counterparts) in this class
            - Make use of pytorch documentation: https://pytorch.org/docs/stable/index.html
        """
        super().__init__()
        #raise NotImplementedError("Your Code Goes Here")

        alpha = 1./(dim_in ** .5)

        self.weight = Parameter(
            torch.rand(
                (dim_in,dim_out),
                generator = generator,
                requires_grad=True
            ) * 2 * alpha - alpha
        )
        self.bias = Parameter(
            torch.rand(
                dim_out,
                generator = generator,
                requires_grad=True
            ) * 2 * alpha - alpha
        )

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Actually perform multiplication x @ weight + bias

        Args:
            x (torch.Tensor): More specifically a torch.FloatTensor, with shape of (n, dim_in).
                Input data.

        Returns:
            torch.Tensor: More specifically a torch.FloatTensor, with shape of (n, dim_out).
                Output data.
        """
        #raise NotImplementedError("Your Code Goes Here")
        return(torch.mm(x, self.weight) + self.bias)

class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        #raise NotImplementedError("Your Code Goes Here")
        self.linear_0 = LinearLayer(d,h,generator=RNG)
        self.linear_1 = LinearLayer(h,k,generator=RNG)

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        #raise NotImplementedError("Your Code Goes Here")
        x_out = self.linear_0(x)
        x_out = relu(x_out)
        x_out = self.linear_1(x_out)
        return(x_out)


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        #raise NotImplementedError("Your Code Goes Here")
        self.linear_0 = LinearLayer(d,h0,generator=RNG)
        self.linear_1 = LinearLayer(h0,h1,generator=RNG)
        self.linear_2 = LinearLayer(h1,k,generator=RNG)

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operati
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        #raise NotImplementedError("Your Code Goes Here")
        x_out = self.linear_0(x)
        x_out = relu(x_out)
        x_out = self.linear_1(x_out)
        x_out = relu(x_out)
        x_out = self.linear_2(x_out)
        return(x_out)


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    #raise NotImplementedError("Your Code Goes Here")

    loss_list = []
    mean_loss = 1.
    iter_val = 0

    while mean_loss > .01:
        iter_val += 1
        epoch_loss = 0.

        for (x,y) in tqdm(train_loader):
            optimizer.zero_grad()
            preds = model(x)
            loss = cross_entropy(preds,y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                epoch_loss += loss.item()

        mean_loss = epoch_loss / len(train_loader)
        print('Mean Loss from Epoch {}: {}'.format(iter_val, mean_loss))
        loss_list.append(mean_loss)

    return(loss_list)


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    #raise NotImplementedError("Your Code Goes Here")

    data_loader_train = DataLoader(
        TensorDataset(x,y),
        batch_size=32,
        shuffle=True,
        generator=RNG,
    )

    data_loader_test = DataLoader(
        TensorDataset(x_test,y_test),
        batch_size=32,
        shuffle=True,
        generator=RNG,
    )

    d = 784
    k = 10

    f1 = F1(64,d,k)
    f2 = F2(32,32,d,k)
    lr = 5e-4

    results_f1 = train(f1, Adam(f1.parameters(), lr = lr), data_loader_train)
    results_f2 = train(f2, Adam(f2.parameters(), lr = lr), data_loader_train)

    # plot the results
    plt.plot(results_f1, 'r', label = 'F1')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('F1 Model (Shallow)')
    plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW3/A5a.png')
    plt.close()


    plt.plot(results_f2, 'r', label = 'F2')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('F2 Model (Deep)')
    plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW3/A5b.png')
    plt.close()

    # calculate the loss and accuracy

    test_loss_f1 = 0.
    test_loss_f2 = 0.

    correct_f1 = 0.
    correct_f2 = 0.

    for (x,y) in data_loader_test:
        test_loss_f1 += cross_entropy(f1(x), y).item()
        test_loss_f2 += cross_entropy(f2(x), y).item()

        correct_f1 += torch.sum(f1(x).argmax(dim=1) == y).item()
        correct_f2 += torch.sum(f2(x).argmax(dim=1) == y).item()

    print('Test Loss (F1): {}'.format(test_loss_f1 / len(data_loader_test)))
    print('Test Loss (F2): {}'.format(test_loss_f2 / len(data_loader_test)))
    print('Accuracy (F1): {}'.format(correct_f1 / len(y_test)))
    print('Accuracy (F2): {}'.format(correct_f2 / len(y_test)))

    print('F1 Summary:')
    summary(f1, input_size=(32, 784))
    print('F2 Summary:')
    summary(f2, input_size=(32, 784))


# %%
if __name__ == "__main__":
    main()

# %%
# (x, y), (x_test, y_test) = load_dataset("mnist")
# x = torch.from_numpy(x).float()
# y = torch.from_numpy(y).long()
# x_test = torch.from_numpy(x_test).float()
# y_test = torch.from_numpy(y_test).long()


# %%
