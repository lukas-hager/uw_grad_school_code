# %%

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import load_dataset, problem


@problem.tag("hw4-A")
def F1(h: int) -> nn.Module:
    """Model F1, it should performs an operation W_d * W_e * x as written in spec.

    Note:
        - While bias is not mentioned explicitly in equations above, it should be used.
            It is used by default in nn.Linear which you can use in this problem.

    Args:
        h (int): Dimensionality of the encoding (the hidden layer).

    Returns:
        nn.Module: An initialized autoencoder model that matches spec with specific h.
    """
    #raise NotImplementedError
    # 
    # ("Your Code Goes Here")
    model = nn.Sequential(
        nn.Linear(784, h),
        nn.Linear(h, 784)
    )
    
    return(model)


@problem.tag("hw4-A")
def F2(h: int) -> nn.Module:
    """Model F1, it should performs an operation ReLU(W_d * ReLU(W_e * x)) as written in spec.

    Note:
        - While bias is not mentioned explicitly in equations above, it should be used.
            It is used by default in nn.Linear which you can use in this problem.

    Args:
        h (int): Dimensionality of the encoding (the hidden layer).

    Returns:
        nn.Module: An initialized autoencoder model that matches spec with specific h.
    """
    #raise NotImplementedError("Your Code Goes Here")

    model = nn.Sequential(
        nn.Linear(784, h),
        nn.ReLU(),
        nn.Linear(h, 784),
        nn.ReLU()
    )

    return(model)


@problem.tag("hw4-A")
def train(
    model: nn.Module, optimizer: Adam, train_loader: DataLoader, epochs: int = 40
) -> float:
    """
    Train a model until convergence on train set, and return a mean squared error loss on the last epoch.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
            Hint: You can try using learning rate of 5e-5.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce x
            where x is FloatTensor of shape (n, d).

    Note:
        - Unfortunately due to how DataLoader class is implemented in PyTorch
            "for x_batch in train_loader:" will not work. Use:
            "for (x_batch,) in train_loader:" instead.

    Returns:
        float: Final training error/loss
    """
    #raise NotImplementedError("Your Code Goes Here")

    loss_fn = nn.MSELoss()
    final_epoch_loss = 0.

    for epoch in range(epochs):
        for (x_batch,) in tqdm(train_loader):
            optimizer.zero_grad()
            #model.forward()
            loss = loss_fn(model(x_batch), x_batch)
            loss.backward()
            optimizer.step()

            if epoch == epochs - 1:
                with torch.no_grad():
                    final_epoch_loss += loss.item()

    return(final_epoch_loss / len(train_loader))


@problem.tag("hw4-A")
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Evaluates a model on a provided dataset.
    It should return an average loss of that dataset.

    Args:
        model (Module): TRAINED Model to evaluate. Either F1, or F2 in this problem.
        loader (DataLoader): DataLoader with some data.
            You can iterate over it like a list, and it will produce x
            where x is FloatTensor of shape (n, d).

    Returns:
        float: Mean Squared Error on the provided dataset.
    """
    #raise NotImplementedError("Your Code Goes Here")

    loss_fn = nn.MSELoss()

    total_loss = 0.

    with torch.no_grad():
        for (x_test,) in loader:
            loss = loss_fn(model(x_test), x_test)
            total_loss += loss.item()

    return(total_loss / len(loader))


@problem.tag("hw4-A", start_line=9)
def main():
    """
    Main function of autoencoders problem.

    It should:
        A. Train an F1 model with hs 32, 64, 128, report loss of the last epoch
            and visualize reconstructions of 10 images side-by-side with original images.
        B. Same as A, but with F2 model
        C. Use models from parts A and B with h=128, and report reconstruction error (MSE) on test set.

    Note:
        - For visualizing images feel free to use images_to_visualize variable.
            It is a FloatTensor of shape (10, 784).
        - For having multiple axes on a single plot you can use plt.subplots function
        - For visualizing an image you can use plt.imshow (or ax.imshow if ax is an axis)
    """
    (x_train, y_train), (x_test, _) = load_dataset("mnist")
    x = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()

    # Neat little line that gives you one image per digit for visualization in parts a and b
    images_to_visualize = x[[np.argwhere(y_train == i)[0][0] for i in range(10)]]

    train_loader = DataLoader(TensorDataset(x), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test), batch_size=32, shuffle=True)
    #raise NotImplementedError("Your Code Goes Here")

    for h in [32,64,128]:
        for i, model in enumerate([F1(h), F2(h)]):
            
            model_name = ['F1', 'F2'][i]
            problem = ['a', 'b'][i]

            optimizer = Adam(model.parameters(), lr = 5e-5)
            mse_train = train(model, optimizer, train_loader)

            print('{} Hidden Layer Size {}: {}'.format(model_name, h, mse_train))

            reconstructions = model(images_to_visualize).detach().numpy()
            f, axarr = plt.subplots(2,10)
            for i in range(10):
                axarr[0,i].imshow(images_to_visualize[i].reshape((28,28)), cmap='Greys')
                axarr[1,i].imshow(reconstructions[i].reshape((28,28)), cmap='Greys')
                # axarr[0,i].title.set_text('Original Image')

            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
            f.tight_layout()
            f.suptitle('{} Original Images and Reconstructions (h={})'.format(model_name, h))
            f.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW4/A6{}_{}.png'.format(problem, h))
            plt.close('all')

            if h == 128:
                print('{} Test Error: {}'.format(model_name,evaluate(model, test_loader)))


# %%
if __name__ == "__main__":
    main()

# %%