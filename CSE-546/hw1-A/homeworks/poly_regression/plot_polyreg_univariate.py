import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    from polyreg import PolynomialRegression  # type: ignore
else:
    from .polyreg import PolynomialRegression

if __name__ == "__main__":
    """
        Main function to test polynomial regression
    """

    # load the data
    filePath = "data/polyreg/polydata.dat"
    file = open(filePath, "r")
    allData = np.loadtxt(file, delimiter=",")

    X = allData[:, [0]]
    y = allData[:, [1]]

    lambdas = [0,.1,.01]

    for val,lambda_val in enumerate(lambdas):

        # regression with degree = d
        d = 8
        model = PolynomialRegression(degree=d, reg_lambda=lambda_val)
        model.fit(X, y)

        # output predictions
        xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
        ypoints = model.predict(xpoints)

        # plot curve
        plt.figure()
        plt.plot(X, y, "rx")
        plt.title(f"PolyRegression with d = {d}, λ = {lambda_val}")
        plt.plot(xpoints, ypoints, "b-")
        plt.xlabel("X")
        plt.ylabel("Y")
        #plt.show()
        plt.savefig('/Users/hlukas/Google Drive/Grad School/2021-2022/Autumn/CSE 546/Assignments/HW1/A4_{}.png'.format(val))
