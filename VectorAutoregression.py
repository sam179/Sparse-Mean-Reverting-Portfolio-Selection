from sklearn import linear_model
import numpy as np
from statsmodels.tsa.api import VAR

ALPHA = 0.09


class VectorAutoregression:
    """
    A class to perform vector autoregression (VAR) analysis on given price data.

    Attributes:
        N (int): The number of assets in the data.
        T (int): The number of time periods in the data.
        prices (numpy.ndarray): A 2D array of asset prices, where the rows correspond to different assets
            and the columns correspond to different time periods.
        data (numpy.ndarray): The prices array with the last row removed, which is used to fit the model.

    Methods:
        lasso_coefficients(alpha=ALPHA): Returns the coefficient matrix A computed using LASSO regression.
        VAR_coefficients(prices, lag): Returns the coefficient matrix A computed using VAR analysis.
    """
    def __init__(self, prices):
        self.N = prices.shape[0]
        self.T = prices.shape[1]
        self.prices = prices.T
        self.data = self.prices[:-1, ]

    def lasso_coefficients(self, alpha=ALPHA):
        """
        Computes the coefficient matrix A using LASSO regression.

        Args:
            alpha (float): The regularization parameter for LASSO regression. Default is ALPHA.

        Returns:
            numpy.ndarray: The coefficient matrix A.
        """
        A = np.empty((self.data.shape[1], self.N))

        for i in range(self.N):
            labels = self.prices[1:, i]
            model = linear_model.Lasso(alpha=alpha)
            model.fit(self.data, labels)
            A[i, :] = model.coef_

        return A

    def VAR_coefficients(self, prices, lag):
        """
        Computes the coefficient matrix A using VAR analysis.

        Args:
            prices (numpy.ndarray): A 2D array of asset prices, where the rows correspond to different assets
                and the columns correspond to different time periods.
            lag (int): The number of lags to use in the VAR model.

        Returns:
            numpy.ndarray: The coefficient matrix A.
        """
        model = VAR(prices.T)
        results = model.fit(lag)
        A = results.coefs[0]

        return A
