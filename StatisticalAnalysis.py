import numpy as np
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss, coint
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticalAnalysis:
    def __init__(self, returns, prices, asset_names, cluster_prices, cluster_returns, portfolio_weights, predictability_limit, cluster_names):
        self.returns = returns
        self.prices = prices
        self.asset_names = asset_names
        self.cluster_prices = cluster_prices
        self.cluster_returns = cluster_returns
        self.portfolio_weights = portfolio_weights
        self.half_life = None
        self.predictability_limit = predictability_limit
        self.mean_reversion_speed = None
        self.cluster_names = cluster_names

    def calculate_half_life(self, y):
        """
        Calculate the half-life of a mean-reverting time series using the Augmented Dickey-Fuller (ADF) test.

        Parameters:
        y (array-like): The time series data.

        Returns:
        The half-life of the mean-reverting time series.
        """

        # Calculate the first difference of the series
        delta_y = np.diff(y)

        # Calculate the lagged first difference
        lagged_delta_y = np.roll(delta_y, 1)

        # Create the lagged first difference and constant columns
        X = np.column_stack((lagged_delta_y, np.ones_like(lagged_delta_y)))

        # Trim the first element of y so the dimensions match
        y = y[1:]

        # Fit a linear regression model to the lagged first difference and constant columns
        model = np.linalg.lstsq(X, y, rcond=None)
        alpha = model[0][1]
        beta = model[0][0]

        # Calculate the residual of the linear regression model
        residuals = y - (beta * lagged_delta_y + alpha)

        # Calculate the ADF test statistic and p-value
        adf = adfuller(residuals, maxlag=1)

        # Calculate the half-life using the ADF test statistic
        half_life = -(np.log(2) / beta)

        self.half_life = half_life
        return half_life

    def calculate_mean_reversion_speed(self):
        """
        Calculates the mean reversion speed of a time series using the Ornstein-Uhlenbeck equation.

        Parameters:
        prices (array): A SD numpy array of arrays of the time series prices.

        Returns:
        float: The mean reversion speed of the time series.
        """
        returns = self.portfolio_weights.T @ self.cluster_returns
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)
        mean_reversion_speed = -mean_returns / (std_returns ** 2)
        self.mean_reversion_speed = mean_reversion_speed
        return mean_reversion_speed

    def portfolio_return_plot(self):
        """
        Plots the mean reversion of portfolio returns, with predictability and half life.

        Parameters:
        self (obj): An instance of the SAC class.

        Returns:
        None.
        """

        plt.figure(figsize=(14, 5))
        plt.title('Mean Reversion of Portfolio Returns, with predictability: {}, and half life: {}'.format(self.predictability_limit, np.round(self.half_life, decimals=2)))
        plt.plot(self.portfolio_weights.T @ self.cluster_returns)
        plt.show()


    def graphical_lasso_heatmap(self, precision_matrix):
        """
        Plots a heatmap of the precision matrix created by the Graphical Lasso algorithm.

        Parameters:
        self (obj): An instance of the SAC class.
        precision_matrix (np.array): A numpy array of shape (n_clusters, n_clusters) representing the precision matrix.

        Returns:
        None.
        """

        plt.figure(figsize=(14, 5))
        plt.title('Graphical Lasso makes the matrix sparse by removing weaker correlations.')
        sns.heatmap(precision_matrix, cmap='coolwarm', square=True)
        plt.show()


    def weight_allocation_table(self):
        """
        Prints a table of weights allocated to each cluster in the portfolio.

        Parameters:
        self (obj): An instance of the SAC class.

        Returns:
        None.
        """

        weight_table = pd.DataFrame([self.portfolio_weights.T], columns=sac.cluster_names)
        styled_df = weight_table.style\
        .set_table_styles([{'selector': 'thead th', 'props': [('background-color', 'black'), ('color', 'white')]}])\
        .set_properties(**{'text-align': 'center'})\
        .set_caption('Weights Allocation')

        # display the styled dataframe
        display(styled_df)


    def test_stationarity(self):
        """
        Tests whether the time series is stationary (mean-reverting).

        Parameters:
        self (obj): An instance of the SAC class.

        Returns:
        None.
        """

        result = adfuller(self.portfolio_weights.T @ self.cluster_returns)
        test_statistic = result[0]
        p_value = result[1]

        # Check if the test statistic is less than the critical value at 5% significance level
        if test_statistic < result[4]['5%']:
            print("Reject null hypothesis: time series is stationary (mean-reverting)")
            print('P-Value, ', p_value)

        else:
            print("Fail to reject null hypothesis: time series is not stationary (not mean-reverting)")
