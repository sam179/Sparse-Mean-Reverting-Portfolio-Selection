import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import GraphicalLasso
from VectorAutoregression import VectorAutoregression
from StatisticalAnalysis import StatisticalAnalysis
import cvxpy as cp
from ETF import ETF

ALPHA = 0.00015
LAG = 1
PREDICTABILITY_LIMIT = 0.0015
CLUSTER_ID = 2


class SparseAssetClusters:
    def __init__(self, prices, asset_names):
        """
        Initialize SparseAssetClusters instance.
        :param prices: array-like, shape (n_samples, n_features)
            The historical prices of the assets.
        :param asset_names: list
            List of asset names.
        """
        self.returns = None
        self.prices = prices
        self.asset_names = asset_names
        self.clusters = None
        self.cluster_prices = None
        self.cluster_returns = None
        self.portfolio_weights = None
        self.graphical_lasso = None
        self.cluster_names = None

    def calculate_returns(self):
        """
        Calculate the returns of the assets.
        """
        self.returns = (np.diff(self.prices, axis=1) / self.prices[:, :-1]).T
        return self.returns

    def graphical_lasso_selection(self, alpha=ALPHA):
        """
        Perform Graphical Lasso selection.
        :param alpha: float, optional (default=0.00015)
            The regularization parameter.
        :return: array-like, shape (n_features, n_features)
            The inverse covariance matrix.
        """
        covariance_matrix_inverse = GraphicalLasso(alpha=alpha).fit(self.returns).precision_
        return covariance_matrix_inverse

    def find_asset_clusters(self):
        """
        Find asset clusters.
        :return: array-like, shape (n_features, )
            The array containing cluster assignments for each asset.
        """
        graphical_lasso = self.graphical_lasso_selection()
        self.graphical_lasso = graphical_lasso
        vector_autoregression = VectorAutoregression(self.prices)
        var_coeffients = vector_autoregression.lasso_coefficients()
        var_symmetric = var_coeffients.T @ var_coeffients

        intersection = np.multiply(graphical_lasso, var_symmetric)
        self.clusters = self.depth_first_search(intersection)
        unique, counts = np.unique(self.clusters, return_counts=True)
        print('[Cluster Index, Cluster Size')
        print(np.asarray((unique, counts)).T)

        return intersection

    def depth_first_search(self, Graph):
        """
        Perform depth-first search to find clusters.
        :param Graph: array-like, shape (n_features, n_features)
            The intersection matrix of Graphical Lasso and VAR coefficients.
        :return: array-like, shape (n_features, )
            The array containing cluster assignments for each asset.
        """
        n = Graph.shape[0]
        visited = np.zeros(n)
        clusters = np.zeros(n)
        cluster_index = 0

        for i in range(n):
            if visited[i] == 0:
                cluster_index += 1
                self.traverse(Graph, clusters, visited, cluster_index, i)
        return clusters

    def traverse(self, Graph, clusters, visited, cluster_index, i):
        """
        Traverse the given graph and assign each node to a cluster based on their connectivity.

        Args:
        - Graph (ndarray): An adjacency matrix representing the graph.
        - clusters (ndarray): An array to store the cluster index for each node.
        - visited (ndarray): An array to keep track of visited nodes.
        - cluster_index (int): The index of the cluster to assign nodes to.
        - i (int): The index of the node to start the traversal from.

        """
        n = Graph.shape[0]
        visited[i] = 1
        clusters[i] = cluster_index

        for k in range(i + 1, n):
            if visited[k] == 0 and Graph[i][k] != 0:
                self.traverse(Graph, clusters, visited, cluster_index, k)

    def select_cluster(self, cluster_id):
        """
        Select the assets belonging to a particular cluster.

        Args:
        - cluster_id (int): The index of the cluster.

        Returns:
        An array of asset indices belonging to the specified cluster.
        """
        assets = np.array([])
        clusters = self.clusters

        for idx in range(len(clusters)):
            if clusters[idx] == cluster_id:
                assets = np.append(assets, idx)

        return assets

    def cluster_data(self, cluster_assets):
        """
        Cluster the data for the specified assets.

        Args:
        - cluster_assets (ndarray): An array of asset indices to cluster.

        Returns:
        - cluster_prices (ndarray): An array of prices for the specified assets.
        - cluster_covariance_matrix (ndarray): The covariance matrix of returns for the specified assets.
        - VAR_coefficients (ndarray): The VAR coefficients for the specified assets.
        """
        cluster_prices = []
        cluster_names = []
        for asset in cluster_assets:
            cluster_prices.append(self.prices[int(asset), :])
            cluster_names.append(self.asset_names[int(asset)])
        self.cluster_names = np.array(cluster_names)
        self.cluster_prices = np.array(cluster_prices)
        self.cluster_returns = np.diff(self.cluster_prices, axis=1) / self.cluster_prices[:, :-1]
        cluster_covariance_matrix = EmpiricalCovariance().fit(self.cluster_returns.T).covariance_
        VAR_coefficients = VectorAutoregression(self.prices).VAR_coefficients(self.cluster_prices, LAG)

        return self.cluster_prices, cluster_covariance_matrix, VAR_coefficients

    def box_tiao_optimization(self, cluster_covariance_matrix, VAR_coefficients):
        """
        Perform the Box-Tiao optimization to obtain the portfolio weights.

        Args:
        - cluster_covariance_matrix (ndarray): The covariance matrix of returns for the specified assets.
        - VAR_coefficients (ndarray): The VAR coefficients for the specified assets.

        Returns:
        The optimal portfolio weights for the specified assets.
        """
        n = self.cluster_prices.shape[0]
        X = cp.Variable((n, n), PSD=True)

        obj = cp.Maximize(cp.trace(cluster_covariance_matrix @ X))

        constraints = [
            cp.trace(VAR_coefficients @ cluster_covariance_matrix @ VAR_coefficients @ X) <= PREDICTABILITY_LIMIT,
            cp.trace(X) == 1,

        ]

        prob = cp.Problem(obj, constraints)
        prob.solve()

        print("Status: ", prob.status)
        print("Optimal value: ", prob.value)
        # print("Solution:\n", X.value)

        eig_vals, eig_vecs = np.linalg.eigh(X.value)
        self.portfolio_weights = X.value.T.dot(eig_vecs[:, -1]) / np.sum(eig_vecs[:, -1])

        return self.portfolio_weights


if __name__ == "__main__":
    etf = ETF()
    prices = etf.get_etf_data()
    sac = SparseAssetClusters(prices, etf.etfs)
    sac.calculate_returns()
    asset_clusters = sac.find_asset_clusters()
    clusters = sac.depth_first_search(asset_clusters)

    cluster_assets = sac.select_cluster(CLUSTER_ID)
    _, cov, A = sac.cluster_data(cluster_assets)
    weights = sac.box_tiao_optimization(cov, A)
    portfolio_returns = weights.T @ sac.cluster_returns
    sa = StatisticalAnalysis(sac.returns, sac.prices, sac.asset_names, sac.cluster_prices, sac.cluster_returns,
                             sac.portfolio_weights, PREDICTABILITY_LIMIT, sac.cluster_names)
    sa.graphical_lasso_heatmap(sac.graphical_lasso)

    half_life = sa.calculate_half_life(portfolio_returns)
    sa.portfolio_return_plot()
    sa.weight_allocation_table()
    sa.test_stationarity()

