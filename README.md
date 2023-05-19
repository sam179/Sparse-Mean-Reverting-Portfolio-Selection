# Sparse Mean-Reverting Portfolio Selection

One of the most fundamental signals in statistical arbitrage is the mean-reversion properties of assets. Most commonly used strategy that utilizes mean-reversion is a Pairs Trading strategy, where cointegrated pair of assets are traded. However, when the size of the of pool of assets available increases, conventional Pairs Trading techniques becomes more and more complicated to implement. Hence, a sophisticated multi-asset oriented methodology is required to provide a tangible solution to the curse of dimensionality.

There are many advantages of a sparse portfolio. Firstly, the number of assets in the investible universe can be rather large. Secondly, trading a large number of assets is complicated. PnL attribution is extemely hard. Managing the assets is tricky, as there can be liquidity issues. Importantly, transactions costs can get very high. 


The objectives of this paper are to develop a trading strategy such that,

- The portfolio, as a linear combination of the assets, should be strongly mean-reverting.
- The portfolio should be sparse, i.e. the cardinality of portfolio K << N, where N is the total number of assets in the investing universe.



The mean reversion characteristic of a portfolio is measured through a metric of predictability. 


### Predictability and Box Tiao Canonical Decomposition

Suppose that the portfolio value $\Pi$ follows the recurrence below:

$$
\Pi_t = \hat{\Pi}_{t-1} + \varepsilon_t
$$

where $\hat{\Pi}_{t-1}$ is the predicted value of $\Pi_t$ based on all the past portfolio values $\Pi_0$, $\Pi_1$, ..., $\Pi_{t-1}$, and $\varepsilon_t \sim N(0, \Sigma)$ is a vector of i.i.d Gaussian noise independent of all past portfolio values up to $t-1$.

Now let $\text{var}(\Pi_t) = \sigma^2$, $\text{var}(\hat{\Pi}_{t-1}) = \hat{\sigma}^2$, and by definition we know $\text{var}(\varepsilon_t) = \Sigma$.

Therefore,

$$
\text{var}(\Pi_t) = \text{var}(\hat{\Pi}_{t-1} + \varepsilon_t) = \text{var}(\hat{\Pi}_{t-1}) + \text{var}(\varepsilon_t)
$$

The second equal sign stands due to the independence between $\hat{\Pi}_{t-1}$ and $\varepsilon_t$.

Substitute in the variance, we get:

$$
\sigma^2 = \hat{\sigma}^2 + \Sigma
$$

$$
1 = \frac{\hat{\sigma}^2}{\sigma^2} + \frac{\Sigma}{\sigma^2}
$$

The predictability is then defined as

$$
\nu := \frac{\hat{\sigma}^2}{\sigma^2}
$$

The intuition behind this metric is that if predictability is small, then white noise is the most dominant factor in the portfolio, which infers that the process is strongly mean-reverting. However, when $\nu$ is large, the portfolio value is almost perfectly predictable based on its past information and this usually leads to a momentum portfolio. Therefore, we want to minimize $\nu$ to obtain a mean-reverting portfolio.

Box and Tiao (1977) proposed that the asset prices$\ S_t$ could be assumed to follow a vector autoregressive model of order one – a VAR(1) model. Under this assumption, the predictability can be represented by the covariance matrix of$\ S_t$, denoted by $\Gamma_0$, and the least square estimate of the VAR(1) coefficient matrix A:

$$
\nu = \frac{x^T A \Gamma_0 A^T x}{x^T \Gamma_0 x}
$$

Let’s see how the optimization problem looks now that we have an expression of predictability:

$$
\text{minimize } \nu = \frac{x^T A \Gamma_0 A^T x}{x^T \Gamma_0 x} \\
\text{subject to } \lVert x \rVert_0 = k
$$

## Generating Sparsity

We generate a sparse portfolio by dividing the assets into clusters, which should reflect the underlying structure of asset prices. 

For this, we combine two methodologies. 

### Covariance Selection

Under the assumption of a Gaussian model, the inverse covariance matrix, $\Gamma_0^{-1}$, are indicators of conditional independence between variables. If we can set a reasonable portion of elements in $\Gamma_0^{-1}$
to zero, the remaining elements can represent the underlying structure of the asset price dependencies.

This means we can use an undirected graph to represent the underlying structure of asset price dynamics. The sparse estimate can be obtained by a graphical Lasso model, which solves a penalized maximum likelihood estimation problem:

$$
\max_X \, \ln \det X - \mathrm{Tr}(\Sigma X) - \alpha \lVert X \rVert_1
$$

Here, $\alpha > 0$ is a parameter that determines how many zero elements there will be in the final sparse estimate. 


### Structured VAR estimate

According to the Box and Tiao (1977) paper, we model asset price $\ S_t$ as a VAR(1) process, 

$$
\ S_t = S_{t-1} A + Z_t
$$


where$\ S_{t-1}$ is the lag-one process of$\ S_t$, and$\ Z_t$ $\sim$ N(0, $\Sigma$) is an i.i.d Gaussian noise independent of$\ S_{t-1}$. The dense least square estimate of A is then obtained by minimizing the objective function along with an$\ l_1$ norm penalty given by the following expression:

$$
\text{arg min}_{a} \, \lVert S_{it} - S_{t-1}a \rVert^2 + \lambda \lVert a \rVert_1 \text{ for all } i
$$

The power of Lasso lies in that we can append different forms of$\ l_1$ norm to exert control on the final structure of the sparse estimate of A.

### Cluster Creation

Both the inverse covariance matrix $\Gamma_0^{-1}$ and the VAR(1) coefficient matrix A contains the conditional dependence structure of the asset price dynamics. We will combine the information from both matrices to obtain clusters from which we can build our sparse mean-reverting portfolio.


Below is a visual illustration of how we create these clusters. 

![Alt Text](https://hudsonthames.org/wp-content/uploads/2021/02/cluster-1.gif)

## Convex Relaxation

We apply convex relaxation to the original optimization problem to be able to produce a stable solution. 

From an $\ell_{0}$ constraint, 

$$
\begin{aligned}
\text{minimize } \nu = \frac{x^T A \Gamma_0 A^T x}{x^T \Gamma_0 x} \\
\text{subject to } \lVert x \rVert_0 = k
\end{aligned}
$$

we switch to an $\ell_{1}$ constraint and we move the denominator of the ratio to constraints so that the problem is a proper convex problem. 

$$
\begin{aligned}
& \text{minimize } \quad \mathbf{Tr} (A \Gamma_0 A^T X) + \rho \lVert X \rVert_1 \\
& \text{subject to } \quad \mathbf{Tr} (\Gamma_0 X) \geq V \\
& \phantom{\text{subject to }} \quad \mathbf{Tr}(X) = 1
\end{aligned}
$$

In our implementation, however, to make the solution even more stable, we work on one of the recommendations given in the paper and convert the minimizing problem to a maximizing problem and setting an upper bound on the numerator. 

$$
\begin{aligned}
& \text{maximize } \quad \mathbf{Tr} (\Gamma_0 X) + \rho \lVert X \rVert_1 \\
& \text{subject to } \quad \mathbf{Tr} (A \Gamma_0 A^T X) \leq V \\
& \phantom{\text{subject to }} \quad \mathbf{Tr}(X) = 1
\end{aligned}
$$

Here $\ V$ becomes our predictability limit. We would like to limit the predictability in order to preserve the mean reversion. 

## Tests

### ADF 

Augmented Dicky Fuller is a standard statistical test. It is used to test whether a given Time series is stationary or not. Since this is a universally used testing technique, we won't delve in the functionality of the test but rather it's application in our project. 

1. We use ADF to calculate half life of the time series. A half life parameter is a measurement of the mean reversion speed, which estimates the time it would take for a signal in a time series to approach the midpoint with its mean and current value. Alternative parameters can be Ornstein Uhlenbeck mean-reversion speed. We work with half-life in the project as I believe it addresses the most direct application of the project i.e. it gives an approximate trading window. To optimally run a mean-reverting strategy, horizon of the trade should be short, meaning mean-reversion should be stronger than a month. This is the horizon used in the paper, and we agree with it. 


2. Additionally, we run a Null Hypothesis test on the ADF test statistic. This ascertains the stationarity of the final portfolio. 


### Data

We use a list of 44 international ETFs as the available assets. We consider 5-year prices from 2016 to 2021. The list is in an excel file which will be attached in the submission. 
