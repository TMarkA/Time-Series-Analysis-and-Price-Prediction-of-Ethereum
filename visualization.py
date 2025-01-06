import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import probplot

def plot_prices(adj_close, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(adj_close)
    plt.title(f'{ticker} Price Trend')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.show()

def plot_log_returns(adj_close):
    log_returns = np.log(adj_close / adj_close.shift(1)).dropna()
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.hist(log_returns, bins=60, density=True, edgecolor='black')
    plt.title('Histogram of Log Returns')
    plt.subplot(1, 2, 2)
    probplot(log_returns, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Log Returns')
    plt.tight_layout()
    plt.show()

def seasonal_decomposition(adj_close):
    result = seasonal_decompose(adj_close, model='multiplicative', period=30)
    result.plot()
    plt.show()

def autocorrelation_plot(daily_returns):
    plt.figure(figsize=(14, 7))
    plot_acf(daily_returns)
    plt.title('ACF of Daily Returns')
    plt.show()
