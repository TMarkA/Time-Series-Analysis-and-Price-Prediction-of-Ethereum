import numpy as np
import statsmodels.tsa.stattools as tsa
from scipy.stats import kruskal

def descriptive_statistics(daily_returns):
    """
    Calculate mean and standard deviation of daily returns.
    """
    mean_return = daily_returns.mean()
    std_dev_return = daily_returns.std()
    return mean_return, std_dev_return

def stationarity_test(daily_returns):
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    """
    adf_test = tsa.adfuller(daily_returns, autolag='AIC')
    return {
        'ADF Statistic': adf_test[0],
        'p-value': adf_test[1],
        'Critical Values': adf_test[4]
    }

def seasonality_test(data):
    """
    Perform Kruskal-Wallis test for seasonality.
    """
    data['YearMonth'] = data.index.to_period('M')
    groups = [group['Close'].values for _, group in data.groupby('YearMonth')]
    stat, p_value = kruskal(*groups)
    return stat, p_value
