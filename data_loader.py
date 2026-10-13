"""Loading daily ETH/USD prices and turning them into returns."""
import numpy as np
import pandas as pd

COINMETRICS_URL = 'https://raw.githubusercontent.com/coinmetrics/data/master/csv/{asset}.csv'


def load_prices(start, end, asset='eth'):
    """Daily closing prices in USD as a pandas Series, from start to end (both dates included).

    The prices are the Coin Metrics daily reference rate (column PriceUSD), from their free community data on
    GitHub. No API key is needed and the history doesn't change between downloads.
    """
    data = pd.read_csv(COINMETRICS_URL.format(asset=asset), usecols=['time', 'PriceUSD'], parse_dates=['time'])
    prices = data.set_index('time')['PriceUSD']
    prices = prices.loc[start:end].asfreq('D').ffill().dropna()   # ETH trades every day; fill rare gaps
    prices.name = 'Price'
    prices.index.name = 'Date'
    return prices


def simple_returns(prices):
    """Daily percentage change of the price."""
    return prices.pct_change().dropna()


def log_returns(prices):
    """Daily log returns. They add up over time, which is what the simulation and the tests need."""
    return np.log(prices).diff().dropna()
