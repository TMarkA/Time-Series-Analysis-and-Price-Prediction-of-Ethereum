import yfinance as yf
import pandas as pd

def download_data(ticker, start_date, end_date):
    """
    Download historical cryptocurrency data from Yahoo Finance.
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    data = data.ffill()
    return data['Adj Close']

def calculate_daily_returns(adj_close):
    """
    Calculate daily returns from adjusted close prices.
    """
    return adj_close.pct_change().dropna()
