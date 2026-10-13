"""Statistical properties of the price series: distribution, stationarity, autocorrelation, seasonality."""
import warnings

import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller

DAYS_PER_YEAR = 365   # crypto trades every day, so a year has 365 return observations, not 252


def descriptive_statistics(log_ret, periods_per_year=DAYS_PER_YEAR):
    """Mean, volatility and shape of the daily log returns, daily and annualised.

    Annualising: the mean scales with the number of days, the volatility with its square root.
    """
    return {
        'mean_daily': log_ret.mean(),
        'std_daily': log_ret.std(),
        'mean_annual': log_ret.mean() * periods_per_year,
        'vol_annual': log_ret.std() * periods_per_year ** 0.5,
        'skew': stats.skew(log_ret),
        'excess_kurtosis': stats.kurtosis(log_ret),        # 0 for a normal distribution
        'worst_day': log_ret.idxmin(),
        'best_day': log_ret.idxmax(),
    }


def normality_test(log_ret):
    """Jarque-Bera test: checks skewness and kurtosis together. H0 = returns are normally distributed."""
    result = stats.jarque_bera(log_ret)
    return {'statistic': result.statistic, 'p_value': result.pvalue}


def stationarity_test(series):
    """Augmented Dickey-Fuller test. H0 = the series has a unit root (is not stationary).

    A small p-value means the series is stationary. Prices usually fail this test, returns pass it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)   # newer statsmodels announces a change of return type
        stat, p_value, lags, nobs, crit, _ = adfuller(series, autolag='AIC')
    return {'adf_statistic': stat, 'p_value': p_value, 'lags_used': lags, 'critical_values': crit}


def autocorrelation(returns, lags=30):
    """Autocorrelation of returns and of absolute returns, plus the 95% band for 'no autocorrelation'.

    Returns themselves tell us whether the direction of tomorrow's move is predictable.
    Absolute returns tell us whether big moves follow big moves (volatility clustering).
    """
    band = 1.96 / len(returns) ** 0.5
    acf_ret = pd.Series(acf(returns, nlags=lags)[1:], index=range(1, lags + 1))
    acf_abs = pd.Series(acf(returns.abs(), nlags=lags)[1:], index=range(1, lags + 1))
    return acf_ret, acf_abs, band


def decompose(prices, period=30):
    """Multiplicative decomposition of the price into trend, seasonal factor and residual.

    Multiplicative fits better than additive because the swings grow with the price level.
    """
    return seasonal_decompose(prices, model='multiplicative', period=period)


def month_of_year_test(log_ret):
    """Kruskal-Wallis test: do daily returns differ between calendar months (January, February, ...)?

    Uses returns, not prices: price levels differ between months simply because of the trend.
    H0 = all months have the same return distribution.
    """
    groups = [g.values for _, g in log_ret.groupby(log_ret.index.month)]
    stat, p_value = stats.kruskal(*groups)
    return {'statistic': stat, 'p_value': p_value}
