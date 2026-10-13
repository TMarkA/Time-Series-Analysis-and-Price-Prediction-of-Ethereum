"""Two ways to look six months ahead: a SARIMA model on monthly prices and a Monte Carlo simulation."""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis import DAYS_PER_YEAR


def sarima_forecast(prices, months=6, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """Fit a SARIMA model on month-end prices and forecast the next months.

    The daily series is reduced to month-end closes first, so the seasonal period of 12 means one year.
    Returns the fitted model and a table with the forecast and its 95% interval.
    """
    monthly = prices.resample('ME').last()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # start-value warnings from statsmodels, not relevant here
        model = sm.tsa.SARIMAX(monthly, order=order, seasonal_order=seasonal_order).fit(disp=False)
    fc = model.get_forecast(steps=months)
    ci = fc.conf_int(alpha=0.05)
    table = pd.DataFrame({'forecast': fc.predicted_mean, 'lower_95': ci.iloc[:, 0], 'upper_95': ci.iloc[:, 1]})
    return model, monthly, table


def monte_carlo_simulation(S0, mu_daily, sigma_daily, days=182, n_paths=1000, seed=42):
    """Simulate future price paths as a geometric Brownian motion, one step per calendar day.

    mu_daily and sigma_daily are the mean and standard deviation of daily LOG returns, so each simulated
    day simply draws a log return from N(mu_daily, sigma_daily) - no drift correction needed.
    182 days = half a year, because ETH trades every day.
    Returns an array of shape (n_paths, days + 1); column 0 is today's price.
    """
    rng = np.random.default_rng(seed)
    log_steps = rng.normal(mu_daily, sigma_daily, size=(n_paths, days))
    paths = S0 * np.exp(np.cumsum(log_steps, axis=1))
    return np.hstack([np.full((n_paths, 1), S0), paths])


def summarise_paths(paths, S0):
    """Key numbers of the simulated end prices."""
    end = paths[:, -1]
    return {
        'mean': end.mean(),
        'median': np.median(end),
        'p5': np.percentile(end, 5),
        'p95': np.percentile(end, 95),
        'prob_below_start': (end < S0).mean(),
    }
