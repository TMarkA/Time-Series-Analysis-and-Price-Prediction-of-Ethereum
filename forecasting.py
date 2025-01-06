import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def sarima_forecast(adj_close, periods=6):
    """
    Perform SARIMA forecasting.
    """
    weekly_data = adj_close.resample('M').last().dropna()
    weekly_data = weekly_data.asfreq('M')
    model = sm.tsa.SARIMAX(weekly_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    forecast = results.get_forecast(steps=periods)
    forecast_mean = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_data, label='Observed')
    plt.plot(forecast_mean.index, forecast_mean.values, label='Forecast', color='red')
    plt.fill_between(forecast_conf_int.index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3)
    plt.title('SARIMA Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    return forecast_mean

def monte_carlo_simulation(S0, mu, sigma, T, N, num_simulations):
    """
    Perform Monte Carlo simulation for future price paths.
    """
    dt = T / N
    S = np.zeros((num_simulations, N + 1))
    S[:, 0] = S0
    for t in range(1, N + 1):
        S[:, t] = S[:, t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(num_simulations))
    plt.figure(figsize=(10, 6))
    for i in range(num_simulations):
        plt.plot(S[i, :], lw=0.8, alpha=0.6)
    plt.title('Monte Carlo Simulation')
    plt.xlabel('Time (years)')
    plt.ylabel('Price')
    plt.show()
    return S
