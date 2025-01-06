from data_loader import download_data, calculate_daily_returns
from analysis import descriptive_statistics, stationarity_test, seasonality_test
from forecasting import sarima_forecast, monte_carlo_simulation
from visualization import plot_prices, plot_log_returns, seasonal_decomposition, autocorrelation_plot

if __name__ == '__main__':
    # Parameters
    ticker = 'ETH-USD'
    start_date = '2017-11-09'
    end_date = '2024-06-19'
    
    # Data Loading
    adj_close = download_data(ticker, start_date, end_date)
    daily_returns = calculate_daily_returns(adj_close)
    
    # Visualization
    plot_prices(adj_close, ticker)
    plot_log_returns(adj_close)
    seasonal_decomposition(adj_close)
    autocorrelation_plot(daily_returns)
    
    # Analysis
    mean, std = descriptive_statistics(daily_returns)
    stationarity = stationarity_test(daily_returns)
    seasonality = seasonality_test(adj_close)
    
    # Forecasting
    sarima_forecast(adj_close)
    monte_carlo_simulation(adj_close.iloc[-1], mean * 252, std * np.sqrt(252), 0.5, 126, 1000)
