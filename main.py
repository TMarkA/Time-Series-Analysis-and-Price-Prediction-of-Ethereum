"""Runs the whole analysis: loads the data, prints the results and saves all charts as PNG files in this folder."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

from analysis import (DAYS_PER_YEAR, autocorrelation, decompose, descriptive_statistics, month_of_year_test,
                      normality_test, stationarity_test)
from data_loader import load_prices, log_returns
from forecasting import monte_carlo_simulation, sarima_forecast, summarise_paths
from visualization import (plot_autocorrelation, plot_decomposition, plot_monte_carlo, plot_price,
                           plot_return_distribution, plot_sarima)

# --- settings -----------------------------------------------------------------------------------------------
START, END = '2017-11-09', '2024-06-18'
HORIZON_DAYS = 182              # six months; ETH trades every calendar day
N_PATHS = 1000
OUT = Path('.')                  # charts are saved next to the code


def main():
    prices = load_prices(START, END)
    lr = log_returns(prices)
    print(f'{len(prices)} daily prices, {prices.index[0]:%d.%m.%Y} - {prices.index[-1]:%d.%m.%Y}: '
          f'${prices.iloc[0]:,.0f} -> ${prices.iloc[-1]:,.0f}')

    # 1 Distribution
    st = descriptive_statistics(lr)
    jb = normality_test(lr)
    print(f"Daily log return: mean {st['mean_daily']:.3%}, std {st['std_daily']:.2%} "
          f"(annual: {st['mean_annual']:.0%} and {st['vol_annual']:.0%})")
    print(f"Skew {st['skew']:.2f}, excess kurtosis {st['excess_kurtosis']:.1f}, Jarque-Bera p = {jb['p_value']:.1e}")
    plot_price(prices, OUT / 'fig1_price.png')
    plot_return_distribution(lr, st, OUT / 'fig2_returns.png')

    # 2 Stationarity and autocorrelation
    adf_p, adf_r = stationarity_test(prices), stationarity_test(lr)
    print(f"ADF price: {adf_p['adf_statistic']:.2f} (p = {adf_p['p_value']:.2f}); "
          f"ADF returns: {adf_r['adf_statistic']:.2f} (p = {adf_r['p_value']:.1e})")
    acf_ret, acf_abs, band = autocorrelation(lr)
    print(f'Return lags outside the 95% band: {list(acf_ret[acf_ret.abs() > band].index)}; '
          f'ACF of |returns| at lag 1: {acf_abs[1]:.2f}, at lag 30: {acf_abs[30]:.2f}')
    plot_autocorrelation(acf_ret, acf_abs, band, OUT / 'fig3_autocorrelation.png')

    # 3 Seasonality
    dec = decompose(prices)
    kw = month_of_year_test(lr)
    print(f'Seasonal factor (30 days): {dec.seasonal.min() - 1:+.1%} to {dec.seasonal.max() - 1:+.1%}; '
          f"Kruskal-Wallis across calendar months p = {kw['p_value']:.2f}")
    plot_decomposition(prices, dec, OUT / 'fig4_decomposition.png')

    # 4 SARIMA
    model, monthly, table = sarima_forecast(prices)
    print('\nSARIMA(1,1,1)(1,1,1,12) on month-end prices, forecast:')
    print(table.round(0).to_string())
    plot_sarima(monthly, table, OUT / 'fig5_sarima.png')

    # 5 Monte Carlo
    paths = monte_carlo_simulation(prices.iloc[-1], st['mean_daily'], st['std_daily'], days=HORIZON_DAYS, n_paths=N_PATHS)
    sm = summarise_paths(paths, prices.iloc[-1])
    print(f"\nMonte Carlo ({N_PATHS} paths, {HORIZON_DAYS} days): mean ${sm['mean']:,.0f}, median ${sm['median']:,.0f}, "
          f"5-95% ${sm['p5']:,.0f} - ${sm['p95']:,.0f}, {sm['prob_below_start']:.0%} of paths end below today")
    plot_monte_carlo(paths, prices.index[-1], sm, st['mean_annual'], st['vol_annual'], OUT / 'fig6_monte_carlo.png')
    print(f'\nCharts saved to {OUT.resolve()}')


if __name__ == '__main__':
    main()
