"""Charts. Each function draws one figure, saves it if a path is given, and returns it."""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update({
    'font.size': 10.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9A9A9A',
    'axes.labelcolor': '#444444', 'xtick.color': '#555555', 'ytick.color': '#555555',
    'savefig.dpi': 200, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
    'text.parse_math': False,                    # '$' in labels is a dollar sign, not maths
})
BLUE, GREY, LIGHT, DARK, INK = '#1F4E9A', '#B9BDC5', '#DDE1E7', '#3A3A3A', '#222222'
SOURCE = 'Data: Coin Metrics daily ETH reference rate (USD)'
usd = mticker.FuncFormatter(lambda v, _: f'${v:,.0f}')
pct = mticker.PercentFormatter(1.0, decimals=0)


def _title(ax, main, sub):
    ax.set_title(main, loc='left', fontsize=13, fontweight='bold', color=INK, pad=24)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5, color='#666666')


def _finish(fig, path, source=SOURCE):
    fig.text(0.01, -0.03, source, fontsize=8, color='#888888')
    if path:
        fig.savefig(path)
    return fig


def plot_price(prices, path=None):
    """Price history on a log scale, with the two big crashes marked (peak to trough)."""
    crashes = [(prices.loc[:'2018-06'].idxmax(), prices.loc['2018':'2019'].idxmin()),
               (prices.loc['2021'].idxmax(), prices.loc['2022'].idxmin())]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(prices.index, prices.values, color=BLUE, lw=1.3)
    ax.set_yscale('log')
    ax.set_yticks([100, 300, 1000, 3000])
    ax.yaxis.set_major_formatter(usd)
    falls = []
    for peak, trough in crashes:
        fall = prices[trough] / prices[peak] - 1
        falls.append(fall)
        ax.annotate('', xy=(trough, prices[trough]), xytext=(peak, prices[peak]),
                    arrowprops=dict(arrowstyle='->', color=DARK, lw=1, connectionstyle='arc3,rad=-0.25'))
        ax.text(peak + (trough - peak) / 2, prices[peak] * 1.05, f'{fall:.0%}', ha='center',
                fontsize=10, color=DARK, fontweight='bold')
    ax.grid(axis='y', color='#EEEEEE')
    lo, hi = sorted(-f for f in falls)
    _title(ax, f'ETH went from ${prices.iloc[0]:,.0f} to ${prices.iloc[-1]:,.0f}, with two crashes of {lo:.0%}-{hi:.0%}',
           f'Daily price in USD, log scale, {prices.index[0]:%d %b %Y} - {prices.index[-1]:%d %b %Y}')
    return _finish(fig, path)


def plot_return_distribution(log_ret, stats_dict, path=None):
    """Histogram of daily log returns against a normal curve, plus a Q-Q plot."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={'width_ratios': [1.3, 1]})
    bins = np.linspace(-0.3, 0.3, 81)
    a1.hist(log_ret.clip(-0.3, 0.3), bins=bins, density=True, color=GREY, edgecolor='white', lw=0.4)
    x = np.linspace(-0.3, 0.3, 400)
    a1.plot(x, stats.norm.pdf(x, log_ret.mean(), log_ret.std()), color=BLUE, lw=1.8,
            label='Normal distribution, same mean and volatility')
    a1.set_yticks([])
    a1.spines['left'].set_visible(False)
    a1.xaxis.set_major_formatter(pct)
    a1.legend(frameon=False, fontsize=9, loc='upper left')
    worst = stats_dict['worst_day']
    a1.text(0.98, 0.55, f"Excess kurtosis {stats_dict['excess_kurtosis']:.1f}\nSkew {stats_dict['skew']:.2f}\n"
                        f"Worst day {np.exp(log_ret[worst]) - 1:.0%}\n({worst:%d %b %Y})",
            transform=a1.transAxes, ha='right', fontsize=9.5, color=DARK)
    (osm, osr), (slope, icpt, _) = stats.probplot(log_ret, dist='norm')
    a2.scatter(osm, osr, s=6, color=GREY)
    a2.plot(osm, slope * osm + icpt, color=BLUE, lw=1.5)
    a2.set_xlabel('Normal quantiles')
    a2.set_ylabel('Observed daily log return')
    a2.yaxis.set_major_formatter(pct)
    a2.grid(color='#F0F0F0')
    _title(a1, 'Daily returns have much fatter tails than a normal distribution',
           'Left: histogram of daily log returns (clipped at ±30%). Right: Q-Q plot against the normal')
    return _finish(fig, path)


def plot_autocorrelation(acf_ret, acf_abs, band, path=None):
    """Autocorrelation of returns next to autocorrelation of absolute returns."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, vals, name, colour in ((axes[0], acf_ret, 'Returns', GREY),
                                   (axes[1], acf_abs, 'Absolute returns (size of the move)', BLUE)):
        ax.axhspan(-band, band, color=LIGHT, lw=0)
        ax.vlines(vals.index, 0, vals.values, color=colour, lw=2.2)
        ax.axhline(0, color='#9A9A9A', lw=0.8)
        ax.set_xlabel('Lag (days)')
        ax.text(0.98, 0.92, name, transform=ax.transAxes, ha='right', fontsize=10, color=DARK, fontweight='bold')
    axes[0].set_ylabel('Autocorrelation')
    axes[0].set_ylim(-0.08, 0.2)
    _title(axes[0], 'Returns are close to unpredictable, the size of the moves is not',
           'Autocorrelation of daily log returns, lags 1-30. Shaded band = 95% bound for "no autocorrelation"')
    return _finish(fig, path)


def plot_decomposition(prices, dec, path=None):
    """Price with trend, seasonal factor and residual of the multiplicative decomposition."""
    seas = dec.seasonal
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={'height_ratios': [1.6, 1, 1]})
    axes[0].plot(prices.index, prices.values, color=GREY, lw=0.9, label='Price')
    axes[0].plot(dec.trend.index, dec.trend.values, color=BLUE, lw=1.5, label='Trend (30-day)')
    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    axes[0].yaxis.set_major_formatter(usd)
    axes[1].plot(seas.index, (seas - 1).values, color=BLUE, lw=0.8)
    axes[1].set_ylim(-0.06, 0.06)
    axes[1].set_ylabel('Seasonal')
    axes[2].plot(dec.resid.index, (dec.resid - 1).values, color=GREY, lw=0.6)
    axes[2].set_ylim(-0.4, 0.4)
    axes[2].set_ylabel('Residual')
    for ax in axes[1:]:
        ax.yaxis.set_major_formatter(pct)
    for ax in axes:
        ax.grid(axis='y', color='#F0F0F0')
    _title(axes[0], f'The 30-day seasonal pattern moves the price by only {seas.min() - 1:+.1%} to {seas.max() - 1:+.1%}',
           'Multiplicative decomposition of the daily price (period 30 days). Seasonal and residual as % deviation')
    return _finish(fig, path)


def plot_sarima(monthly, table, path=None, history_from='2022-01'):
    """Recent monthly closes, the SARIMA forecast and its 95% interval."""
    hist = monthly.loc[history_from:]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(hist.index, hist.values, color=GREY, lw=1.6, marker='o', ms=3, label='Monthly close')
    ax.fill_between(table.index, table['lower_95'], table['upper_95'], color=BLUE, alpha=0.12, lw=0,
                    label='95% forecast interval')
    ax.plot([hist.index[-1]] + list(table.index), [hist.iloc[-1]] + list(table['forecast']), color=BLUE,
            lw=2, marker='o', ms=4, label='SARIMA forecast')
    ax.axvline(hist.index[-1], color='#CCCCCC', lw=1)
    ax.yaxis.set_major_formatter(usd)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.grid(axis='y', color='#EEEEEE')
    lo, hi = (round(v, -2) for v in (table['forecast'].min(), table['forecast'].max()))
    _title(ax, f'SARIMA expects ETH to stay around ${lo:,.0f}-{hi:,.0f} over the next six months',
           f'Month-end prices; SARIMA(1,1,1)(1,1,1,12) fitted on {monthly.index[0]:%b %Y} - {monthly.index[-1]:%b %Y}')
    return _finish(fig, path)


def plot_monte_carlo(paths, start_date, summary, mu_annual, vol_annual, path=None, cap=15000):
    """Percentile bands of the simulated paths and the distribution of the end prices."""
    days = pd.date_range(start_date, periods=paths.shape[1], freq='D')
    q = lambda p: np.percentile(paths, p, axis=0)
    end = paths[:, -1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={'width_ratios': [2.2, 1]}, sharey=True)
    a1.fill_between(days, q(5), q(95), color=BLUE, alpha=0.10, lw=0, label='5-95% of simulated paths')
    a1.fill_between(days, q(25), q(75), color=BLUE, alpha=0.22, lw=0, label='25-75%')
    a1.plot(days, q(50), color=BLUE, lw=1.8, label='Median path')
    a1.xaxis.set_major_locator(mdates.MonthLocator())
    a1.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    a1.yaxis.set_major_formatter(usd)
    a1.legend(frameon=False, fontsize=9, loc='upper left')
    a1.grid(axis='y', color='#EEEEEE')
    a1.set_ylim(0, cap)
    a2.hist(end[end <= cap], bins=np.linspace(0, cap, 51), orientation='horizontal', color=GREY, edgecolor='white', lw=0.4)
    box = dict(facecolor='white', edgecolor='none', pad=1.5)
    x_txt = a2.get_xlim()[1] * 0.97
    a2.axhline(summary['median'], color=BLUE, lw=1.5)
    a2.text(x_txt, summary['median'] + 250, f"Median ${summary['median']:,.0f}", ha='right', fontsize=9, color=BLUE, bbox=box)
    a2.axhline(paths[0, 0], color=DARK, lw=1, ls='--')
    a2.text(x_txt, paths[0, 0] - 300, f'Start ${paths[0, 0]:,.0f}\n({summary["prob_below_start"]:.0%} of paths end lower)',
            ha='right', va='top', fontsize=9, color=DARK, bbox=box)
    a2.text(x_txt, cap * 0.96, f'{int((end > cap).sum())} paths above ${cap:,.0f}\nnot shown', ha='right', va='top',
            fontsize=8.5, color='#777777')
    a2.set_xticks([])
    a2.spines['bottom'].set_visible(False)
    a2.set_title(f'Price after {paths.shape[1] - 1} days', fontsize=10, color=DARK, loc='left')
    _title(a1, f"Half the simulated paths end between ${np.percentile(end, 25):,.0f} and ${np.percentile(end, 75):,.0f}",
           f'1,000 paths, geometric Brownian motion, drift {mu_annual:.0%} and volatility {vol_annual:.0%} a year, '
           f'{paths.shape[1] - 1} daily steps')
    return _finish(fig, path)
