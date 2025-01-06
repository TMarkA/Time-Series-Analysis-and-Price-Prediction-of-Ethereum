# Time-Series-Analysis and Price Prediction
My Seminar Project "Time-Series Analysis and Price Prediction of Ethereum using SARIMA and Monte-Carlo Simulation" that I did in 2023 in Otto von GUericke University Magdeburg 


This project provides tools for any financial asset analysis, including time-series analysis, seasonal decomposition, statistical tests, forecasting with SARIMA model and Monte Carlo simulation.
(Note: different types of assets might require extra steps in data processing stage as crypto is traded 24/7)

##  Structure
- **Data Retrieval:** Download data from Yahoo Finance (can be API or CSV file in your case).
- **Descriptive Statistics:** Calculate mean and volatility of asset returns as the are crucial for further analysis.
- **Normality Testing:** Provide histogramm and Q-Q plot to check for normality.
- **Stationarity Testing:** Perform stationarity testing using Augmenter Dicker Fuller method.
- **Seasonality Analysis:** Decompose time-series into trend, seasonality, and residual noise and depict it.
- **Forecasting:** Predict future prices using SARIMA model for the next 6 months.
- **Simulation:** Model future price paths using Monte Carlo method and compare results with SARIMA results.

### Motivation
My primary objective while doing this seminar paper was to deepen my practical knowledge in operating Python as a tool for analysis and modelling aswell as to learn more about the time-series analysis techniques and such methods like ARIMA and Monte Carlo simulation.

The focus of my analysis was on Ethereum's historical price data (ETH/USD), where I aimed to uncover key statistical properties such as distribution, seasonality, and stationarity. These properties provided the foundation for applying various predictive modeling techniques. Using these insights, I used the SARIMA (Seasonal AutoRegressive Integrated Moving Average) model to generate short-term price forecasts for Ethereum over the next six months. Additionally, I conducted a Monte Carlo simulation to simulate potential future price paths, offering a broader perspective on Ethereum’s price trajectory which allowed me to compare both methods.

Although the initial goal of this paper was to calculate the price of an out-of-the-money call option for Ethereum using the forecasts generated by SARIMA and Monte Carlo simulation, I ultimately decided to omit this part of the analysis before posting here. I suppose that the core "toolkit" for analysis, consisting of time-series analysis, SARIMA modeling, and Monte Carlo simulations, was already complete to be applied to any other financial asset. 
