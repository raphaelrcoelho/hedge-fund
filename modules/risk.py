import numpy as np
import pandas as pd
from statsmodels import regression
import statsmodels.api as sm
from scipy.stats import norm

def ewm_variance(cc_returns, window=75):
    return cc_returns.ewm(span=window, min_periods=window).var().dropna()

def ewm_mean(cc_returns, window=75):
    return cc_returns.ewm(span=window, min_periods=window).mean().dropna()

def ewm_cov(cc_returns, window=75):
    return cc_returns.ewm(span=window, min_periods=window).cov().dropna()

def parametric_value_at_risk(mean, volatility, alpha):
    return np.exp(mean + volatility * norm.ppf(alpha)) -1

def non_parametric_value_at_risk(cc_returns, alpha):
    return np.exp(cc_returns.quantile(alpha, axis='index')) - 1

def variance(cov_matrix):
    return pd.DataFrame(
            [ pd.Series(np.diag(cov_matrix[item]), index=cov_matrix.minor_axis) for item in cov_matrix.items ],
            index=cov_matrix.items)

def get_betas(cov_matrix):
    # mkt_cov = simple_returns.cov().ibov
    # mkt_corr = simple_returns.corr().ibov
    # betas = mkt_cov / simple_returns.ibov.var()

    betas = cov_matrix.ix[:, 'ibov', :] / cov_matrix.ix[:, 'ibov', 'ibov']
    return betas.T

def brownian_motion(sigma, dt, time, I):
    sqrt_delta_sigma = np.sqrt(dt) * sigma
    return np.random.normal(loc=0, scale=sqrt_delta_sigma, size=(time-1, I))

def geometric_brownian_motion(mu, sigma, wiener_process, dt):
    return (mu - 0.5 * sigma ** 2) * dt + wiener_process

def linreg(x, y):
    # Running the linear regression
    x.name = 'beta'
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()
    x = x.ix[:, 1:]

    return model.get_robustcov_results(cov_type='HC3')

def monte_carlo(close_prices, cc_returns, volatility, time=252):
    S0 = close_prices.ix[-1]  # last price = initial value of simulation
    mu = cc_returns.mean() * 252
    sigma = volatility * 252 ** 0.5

    dt = 1 / 252

    I = 20000

    wiener_process = brownian_motion(sigma, dt, time, I)
    cc_forecast = geometric_brownian_motion(mu, sigma, wiener_process, dt)

    forecast_prices = np.zeros_like(cc_forecast) # array for stock prices
    forecast_prices[0] = S0  # all paths start at initial value

    # geometric brownian motion
    for t in range(1, time - 1):
        forecast_prices[t] = forecast_prices[t-1] * np.exp(cc_forecast[t])

    return forecast_prices

def sharpe_ratio(mean, variance):
    expected_excess = (mean - mean.ibov) * 252
    volatility = (variance ** 0.5) * (252 ** 0.5)
    return expected_excess / volatility

# def treynor_ratio(er, returns, market, rf):
#     return (er - rf) / beta(returns, market)

def tracking_error(cc_returns):
    tr = cc_returns.subtract(cc_returns.ibov, axis='index') ** 2
    tr = tr.sum() / (cc_returns.ibov.size - 1)

    return tr ** 0.5

def information_ratio(mean, tracking_error):
    expected_excess = (mean - mean.ibov) * 252
    tracking_error = tracking_error * (252 ** 0.5)
    return expected_excess / tracking_error
