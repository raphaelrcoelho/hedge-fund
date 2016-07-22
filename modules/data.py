import numpy as np
import pandas as pd

def simple_returns(adj_close_prices):
	sr = adj_close_prices / adj_close_prices.shift(1) - 1
	
	return sr.dropna() 

def premiums(simple_returns, risk_free):
	return simple_returns.subtract(risk_free, axis='index')

def build_portfolio(close_prices, simple_returns, cc_returns, cum_returns, weights):
	close_prices.drop('portfolio', axis=1, inplace=True, errors='ignore')
	
	close_prices['portfolio'] = (cum_returns.ix[:, :-1] * weights).sum(axis=1)
	simple_returns['portfolio'] = (close_prices.portfolio / close_prices.portfolio.shift(1) - 1).dropna()
	cc_returns['portfolio'] = np.log(1 + simple_returns.portfolio)
	cumulative_returns['portfolio'] = close_prices.portfolio / close_prices.portfolio.ix[0]
	
	return close_prices, simple_returns, cc_returns, cum_returns

def prepare_backtest(adj_close_prices, close_prices, date_index):
	acp = adj_close_prices.ix[date_index, :]
	cp = close_prices.ix[date_index, :]
	sr = simple_returns(acp)
	cc = np.log(1 + sr)
	cr = acp / acp.ix[0]
	
	return acp, cp, sr, cc, cr
