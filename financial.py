# -*- coding: utf-8 -*-

import yfinance as yf
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import cvxopt as opt
from cvxopt import blas, solvers
import seaborn as sns

ticker_list = ['PYPL', 'VOD', 'SQ', 'DHER.DE', 'SPCE', 'LAC']
data = yf.download(tickers=ticker_list, period='YTD', group_by='ticker')
data2 = yf.download(tickers=ticker_list, period="5Y", group_by='ticker')

x = data.index
x2 = data2.index


fig0 = plt.figure(figsize=[30,20])
ax0 = plt.subplot()
plt.plot(x2, data2.PYPL['Adj Close'])
plt.plot(x2, data2.VOD['Adj Close'])
plt.plot(x2, data2.SQ['Adj Close'])
plt.plot(x2, data2['DHER.DE']['Adj Close'])
plt.plot(x2, data2.SPCE['Adj Close'])
plt.plot(x2, data2.LAC['Adj Close'])
plt.title("Chosen stock over past 5 years", fontsize = 16)
plt.xlabel('Year', fontsize = 14)
plt.ylabel('Value', fontsize = 14)
plt.show()

fig = plt.figure(figsize=[70, 40])
ax = plt.subplot(2,3,1)
plt.grid(visible=True, linestyle=':')
plt.plot(x, data.PYPL['Adj Close'], color='C0')
ax.set_title("Paypal" ,fontsize = 16)
ax2 = plt.subplot(2,3,2)
plt.grid(visible=True, linestyle=':')
plt.plot(x, data.SQ['Adj Close'], color='C1')
ax2.set_title('Block' ,fontsize = 16)
ax3 = plt.subplot(2,3,3)
plt.grid(visible=True, linestyle=':')
plt.plot(x, data['DHER.DE']['Adj Close'], color='C2')
ax3.set_title('Delivery Hero' ,fontsize = 16)
ax4 = plt.subplot(2,3,4)
plt.grid(visible=True, linestyle=':')
plt.plot(x, data.SPCE['Adj Close'], color='C3')
ax4.set_title('Virgin Galactic' ,fontsize = 16)
ax5 = plt.subplot(2,3,5)
plt.grid(visible=True, linestyle=':')
plt.plot(x, data.LAC['Adj Close'], color='C4')
ax5.set_title("Lithium Americas Corp." ,fontsize = 16)
ax6 = plt.subplot(2,3,6)
plt.grid(visible=True, linestyle=':')
plt.plot(x, data.VOD['Adj Close'], color='C5')
ax6.set_title("Vodafone" ,fontsize = 16)
fig.tight_layout()
plt.show()

all_returns_paypal = data2.PYPL['Adj Close'].pct_change()
all_returns_vodafone = data2.VOD['Adj Close'].pct_change()
all_returns_block = data2.SQ['Adj Close'].pct_change()
all_returns_dhero = data2['DHER.DE']['Adj Close'].pct_change()
all_returns_virgin = data2.SPCE['Adj Close'].pct_change()
all_returns_lac = data2.LAC['Adj Close'].pct_change()

fig02 = plt.figure(figsize=[20,10])
plt.bar(ticker_list, [all_returns_paypal.mean(), all_returns_vodafone.mean(), all_returns_block.mean(), all_returns_dhero.mean(), all_returns_virgin.mean(), all_returns_lac.mean()])
plt.title("All time average returns" ,fontsize = 16)

returns_paypal = data.PYPL['Adj Close'].pct_change()
returns_vodafone = data.VOD['Adj Close'].pct_change()
returns_block = data.SQ['Adj Close'].pct_change()
returns_dhero = data['DHER.DE']['Adj Close'].pct_change()
returns_virgin = data.SPCE['Adj Close'].pct_change()
returns_lac = data.LAC['Adj Close'].pct_change()

fig2 = plt.figure(figsize = [20,10])
plt.bar(ticker_list, [returns_paypal.mean(), returns_vodafone.mean(), returns_block.mean(), returns_dhero.mean(), returns_virgin.mean(), returns_lac.mean()])
plt.title("Average returns this year" ,fontsize = 16)
plt.show()

df = pd.DataFrame({'Paypal': returns_paypal.fillna(0), 'Vodafone': returns_vodafone.fillna(0), 'Block': returns_block.fillna(0), 'Delivery Hero': returns_dhero.fillna(0), 'Virgin Galactic': returns_virgin.fillna(0), 'Lithium Americas Corp.': returns_lac.fillna(0)})
df2 = pd.DataFrame({'Paypal': all_returns_paypal.fillna(0), 'Vodafone': all_returns_vodafone.fillna(0), 'Block': all_returns_block.fillna(0), 'Delivery Hero': all_returns_dhero.fillna(0), 'Virgin Galactic': all_returns_virgin.fillna(0), 'Lithium Americas Corp.': all_returns_lac.fillna(0)})

variance = df.var()
variance2 = df2.var()

fig_var = plt.figure(figsize=[30,15])
plt.subplot(1,2,1)
plt.bar(variance.index, variance)
plt.title('Variance in last year', fontsize = 16)
plt.subplot(1,2,2)
plt.title('Variance in last 5 years', fontsize = 16)
plt.bar(variance.index, variance2)
plt.show()

def return_portfolios(expected_returns, cov_matrix):
    port_returns = []
    port_volatility = []
    stock_weights = []
    
    selected = (expected_returns.axes)[0]
    
    num_assets = len(selected) 
    num_portfolios = 10000
    
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, expected_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)
    
    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility}
    
    for counter,symbol in enumerate(selected):
        portfolio[symbol +' Weight'] = [Weight[counter] for Weight in stock_weights]
    
    df = pd.DataFrame(portfolio)
    
    column_order = ['Returns', 'Volatility'] + [stock+' Weight' for stock in selected]
    
    df = df[column_order]
   
    return df
  
  
def optimal_portfolio(returns):
    n = returns.shape[1]
    returns = np.transpose(returns)

    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


random_portfolios = return_portfolios(df2.mean(), df2.cov()) 
dd=df2.to_numpy()
single_std = np.sqrt(np.diagonal(df2.cov()))
weights, returns, risks = optimal_portfolio(dd)

print(random_portfolios['Returns'].max())

high_return = random_portfolios[random_portfolios['Returns']==random_portfolios['Returns'].max()]
pd.set_option('display.max_columns', None)
print(high_return)


random_portfolios.plot.scatter(x='Volatility', y='Returns')
plt.plot(risks, returns, 'y-o')
plt.scatter(single_std, df2.mean(), marker='X', s = 220, color = 'green')
plt.show()



sns.heatmap(df2.corr(), cmap='Blues', annot=True, )
sns.set(rc={'figure.figsize':(20,15)})

