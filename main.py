import numpy as np
import pandas as pd
from pypfopt import expected_returns
from pypfopt import EfficientFrontier, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from matplotlib import pyplot as plt

#reading the data from csv file
df = pd.read_csv('daily_prices.csv')

#Setting Date as index
df['Date'] = pd.to_datetime(df['Date'],format='%d-%m-%Y')
df = df.set_index(df['Date'].values)

#removing the Date Column
df.drop(columns=['Date'], axis = 1, inplace=True)
print(df)

#Representing or Visually showing the portfolio
plt.style.use('fivethirtyeight')
title = 'Share prices of Stocks in Portfolio (Historic)'

#getting the tickers for graph
my_stocks = df

#creating and plotting the graph
for a in my_stocks.columns.values:
    plt.plot(my_stocks[a], label=a)

plt.title(title)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Prices in INR', fontsize=18)
plt.legend(my_stocks.columns.values, loc='upper left')
plt.show()

# show the returns
returns = df.pct_change() 
print(returns)

# calculating the mean and covariance matrix
mean = expected_returns.mean_historical_return(df)
risk = risk_models.sample_cov(df)
cov_matrix_annual = risk
print(cov_matrix_annual)

# optimize for the maximum Sharpe Ratio
e_f = EfficientFrontier(mean, risk)
weights = e_f.max_sharpe()
clean_weights = e_f.clean_weights()
print(clean_weights)
e_f.portfolio_performance(verbose=True)

# calculate the variance
temp1 = weights.values()
weights_list = list(temp1)
port_variance = np.dot(np.transpose(weights_list), np.dot(cov_matrix_annual, weights_list))
percent_var = str(round(port_variance,2)*100) + '%'
print("Annual Variance: ", percent_var)

#Get the discrete allocation of shares per stock
portfolio_val = 300000
latest_prices = get_latest_prices(df)
weights = clean_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value = portfolio_val)
allocation, leftover = da.lp_portfolio()
print("Balance Funds: Rs.", round(leftover,2))

#Get the discrete allocation values
discrete_allocation_list = []
for symbol in allocation:
    discrete_allocation_list.append(allocation.get(symbol))

#create a dataframe for the portfolio
portfolio_df = pd.DataFrame(columns= ['Ticker', 'Allocation '+ str(portfolio_val)])

portfolio_df['Ticker'] = allocation
portfolio_df['Allocation '+ str(portfolio_val)] = discrete_allocation_list

#print the portfolio
print(portfolio_df)