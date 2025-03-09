import pandas as pd
import yfinance as yf
import numpy as np

# Define the commodities and their symbols
commodities = {
    'Crude_Oil': 'CL=F',      # Crude Oil Futures
    'Gold': 'GC=F',           # Gold Futures
    'Silver': 'SI=F',         # Silver Futures
    'Copper': 'HG=F',         # Copper Futures
    'Aluminum': 'ALI=F',      # Aluminum Futures
    'Natural_Gas': 'NG=F',    # Natural Gas Futures
    'Corn': 'ZC=F',           # Corn Futures
    'Wheat': 'ZW=F',          # Wheat Futures
    'Soybeans': 'ZS=F',       # Soybeans Futures
    'Cotton': 'CT=F'          # Cotton Futures
}

# Fetch SP500 data
sp500 = yf.download('^GSPC', start='2014-12-01', end='2024-12-01', interval='1d')[['Close']]

# Flatten SP500 data if it has a multi-level index
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.droplevel(0)

# Rename the column for clarity
sp500.rename(columns={'Close': 'SP500'}, inplace=True)

# Ensure SP500 data has a valid index
if not isinstance(sp500.index, pd.DatetimeIndex):
    sp500.index = pd.to_datetime(sp500.index)

# Initialize DataFrame with SP500 data
data = sp500.copy()

# Fetch commodities data
for name, symbol in commodities.items():
    commodity_prices = yf.download(symbol, start='2014-12-01', end='2024-12-01', interval='1d')['Close']
    
    # Check if commodity data is valid
    if commodity_prices.empty:
        print(f"Warning: No data retrieved for {name} ({symbol}). Skipping.")
        continue
    
    # Ensure commodity data index is datetime
    if not isinstance(commodity_prices.index, pd.DatetimeIndex):
        commodity_prices.index = pd.to_datetime(commodity_prices.index)
    
    # Add the commodity prices to the DataFrame
    data[name] = commodity_prices

# Drop rows with NaN values (in case of missing data for any commodity)
data.dropna(inplace=True)
data.to_csv('sp500_commodities_daily.csv', index_label='Date')
print("Daily data saved to 'sp500_commodities_daily.csv'")

############################################################################

data = pd.read_csv('sp500_commodities_daily.csv')

data['time'] = np.arange(len(data))

returns = data.iloc[:, 1:-1].pct_change()
signal_columns = [f"sig_{i}" for i in range(len(returns.columns))]
returns.columns = signal_columns

processed_data = pd.DataFrame()
processed_data['time'] = data['time']
processed_data['return'] = returns.iloc[:, 0]
processed_data = pd.concat([processed_data, returns], axis=1)
processed_data.dropna(inplace=True)

processed_data.to_csv('sp500_commodities_daily_post_proc.csv', index=False)
print("Post-processed data saved to 'sp500_commodities_daily_perc_chg.csv'")
