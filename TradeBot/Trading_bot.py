# Test of Trading Robot
### Comparing Equity Market and FOREX market in Average Daily Volume to check which has best liquidity
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.trades as trades_endpoints
from datetime import datetime
import seaborn as sns
from itertools import combinations
import os
from datetime import datetime, timedelta



#####################################
##################################### Oanda API to get the EURUSD spread
access_token = 'cffbb5e578fd005009b83c22258c47e9-a57b606d23808ba5f31ae338db2d2af4'
account_id = '101-004-28614224-001'
client = API(access_token)


# 1. Creating and testing signal function
from apscheduler.schedulers.blocking import BlockingScheduler
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import MarketOrderRequest
from oanda_candles import Pair, Gran, CandleClient
from oandapyV20.contrib.requests import TakeProfitDetails, StopLossDetails
access_token = 'cffbb5e578fd005009b83c22258c47e9-a57b606d23808ba5f31ae338db2d2af4'
######################################
######################################

############################## Get candle function #####################################################
########################################################################################################
# Creating function to get candles required for strategies, this will allow me to get candles live
def get_candles(n):
    client = CandleClient(access_token, real=False)
    collector = client.get_collector(Pair.EUR_CHF, Gran.M1) # Choose pair and M1 = 1 min and M15 = 15min time frame 
    candles = collector.grab(n)
    return candles
# test get candle
candle_test = get_candles(10)

# test to see if it works
for candle in candle_test:
    print(float(str(candle.bid.c))<1) # have to convert to string then to float to be able to use as numerical value
########################################################################################################
########################################################################################################


######################################## MAV signal #####################################################
### Moving Average Signal Generator Function
# We will need two Rolling Windows that computes the Averages usually 25 and 50 are chosen

def rolling_25(df, start, end):
    start = int(start)
    end = int(end)
    

    mean_rolling = df['Close'].rolling(window=25, min_periods=1).mean()

    averages_df = mean_rolling[start:end+1]

    return averages_df

# No we need a rolling window on the 50 last days

def rolling_50(df, start, end):
    start = int(start)
    end = int(end)

    mean_rolling = df['Close'].rolling(window=50, min_periods=1).mean()

    averages_df = mean_rolling[start:end+1]

    return averages_df
MA_candles = get_candles(200)


def MA_signal_generator(df, start, end):
    # Generate the rolling averages for the given start and end window
    short_avg = rolling_25(df, start, end)
    long_avg = rolling_50(df, start, end)
    
    signals = pd.Series(0, index=short_avg.index)
    
    # Looping to create signals over the change in average
    for i in short_avg.index:
        if short_avg.loc[i] > long_avg.loc[i]:
            signals.loc[i] = 1  # Buy signal
        elif short_avg.loc[i] < long_avg.loc[i]:
            signals.loc[i] = 2  # Short signal
        # Else, it will remain 0 as initialized
    
    return signals







from oandapyV20.endpoints import orders, trades
from oandapyV20.contrib.requests import MarketOrderRequest




# 2. Connect to market and execute trades

# get candles GBP_USD
def get_candles_GBP(n):
    client = CandleClient(access_token, real=False)
    collector = client.get_collector(Pair.GBP_USD, Gran.M1) # Choose pair and M1 = 1min time frame 
    candles = collector.grab(n)
    return candles



###########################################################################################################################
#################################### MAV FINAL START############################################################################
###########################################################################################################################

def trading_MAV():
    MAV_candles = get_candles_GBP(200)
    MAV_df_close = pd.DataFrame({'Close': [float(str(candle.bid.c)) for candle in MAV_candles]})

    # Signal generation
    signals = MA_signal_generator(MAV_df_close, 50, 200)
    signal = signals.iloc[-1]

    # Initialize DataFrame to track trades
    trade_log_MAV = pd.DataFrame(columns=['Timestamp', 'Trade Type', 'Instrument', 'Units', 'Entry Price', 'Exit Price', 'Profit/Loss', 'Signal'])

    # Access account
    accountID = '101-004-28614224-001'
    client = API(access_token)

    # Retrieve and store entry prices when opening trades
    current_prices = MAV_df_close['Close'].iloc[-1]
    
    #signal = 1

    # Check open positions and close if necessary
    open_positions_response = client.request(trades_endpoints.OpenTrades(accountID))
    for trade in open_positions_response['trades']:
        if trade['instrument'] == 'GBP_USD':

            current_position_type = 'long' if int(trade['currentUnits']) > 0 else 'short'
            current_position_id = trade['id']
            entry_price = float(trade['price'])

            if (current_position_type == 'long' and signal == 2) or (current_position_type == 'short' and signal == 1):
                close_trade = trades_endpoints.TradeClose(accountID, tradeID=current_position_id)
                client.request(close_trade)
                exit_price = current_prices  
                profit_loss = (exit_price - entry_price) * int(trade['currentUnits']) if current_position_type == 'long' else (entry_price - exit_price) * abs(int(trade['currentUnits']))
                new_row = pd.DataFrame({
                    'Timestamp': [datetime.now()],
                    'Trade Type': ['Close'],
                    'Instrument': ['GBP_USD'],
                   'Units': [int(trade['currentUnits'])],
                    'Entry Price': [entry_price],
                    'Exit Price': [exit_price],
                    'Profit/Loss': [profit_loss],
                    'Signal': [signal]
                })
                trade_log_MAV = pd.concat([trade_log_MAV, new_row], ignore_index=True)
                print('MAV: Closed position due to signal change. Profit/Loss:', profit_loss)

    # Open new trade based on signal
    if signal in [1, 2]:  # Check if there's a signal to trade
        units = 1000 if signal == 1 else -1000
        trade_type = 'Buy' if signal == 1 else 'Sell'
        mo = MarketOrderRequest(instrument='GBP_USD', units=units)
        response = client.request(orders.OrderCreate(accountID, data=mo.data))
        entry_price = current_prices  # Assuming immediate execution at current price
        new_row = pd.DataFrame({
            'Timestamp': [datetime.now()], 
            'Trade Type': [trade_type], 
            'Instrument': ['GBP_USD'], 
            'Units': [units], 
            'Entry Price': [entry_price], 
            'Exit Price': [None], 
            'Profit/Loss': [0],
            'Signal': [signal]
        })
        trade_log_MAV = pd.concat([trade_log_MAV, new_row], ignore_index=True)
        print(f"MAV Opened new {trade_type.lower()} position: GBP_USD")

    print('####################### MAV #######################')
    print(trade_log_MAV)
    print('####################### MAV #######################')

    filename = 'trade_log_MAV.csv'
    if os.path.exists(filename):
        trade_log_MAV.to_csv(filename, mode='a', header=False, index=False)
    else:
        trade_log_MAV.to_csv(filename, mode='w', header=True, index=False)


    return trade_log_MAV
###########################################################################################################################
#################################### MAV FINAL END############################################################################
###########################################################################################################################

print('CAREFUL: We are waiting for a trade, it happens every 15 minutes + 1. Thank you for your patience.')

# I just test if the order is passed: first attempt-> passed but got canceled, market are closed? --> market were indeed closed
#trading_COINT() #remove '#' to test



def analyze_trades():
    # Load the trade log
    trade_log_MAV = pd.read_csv('trade_log_MAV.csv')

    

    # Example analysis: Calculate total profit/loss
    total_profit_loss_MAV = trade_log_MAV['Profit/Loss'].sum()
    print(f"Total Profit/Loss MAV: {total_profit_loss_MAV}")

# Scheduler to put updated data and trades every 15min
scheduler = BlockingScheduler()

scheduler.add_job(
    analyze_trades,
    'cron',
    day_of_week='mon-fri',
    hour='23',  # For example, analyze trades at the end of each trading day
    minute='59',
    #start_date='2024-01-12 23:59:00',
    timezone='America/Chicago'
)


# Schedule trading_MR to execute periodically
scheduler.add_job(
    trading_MAV,
    'cron',
    day_of_week='mon-fri',
    hour='0-23',
    minute='*', # * is every minute, can change it easily to 15 for example 1,16,31,46
    #start_date='2024-01-12 12:00:00',
    timezone='America/Chicago'
)


# Start the scheduler to begin executing
scheduler.start()