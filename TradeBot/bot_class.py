import os
import pandas as pd
import numpy as np
from datetime import datetime
from oandapyV20 import API
from oandapyV20.contrib.requests import MarketOrderRequest
from oandapyV20.endpoints import trades as trades_endpoints
from oandapyV20.endpoints.orders import OrderCreate
from oanda_candles import Pair, Gran, CandleClient
import logging

class TradingBot:
    def __init__(self):
        self.access_token = 'cffbb5e578fd005009b83c22258c47e9-a57b606d23808ba5f31ae338db2d2af4'
        self.account_id = '101-004-28614224-001'
        self.client = API(self.access_token)
        self.trade_log_file = 'trade_log_MAV.csv'

    def get_candles(self, pair, granularity, n):
        client = CandleClient(self.access_token, real=False)
        collector = client.get_collector(pair, granularity)
        candles = collector.grab(n)
        return pd.DataFrame({'Close': [float(str(candle.bid.c)) for candle in candles]})

    def get_candles_GBP(self,n):
        client = CandleClient(self.access_token, real=False)
        collector = client.get_collector(Pair.GBP_USD, Gran.M1) # Choose pair and M1 = 1min time frame 
        candles = collector.grab(n)
        return candles

    def rolling_avg(self, df, window):
        return df['Close'].rolling(window=window, min_periods=1).mean()

    def generate_signals(self, df, start, end):
        short_avg = self.rolling_avg(df, 25)[start:end+1]
        long_avg = self.rolling_avg(df, 50)[start:end+1]
        signals = pd.Series(0, index=short_avg.index)
        
        for i in short_avg.index:
            if short_avg.loc[i] > long_avg.loc[i]:
                signals.loc[i] = 1  # Buy signal
            elif short_avg.loc[i] < long_avg.loc[i]:
                signals.loc[i] = 2  # Sell signal
        
        return signals

    def execute_trade(self, signal, current_price):
        # Close positions based on signals
        open_positions = self.client.request(trades_endpoints.OpenTrades(self.account_id))
        trade_log = pd.DataFrame(columns=['Timestamp', 'Trade Type', 'Instrument', 'Units', 'Entry Price', 'Exit Price', 'Profit/Loss', 'Signal'])

        for trade in open_positions['trades']:
            if trade['instrument'] == 'GBP_USD':
                position_type = 'long' if int(trade['currentUnits']) > 0 else 'short'
                trade_id = trade['id']
                entry_price = float(trade['price'])

                if (position_type == 'long' and signal == 2) or (position_type == 'short' and signal == 1):
                    self.client.request(trades_endpoints.TradeClose(self.account_id, tradeID=trade_id))
                    profit_loss = (current_price - entry_price) * int(trade['currentUnits']) if position_type == 'long' else (entry_price - current_price) * abs(int(trade['currentUnits']))
                    trade_log = trade_log.append({
                        'Timestamp': datetime.now(),
                        'Trade Type': 'Close',
                        'Instrument': 'GBP_USD',
                        'Units': int(trade['currentUnits']),
                        'Entry Price': entry_price,
                        'Exit Price': current_price,
                        'Profit/Loss': profit_loss,
                        'Signal': signal
                    }, ignore_index=True)

        # Open a new trade
        if signal in [1, 2]:
            units = 1000 if signal == 1 else -1000
            trade_type = 'Buy' if signal == 1 else 'Sell'
            mo = MarketOrderRequest(instrument='GBP_USD', units=units)
            self.client.request(OrderCreate(self.account_id, data=mo.data))
            trade_log = trade_log.append({
                'Timestamp': datetime.now(),
                'Trade Type': trade_type,
                'Instrument': 'GBP_USD',
                'Units': units,
                'Entry Price': current_price,
                'Exit Price': None,
                'Profit/Loss': 0,
                'Signal': signal
            }, ignore_index=True)

        # Save to CSV
        if os.path.exists(self.trade_log_file):
            trade_log.to_csv(self.trade_log_file, mode='a', header=False, index=False)
        else:
            trade_log.to_csv(self.trade_log_file, mode='w', header=True, index=False)
    
    def trading_MAV(self):
        logging.info('MAV: Trading logic started')
        MAV_candles = self.get_candles_GBP(200)
        MAV_df_close = pd.DataFrame({'Close': [float(str(candle.bid.c)) for candle in MAV_candles]})
        logging.info('MAV: Retrieved candles')
        # Signal generation
        signals = self.generate_signals(MAV_df_close, 50, 200)
        signal = signals.iloc[-1]

        # Initialize DataFrame to track trades
        trade_log_MAV = pd.DataFrame(columns=['Timestamp', 'Trade Type', 'Instrument', 'Units', 'Entry Price', 'Exit Price', 'Profit/Loss', 'Signal'])

        # Access account
        accountID = '101-004-28614224-001'
        client = API(self.access_token)

        # Retrieve and store entry prices when opening trades
        current_prices = MAV_df_close['Close'].iloc[-1]
        logging.info('MAV: Retrieved current prices')
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
            response = client.request(OrderCreate(accountID, data=mo.data))
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
            logging.info('MAV: Opened new trade')

        print('####################### MAV #######################')
        print(trade_log_MAV)
        print('####################### MR #######################')

        filename = 'trade_log_MAV.csv'
        if os.path.exists(filename):
            trade_log_MAV.to_csv(filename, mode='a', header=False, index=False)
        else:
            trade_log_MAV.to_csv(filename, mode='w', header=True, index=False)


        return trade_log_MAV
    
    def trading_logic(self):
        candles = self.get_candles(Pair.GBP_USD, Gran.M1, 200)
        signals = self.generate_signals(candles, 50, 200)
        signal = signals.iloc[-1]
        current_price = candles['Close'].iloc[-1]
        self.execute_trade(signal, current_price)

    def analyze_trades(self):
        if os.path.exists(self.trade_log_file):
            trade_log = pd.read_csv(self.trade_log_file)
            total_profit_loss = trade_log['Profit/Loss'].sum()
            return total_profit_loss
        else:
            return 0

    def run(self):
        # Start the scheduler if it's not already running
        if not self.scheduler.running:
            self.scheduler.start()
            print("Scheduler started.")
        else:
            print("Scheduler is already running.")
