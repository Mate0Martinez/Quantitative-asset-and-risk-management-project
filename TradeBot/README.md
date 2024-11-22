# TradingBot
A trading bot that uses three strategies: mean-reverting property of stationary time series, cointegration and moving average crossover.

To use the bot you simply have to clone the repository and to run the Trading_bot.py file. It will take decisions every 15 minutes starting one (1, 16, 31, 46)

```sh
git clone https://github.com/Shpetim005/TradingBot.git
cd TradingBot
conda create --name TradingBot python=3.9
conda activate TradingBot
pip install -r requirements.txt
python Trading_bot.py
```


To see the results you can look into the trade_log file or directly on Oanda. You will need my information for this.

Please consider that using the bot on week ends will not work, as market are closed.
