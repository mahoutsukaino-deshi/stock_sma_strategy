import os
import datetime
import pandas as pd
import pandas_datareader
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

START_DATE = datetime.date(2011, 1, 1)
END_DATE = datetime.date(2020, 12, 31)
TICKER = '^N225'
INIT_CASH = 1000000


def get_stock(ticker, start_date, end_date):
    '''
    get stock data from Yahoo Finance
    '''
    dirname = 'data'
    os.makedirs(dirname, exist_ok=True)
    fname = f'{dirname}/{ticker}.pkl'
    df_stock = pd.DataFrame()
    if os.path.exists(fname):
        df_stock = pd.read_pickle(fname)
        start_date = df_stock.index.max() + datetime.timedelta(days=1)
    if end_date > start_date:
        df = pandas_datareader.data.DataReader(
            ticker, 'yahoo', start_date, end_date)
        df_stock = pd.concat([df_stock, df[~df.index.isin(df_stock.index)]])
        df_stock.to_pickle(fname)
    return df_stock


class SmaCross(Strategy):
    '''
    SMS Cross Strategy
    '''
    long_term = 75
    short_term = 25

    def init(self):
        close = self.data['Close']
        self.long_sma = self.I(talib.SMA, close, self.long_term)
        self.short_sma = self.I(talib.SMA, close, self.short_term)

    def next(self):
        if crossover(self.short_sma, self.long_sma):
            self.buy()
        elif crossover(self.long_sma, self.short_sma):
            self.sell()


def main():
    df = get_stock(TICKER, START_DATE, END_DATE)

    bt = Backtest(
        df,
        SmaCross,
        cash=INIT_CASH,
        trade_on_close=False,
        exclusive_orders=True
    )

    output = bt.run()
    print(output)
    bt.plot()


if __name__ == '__main__':
    main()
