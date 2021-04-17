import os
import datetime
import pandas as pd
import pandas_datareader
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, plot_heatmaps

START_DATE = datetime.date(2011, 1, 1)
END_DATE = datetime.date(2020, 12, 31)
TICKER = '^N225'
INIT_CASH = 1000000
MAX_LONG_TERM = 250
TERM_STEP = 1


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

    stats, heatmap = bt.optimize(
        long_term=range(3, MAX_LONG_TERM + 1, TERM_STEP),
        short_term=range(2, MAX_LONG_TERM, TERM_STEP),
        return_heatmap=True,
        constraint=lambda p: p.short_term < p.long_term)

    print(stats)
    bt.plot()
    plot_heatmaps(heatmap, agg='mean', plot_width=2048, filename='heatmap')

    sns.set(font='IPAexGothic')

    d = heatmap.reset_index().pivot('short_term', 'long_term', 'SQN')
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(d, square=True, cmap='seismic', center=0, ax=ax)
    ax.set_title('移動平均期間の組み合わせによるSQN')
    ax.invert_yaxis()
    ax.grid()
    plt.savefig('strategy3.png')
    plt.show()


if __name__ == '__main__':
    main()
