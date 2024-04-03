import warnings
warnings.filterwarnings('ignore')
import backtrader as bt
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pyfolio as pf
import backtrader.analyzers as btanalyzers
import matplotlib.pyplot as plt
from Li_Yunfan_Dataset_Processing import *
from Li_Yunfan_Model import *

class ExtendedPandasData(bt.feeds.PandasData):
    lines = ('RET_1d', 'RET_1w', 'RET_1m', 'RET_3m', 'RET_6m', 'RET_12m',
       'Volatility_1w', 'Volatility_1m', 'Volatility_3m', 'Volatility_6m',
       'Volatility_12m', 'Volume_1d',  'Volume_1w', 'Volume_1m', 'Volume_3m', 'Volume_6m', 
       'Volume_12m','Volume_Volatility_1w','Volume_Volatility_1m', 'Volume_Volatility_3m', 
       'Volume_Volatility_6m','Volume_Volatility_12m', 'SMA', 'BB_UP', 'BB_DOWN', 'macd_line',
       'macd_signal_line', 'macd_histogram', 'ADX', 'di_positive',
       'di_negative', 'RSI', 'USREC', 'DFF', 'UMCSENT', 'UNRATE')

    params = (('RET_1d', -1),
              ('RET_1w', -1),
              ('RET_1m', -1),
              ('RET_3m', -1),
              ('RET_6m', -1),
              ('RET_12m', -1),
              ('Volatility_1w', -1),
              ('Volatility_1m', -1),
              ('Volatility_3m', -1),
              ('Volatility_6m', -1),
              ('Volatility_12m', -1),
              ('Volume_1d', -1),
              ('Volume_1w', -1),
              ('Volume_1m', -1),
              ('Volume_3m', -1),
              ('Volume_6m', -1),
              ('Volume_12m', -1),
              ('Volume_Volatility_1w', -1),
              ('Volume_Volatility_1m', -1),
              ('Volume_Volatility_3m', -1),
              ('Volume_Volatility_6m', -1),
              ('Volume_Volatility_12m', -1),
              ('SMA', -1),
              ('BB_UP', -1),
              ('BB_DOWN', -1),
              ('macd_line', -1),
              ('macd_signal_line', -1),
              ('macd_histogram', -1),
              ('ADX', -1),
              ('di_positive', -1),
              ('di_negative', -1),
              ('RSI', -1),
              ('USREC', -1),
              ('DFF', -1),
              ('UMCSENT', -1),
              ('UNRATE', -1))
    
class Trading_Strategy(bt.Strategy):
    params = (('model', None), ('method', None))

    def __init__(self):
        """
        Initialize the trading strategy. Requires a model and a method to be provided.

        :param model: The predictive model to be used for trading decisions.
        :param method: The method of prediction, such as 'NNR' (Neural Network Regression), 
                       'NNC' (Neural Network Classification), or other non-neural network methods.
        """
        if self.params.model is None:
            raise ValueError("A model must be provided to the strategy.")
        self.model = self.params.model
        self.method = self.params.method
        self.size = 50

    def Decision_making(self, data, threshold=0):
        """
        Determine buy, sell, or hold decisions based on the given threshold.

        :param data: A list containing predicted values.
        :param threshold: The threshold used to make buy (-1) or sell (1) decisions. Default is 0.

        :return: A list of decisions, where 1 represents 'buy' and -1 represents 'sell'.
        """
        decisions = [1 if RET > threshold else -1 for RET in data]
        return decisions

    def next(self):
        """
        Execute the strategy for the next tick. The strategy logic is executed here.
        """
        # Retrieve current data from the feed
        current_data = {
            'RET_1w': self.datas[0].RET_1w[0],
            'RET_1m': self.datas[0].RET_1m[0],
            'RET_3m': self.datas[0].RET_3m[0],
            'RET_6m': self.datas[0].RET_6m[0],
            'RET_12m': self.datas[0].RET_12m[0],
            'Volatility_1w': self.datas[0].Volatility_1w[0],
            'Volatility_1m': self.datas[0].Volatility_1m[0],
            'Volatility_3m': self.datas[0].Volatility_3m[0],
            'Volatility_6m': self.datas[0].Volatility_6m[0],
            'Volatility_12m': self.datas[0].Volatility_12m[0],
            'Volume_1d': self.datas[0].Volume_1d[0],
            'Volume_1w': self.datas[0].Volume_1w[0],
            'Volume_1m': self.datas[0].Volume_1m[0],
            'Volume_3m': self.datas[0].Volume_3m[0],
            'Volume_6m': self.datas[0].Volume_6m[0],
            'Volume_12m': self.datas[0].Volume_12m[0],
            'Volume_Volatility_1w': self.datas[0].Volume_Volatility_1w[0],
            'Volume_Volatility_1m': self.datas[0].Volume_Volatility_1m[0],
            'Volume_Volatility_3m': self.datas[0].Volume_Volatility_3m[0],
            'Volume_Volatility_6m': self.datas[0].Volume_Volatility_6m[0],
            'Volume_Volatility_12m': self.datas[0].Volume_Volatility_12m[0],
            'SMA': self.datas[0].SMA[0],
            'BB_UP': self.datas[0].BB_UP[0],
            'BB_DOWN': self.datas[0].BB_DOWN[0],
            'macd_line': self.datas[0].macd_line[0],
            'macd_signal_line': self.datas[0].macd_signal_line[0],
            'macd_histogram': self.datas[0].macd_histogram[0],
            'ADX': self.datas[0].ADX[0],
            'di_positive': self.datas[0].di_positive[0],
            'di_negative': self.datas[0].di_negative[0],
            'RSI': self.datas[0].RSI[0],
            'USREC': self.datas[0].USREC[0],
            'DFF': self.datas[0].DFF[0],
            'UMCSENT': self.datas[0].UMCSENT[0],
            'UNRATE': self.datas[0].UNRATE[0],
        }

        if self.method == 'NNR':
            # Neural Network Regression approach
            X_new = pd.DataFrame([current_data])
            X_new_tensor = torch.tensor(X_new.values, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(X_new_tensor)
                decision = self.Decision_making(prediction)
                if decision == [1]:
                    self.buy(size = self.size)
                elif decision == [-1]:
                    self.sell(size = self.size)
        
        elif self.method == 'NNC':
            # Neural Network Classification approach
            X_new = pd.DataFrame([current_data])
            X_new_tensor = torch.tensor(X_new.values, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(X_new_tensor)
                prediction = 2 * prediction - 1  # Adjust prediction for decision making
                decision = self.Decision_making(prediction)
                if decision == [1]:
                    self.buy(size = self.size)
                elif decision == [-1]:
                    self.sell(size = self.size)

        else:
            # Non-Neural Network approaches
            X_new = pd.DataFrame([current_data])
            prediction = self.model.predict(X_new)
            decision = self.Decision_making(prediction)
            if decision == [1]:
                self.buy(size = self.size)
            elif decision == [-1]:
                self.sell(size = self.size)

class WinLossRatioAnalyzer(bt.Analyzer):
    
    def __init__(self):
        self.wins = 0
        self.losses = 0

    def notify_trade(self, trade):
        if trade.isclosed:
            if trade.pnl > 0:
                self.wins += 1
            elif trade.pnl < 0:
                self.losses += 1

    def get_analysis(self):
        winloss_ratio = self.wins / self.losses if self.losses > 0 else 'inf'
        return {'winloss_ratio': winloss_ratio}
    
def BackTrader(model, dataset, model_name, file_folder, method='Non_NN'):
    """
    Execute a backtrading strategy using the specified model and dataset.

    :param model: A pre-trained model used for trading decisions.
    :param dataset: The dataset used for backtesting, typically historical stock data.
    :param model_name: The name of the model, used for naming output files.
    :param method: The method of prediction, default is 'Non_NN' for non-neural network models.
    """
    # Initialize a Cerebro instance for backtesting.
    cerebro = bt.Cerebro()
    # Create a data feed from the provided dataset.
    data = ExtendedPandasData(dataname=dataset)
    cerebro.adddata(data)
    # Add a trading strategy to Cerebro and pass the pre-trained model.
    cerebro.addstrategy(Trading_Strategy, model=model, method=method)
    # Set the initial capital for the trading simulation.
    cerebro.broker.setcash(100000.0)
    # Attach a PyFolio analyzer for performance analysis.
    cerebro.addanalyzer(btanalyzers.PyFolio, _name='pyfolio')
    # Then you add the analyzer to Cerebro before running the backtest:
    cerebro.addanalyzer(WinLossRatioAnalyzer, _name='winloss')
    # Run the backtesting strategy.
    results = cerebro.run()
    # Extract the first strategy's results (assuming a single strategy).
    strategy = results[0]
    # Retrieve performance metrics: returns, positions, transactions, and leverage.
    pyfolio_analyzer = strategy.analyzers.getbyname('pyfolio')
    winloss_ratio = strategy.analyzers.winloss.get_analysis()['winloss_ratio']
    returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()
    
    # Calculate the cumulative returns for the trading strategy
    cumulative_returns_trading = (1 + returns).cumprod()
    # Calculate the cumulative returns for a simple buy-and-hold strategy
    cumulative_returns_buy_hold = (1 + dataset['RET_1d']).cumprod()

    # Plot the cumulative returns of both strategies for comparison
    plt.figure(figsize=(10, 6))
    cumulative_returns_trading.plot(label='Trading Strategy')
    cumulative_returns_buy_hold.plot(label='Buy and Hold Strategy')
    plt.legend()
    plt.title(f'{model_name}: Trading Strategy vs. Buy and Hold Strategy')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    plt.savefig(f'./{file_folder}/comparison_{model_name}.png')

    # Create a new figure for plotting.
    plt.figure()
    # Generate a complete tear sheet using PyFolio to analyze the strategy's performance.
    pf.create_simple_tear_sheet(returns)
    plt.suptitle(f'Performance Report for {model_name}, Win/loss Rate = {winloss_ratio}', fontsize=16)
    # Save the performance report to a PNG file using matplotlib's save feature.
    fig = plt.gcf()  # Get the current figure
    fig.savefig(f'./{file_folder}/pyfolio_report{model_name}.png')    

def BackTrader_Method_Ticker(ticker,models,file_folder):

    dataset_processing = Dataset_Processing(start_date = "1999-01-01", end_date = "2021-11-12", ticker = ticker)
    ticker_dataset = dataset_processing.Get_stock()
    dataset = dataset_processing.input_data
    # models = [Model(dataset = dataset,ticker=ticker).Method_LinearRegression(),
    #           Model(dataset = dataset,ticker=ticker).Method_LassoRegression(), 
    #           Model(dataset = dataset,ticker=ticker).Method_RidgeRegression(), 
    #           Model(dataset = dataset,ticker=ticker).Method_LogisticRegression(),
    #           Model(dataset = dataset,ticker=ticker).Method_Random_Forest(),
    #           Model(dataset = dataset,ticker=ticker).Method_Adaboost(),
    #           Model(dataset = dataset,ticker=ticker).Method_NeuralNetworkRegression(),
    #           Model(dataset = dataset,ticker=ticker).Method_NeuralNetworkClassification()]
    dataset = pd.merge(ticker_dataset,dataset,left_index=True, right_index=True, how='right')
    dataset = dataset.dropna()
    BackTrader(models[0],dataset,f'{ticker}_LinearRegression',file_folder)
    BackTrader(models[1],dataset,f'{ticker}_LassoRegression',file_folder)
    BackTrader(models[2],dataset,f'{ticker}_RidgeRegression',file_folder)
    BackTrader(models[3],dataset,f'{ticker}_LogisticRegression',file_folder)
    BackTrader(models[4],dataset,f'{ticker}_RandomForest',file_folder)
    BackTrader(models[5],dataset,f'{ticker}_Adaboost',file_folder)
    BackTrader(models[6],dataset,f'{ticker}_NeuralNetworkRegression',file_folder,'NNR')
    BackTrader(models[7],dataset,f'{ticker}_NeuralNetworkClassification',file_folder,'NNC')

def BackTrader_Method_Ticker_Best(ticker,file_folder):

    dataset_processing = Dataset_Processing(start_date = "1999-01-01", end_date = "2021-11-12", ticker = ticker)
    ticker_dataset = dataset_processing.Get_stock()
    dataset = dataset_processing.input_data
    # models = [Model(dataset = dataset,ticker=ticker).Method_LassoRegression(), Model(dataset = dataset,ticker=ticker).Method_NeuralNetworkClassification()]
    models = [Model(dataset = dataset,ticker=ticker).Method_RidgeRegression(), 
              Model(dataset = dataset,ticker=ticker).Method_Adaboost()]
    dataset = pd.merge(ticker_dataset,dataset,left_index=True, right_index=True, how='right')
    dataset = dataset.dropna()
    # BackTrader(models[0],dataset,f'{ticker}_LassoRegression',file_folder)
    # BackTrader(models[1],dataset,f'{ticker}_NeuralNetworkClassification',file_folder,'NNC')
    # sharpe_ratio_Lasso, max_drawdown_Lasso = BackTrade_Shape_MDD(models[0],dataset,f'{ticker}_LassoRegression')
    # sharpe_ratio_NNC, max_drawdown_NNC = BackTrade_Shape_MDD(models[1],dataset,f'{ticker}_NeuralNetworkClassification','NNC')
    BackTrader(models[0],dataset,f'{ticker}_RidgeRegression',file_folder)
    BackTrader(models[1],dataset,f'{ticker}_Adaboost',file_folder)
    sharpe_ratio_Ridge, max_drawdown_Ridge = BackTrade_Shape_MDD(models[0],dataset,f'{ticker}_RidgeRegression')
    sharpe_ratio_Ada, max_drawdown_Ada = BackTrade_Shape_MDD(models[1],dataset,f'{ticker}_Adaboost')
    return sharpe_ratio_Ridge, max_drawdown_Ridge,sharpe_ratio_Ada, max_drawdown_Ada

def BackTrade_Shape_MDD(model, dataset, model_name, method='Non_NN'):
    """
    Execute a backtrading strategy using the specified model and dataset.

    :param model: A pre-trained model used for trading decisions.
    :param dataset: The dataset used for backtesting, typically historical stock data.
    :param model_name: The name of the model, used for naming output files.
    :param method: The method of prediction, default is 'Non_NN' for non-neural network models.
    """
    # Initialize a Cerebro instance for backtesting.
    cerebro = bt.Cerebro()
    # Create a data feed from the provided dataset.
    data = ExtendedPandasData(dataname=dataset)
    cerebro.adddata(data)
    # Add a trading strategy to Cerebro and pass the pre-trained model.
    cerebro.addstrategy(Trading_Strategy, model=model, method=method)
    # Set the initial capital for the trading simulation.
    cerebro.broker.setcash(100000.0)
    # Attach a PyFolio analyzer for performance analysis.
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_analyzer', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown_analyzer')
    # Run the backtesting strategy.
    results = cerebro.run()
    # Extract the first strategy's results (assuming a single strategy).
    strategy = results[0]
    # Retrieve performance metrics: sharpe_ratio,max_drawdown
    sharpe_ratio = results[0].analyzers.getbyname('sharpe_analyzer').get_analysis()
    max_drawdown = results[0].analyzers.getbyname('drawdown_analyzer').get_analysis()
    return sharpe_ratio, max_drawdown
