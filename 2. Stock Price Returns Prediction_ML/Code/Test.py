from Li_Yunfan_Dataset_Processing import *
from Li_Yunfan_Model import *
from Li_Yunfan_BackTrader import *

def Write_to_csv(df, file_path):
        # Check if the file already exists
        try:
            with open(file_path, 'x') as f:
                df.to_csv(f)
        except FileExistsError:
            # If the file exists, append the data without writing the headers
            with open(file_path, 'a') as f:
                df.to_csv(f)

if __name__ == "__main__":
    tickers = pd.read_csv('./tickers.csv')
    tickers_list = tickers['Ticker'].tolist() 
    for ticker in tickers_list:
        dataset_processing = Dataset_Processing(start_date = "1999-01-01",end_date = "2021-11-12", ticker = ticker)
        dataset = dataset_processing.input_data
        model = Model(dataset,ticker)
        models = model.Runall()
        model.Matrics_to_csv('./Matrics_tickers_1.csv')
        BackTrader_Method_Ticker(ticker,models,'Outputs_10_Tickers')

# Large Universe
    NASD = pd.read_csv("tickers_nasd.csv")
    NYSE = pd.read_csv("tickers_nyse.csv")
    US_Large_Stock = pd.concat([NASD, NYSE], axis=0)
    US_Large_Stock = US_Large_Stock.sort_values(by=['MarketCap'], ascending=False)
    symbols = US_Large_Stock['Symbol'][:200].to_list()
    for symbol in symbols:
        try:
            print(f'Begin to Download Data for {symbol}')
            stock_data = yf.Ticker(symbol)

            if stock_data.history(period="1d").empty:
                print(f"Skipping {symbol} as it may not be listed or is delisted.")
                continue

            dataset_processing = Dataset_Processing(start_date="1999-01-01", end_date="2021-11-12", ticker=symbol)
            dataset = dataset_processing.input_data

            model = Model(dataset, symbol)
            model.Method_RidgeRegression()
            model.Method_Adaboost()
            model.Matrics_to_csv('./Matrics_Large_Stock.csv')

        except:
            print(f"Error occurred with {symbol}")
            continue

    Large_stock = pd.read_csv('./Matrics_Large_Stock.csv')
    Large_stock['Stock'], Large_stock['Model'] = Large_stock['Unnamed: 0'].str.rsplit('_', n=1).str
    Large_stock = Large_stock.groupby('Stock')[['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC Score']].mean().reset_index()
    Large_stock = Large_stock.sort_values(by=['F1 Score'], ascending=False)
    stocks = Large_stock['Stock'][0:10].to_list()
    Ridge_sharpe_list = []
    Ridge_mdd_list = []
    Ada_sharpe_list = []
    Ada_mdd_list = []
    for stock in stocks:
        sharpe_ratio_Ridge, max_drawdown_Ridge,sharpe_ratio_Ada, max_drawdown_Ada = BackTrader_Method_Ticker_Best(stock,'Outputs_10_Large_Stocks')
        Ridge_sharpe_list.append(sharpe_ratio_Ridge['sharperatio'])
        Ridge_mdd_list.append(max_drawdown_Ridge['max']['drawdown'])
        Ada_sharpe_list.append(sharpe_ratio_Ada['sharperatio'])
        Ada_mdd_list.append(max_drawdown_Ada['max']['drawdown'])
    df = pd.DataFrame({
    'Stock': stocks,
    'Ridge Sharpe Ratio': Ridge_sharpe_list,
    'Ridge Negative Max Drawdown': Ridge_mdd_list,
    'Adaboost Sharpe Ratio': Ada_sharpe_list,
    'Adaboost Negative Max Drawdown': Ada_mdd_list
})
    Write_to_csv(df,'./Sort_large_stock.csv')
    for column in df.columns:
        if column != 'Stock':
            # Determine sort order: descending for Sharpe, ascending for Drawdown
            is_sharpe = "Sharpe" in column
            sorted_indices = df[column].sort_values(ascending=not is_sharpe).index
            df[column] = df.loc[sorted_indices, 'Stock'].values
    Write_to_csv(df,'./Sort_large_stock.csv')

    # Small Universe
    NASD = pd.read_csv("tickers_nasd.csv")
    NYSE = pd.read_csv("tickers_nyse.csv")
    US_Small_Stock = pd.concat([NASD, NYSE], axis=0)
    US_Small_Stock = US_Small_Stock.sort_values(by=['MarketCap'], ascending=True)
    symbols = US_Small_Stock['Symbol'][:200].to_list()
    for symbol in symbols:
        try:
            print(f'Begin to Download Data for {symbol}')
            stock_data = yf.Ticker(symbol)

            if stock_data.history(period="1d").empty:
                print(f"Skipping {symbol} as it may not be listed or is delisted.")
                continue

            dataset_processing = Dataset_Processing(start_date="1999-01-01", end_date="2021-11-12", ticker=symbol)
            dataset = dataset_processing.input_data

            model = Model(dataset, symbol)
            model.Method_RidgeRegression()
            model.Method_Adaboost()
            model.Matrics_to_csv('./Matrics_Small_Stock.csv')

        except:
            print(f"Error occurred with {symbol}")
            continue

    Small_stock = pd.read_csv('./Matrics_Small_Stock.csv')
    Small_stock['Stock'], Small_stock['Model'] = Small_stock['Unnamed: 0'].str.rsplit('_', n=1).str
    Small_stock = Small_stock.groupby('Stock')[['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC Score']].mean().reset_index()
    Small_stock = Small_stock.sort_values(by=['F1 Score'], ascending=False)
    stocks = Small_stock['Stock'][0:10].to_list()
    Ridge_sharpe_list = []
    Ridge_mdd_list = []
    Ada_sharpe_list = []
    Ada_mdd_list = []
    for stock in stocks:
        sharpe_ratio_Ridge, max_drawdown_Ridge,sharpe_ratio_Ada, max_drawdown_Ada = BackTrader_Method_Ticker_Best(stock,'Outputs_10_Small_Stocks')
        Ridge_sharpe_list.append(sharpe_ratio_Ridge['sharperatio'])
        Ridge_mdd_list.append(max_drawdown_Ridge['max']['drawdown'])
        Ada_sharpe_list.append(sharpe_ratio_Ada['sharperatio'])
        Ada_mdd_list.append(max_drawdown_Ada['max']['drawdown'])
    df = pd.DataFrame({
        'Stock': stocks,
        'Ridge Sharpe Ratio': Ridge_sharpe_list,
        'Ridge Negative Max Drawdown': Ridge_mdd_list,
        'Adaboost Sharpe Ratio': Ada_sharpe_list,
        'Adaboost Negative Max Drawdown': Ada_mdd_list
    })
    Write_to_csv(df,'./Sort_small_stock.csv')
    for column in df.columns:
        if column != 'Stock':
            # Determine sort order: descending for Sharpe, ascending for Drawdown
            is_sharpe = "Sharpe" in column
            sorted_indices = df[column].sort_values(ascending=not is_sharpe).index
            df[column] = df.loc[sorted_indices, 'Stock'].values
    Write_to_csv(df,'./Sort_small_stock.csv')