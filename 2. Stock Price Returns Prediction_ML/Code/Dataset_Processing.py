from sklearn.preprocessing import MinMaxScaler
import pandas_datareader.data as pdr
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings("ignore")

class Dataset_Processing():
    def __init__(self, start_date, end_date, ticker):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker
        self.ticker_data = self.Get_stock()
        self.macro_factors = self.Factors_macro()
        self.stock_factors = self.Factors_stock(self.ticker_data)
        self.stock_factors = self.Drop_columes(self.stock_factors)
        self.combines_df = self.Merge_Dataframe(self.stock_factors,self.macro_factors)
        self.combines_df = self.Deal_Missing_data(self.combines_df)
        # self.input_data = self.Reduce_Dimension(self.combines_df)
        # self.combines_df = self.Reduce_Dimension(self.combines_df)
        self.input_data = self.Regularization(self.combines_df)

    def Read_Stock(input_type,file_path):
        
        if input_type == '1': # Date Open High Low Close Volume	Adj Close
            data = pd.read_csv(file_path)
            data.drop(columns=['Close'],inplace=True)
            data.rename(columns={'Adj Close':'Close'},inplace=True)
            return data
        
        if input_type == '2': # open close high low	volume money avg high_limit low_limit pre_close	paused factor
            data = pd.read_csv(file_path)
            new_colume = ['Date','Open','High','Low','Volume','Close']
            data = data[['Unnamed: 0','open','high','low','volume','close']]
            data.columns = new_colume
            return data
        
        if input_type == '3': # Date Hour_of_Day Close
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.groupby(['Date']).last().reset_index()
            data.drop(columns=['Hour_of_Day'],inplace=True)
            return data

    def Get_stock(self):
        ticker_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        ticker_data = ticker_data.drop(columns=['Close'])
        ticker_data = ticker_data.rename(columns={'Adj Close': 'Close'})
        return ticker_data
    
    def Drop_columes(self, df):
        df = df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis = 'columns')
        return df

    def Factors_macro(self, tickers=['USREC', 'DFF', 'UMCSENT', 'UNRATE']):
        """
        Retrieve and process macroeconomic factors from the FRED (Federal Reserve Economic Data).

        :param tickers: List of FRED series IDs. Defaults to ['USREC', 'DFF', 'UMCSENT', 'UNRATE'].
                        USREC - U.S. Recession Indicator,
                        DFF - Fed Funds Rate,
                        UMCSENT - Consumer Sentiment Index,
                        UNRATE - Unemployment Rate.
        :return: A DataFrame containing processed macroeconomic factors.

        The function fetches the data for the given tickers, forward fills missing values, 
        applies logarithmic transformation and differencing to the oil prices, 
        and backward fills any remaining missing values.
        """
        factors_macro = pdr.DataReader(tickers, 'fred', self.start_date, self.end_date) # Fetch macroeconomic data from FRED
        factors_macro = factors_macro.ffill() # Forward fill missing values
        factors_macro = factors_macro.bfill() # Backward fill to handle any remaining missing values after transformation
        return factors_macro
    
    def Factors_stock(self, df):

        def Factor_Return(close_prices):
            """
            Calculate logarithmic returns for multiple time periods for a given series of close prices.

            The function computes the logarithmic returns, which are the natural logarithm of the 
            price ratio between two consecutive periods. Logarithmic returns are useful for various 
            financial analyses as they are time additive.

            :param close_prices: Pandas Series representing the closing prices of an asset.
            :return: A tuple of Pandas Series for each time period's logarithmic returns.
                    The time periods are 1 day, 1 week, 1 month, 3 months, 6 months, 12 months.
            """
            RET_1d = np.log(close_prices).diff(1)
            RET_1w = np.log(close_prices).diff(5)
            RET_1m = np.log(close_prices).diff(21)
            RET_3m = np.log(close_prices).diff(63)
            RET_6m = np.log(close_prices).diff(126)
            RET_12m = np.log(close_prices).diff(252)
            return RET_1d, RET_1w, RET_1m, RET_3m, RET_6m, RET_12m
        
        def Factor_Volatility(df):
            """
            Calculate rolling volatilities over different time frames based on daily returns.

            This function computes the standard deviation (a common measure of volatility) of 
            daily returns over specified rolling windows to measure the asset's volatility over 
            different time periods: 1 week, 1 month, 3 months, 6 months, and 1 year.

            :param df: DataFrame containing the asset's daily return data.
            :param daily_ret_col: The name of the column in the DataFrame that contains the daily returns.
            :return: DataFrame with additional columns for each period's volatility.
            """
            # Calculate rolling standard deviation (volatility) for various time periods
            timeframes = {'1w': 5, '1m': 21, '3m': 63, '6m': 126, '12m': 252}
            for period, window in timeframes.items():
                volatility_col = f'Volatility_{period}'
                df[volatility_col] = df['RET_1d'].rolling(window).std()
            return df
        
        def Factor_Volume(Volumes):
            """
            Calculate logarithmic changes in trading volume over various time periods.

            This function computes the logarithmic difference of trading volumes, which can be 
            useful for analyzing volume trends over time.

            :param volume_series: Pandas Series representing trading volumes.
            :return: A tuple of Pandas Series for each time period's logarithmic volume change.
                    The time periods are 1 day, 1 week, 1 month, 3 months, 6 months, and 1 year.
            """
            Volume_1d = np.log(Volumes).diff(1)
            Volume_1w = np.log(Volumes).diff(5)
            Volume_1m = np.log(Volumes).diff(21)
            Volume_3m = np.log(Volumes).diff(63)
            Volume_6m = np.log(Volumes).diff(126)
            Volume_12m = np.log(Volumes).diff(252)
            return Volume_1d, Volume_1w, Volume_1m, Volume_3m, Volume_6m, Volume_12m
        
        def Factor_Volume_Volatility(df):
            """
            Calculate the rolling volatility of daily trading volume over different time frames.

            This function computes the standard deviation (volatility) of daily trading volumes over 
            specified rolling windows to measure the volatility in trading activity over different 
            time periods: 1 week, 1 month, 3 months, 6 months, and 1 year.

            :param df: DataFrame containing the asset's daily trading volume data.
            :param daily_volume_col: The name of the column in the DataFrame that contains the daily trading volumes.
            :return: DataFrame with additional columns for each period's volume volatility.
            """
            # Calculate rolling standard deviation (volatility) for various time periods
            timeframes = {'1w': 5, '1m': 21, '3m': 63, '6m': 126, '12m': 252}
            for period, window in timeframes.items():
                volatility_col = f'Volume_Volatility_{period}'
                df[volatility_col] = df['Volume_1d'].rolling(window).std()
            return df
        
        def Factor_SMA(df_close, window=10):
            """
            Calculate the Simple Moving Average (SMA) for a given close price series.

            SMA is a widely used indicator in technical analysis that helps smooth out price data 
            by creating a constantly updated average price over a specific time period.

            :param df_close: Pandas Series representing the close prices of a stock or asset.
            :param window: The size of the moving window for which the average is calculated.
                           This represents the number of time periods included in the calculation.
            :return: A Pandas Series representing the SMA of the provided close prices.
            """
            return df_close.rolling(window).mean() # Calculate the rolling mean (SMA) for the specified window

        def Factor_Bollinger_Bands(df_close, window=10, width=1):
            """
            Calculate Bollinger Bands for a given close price series.

            Bollinger Bands consist of three lines: 
            - The middle line is a simple moving average (SMA).
            - The upper line is calculated as SMA plus a certain number of standard deviations (width).
            - The lower line is SMA minus the same number of standard deviations (width).

            :param df_close: Pandas Series representing the close prices of a stock or asset.
            :param window: The size of the moving window for SMA and standard deviation calculation.
            :param width: The number of standard deviations to add/subtract from the SMA for the upper/lower bands.
            :return: A tuple containing the SMA (middle band), BB_UP (upper band), and BB_DOWN (lower band).
            """
            SMA = Factor_SMA(df_close, window) # Calculate the Simple Moving Average (SMA) - Middle Band
            sigma = df_close.rolling(window).std() # Calculate the rolling standard deviation (sigma) for the specified window
            BB_UP = SMA + sigma * width # Calculate the Upper Bollinger Band (BB_UP)
            BB_DOWN = SMA - sigma * width # Calculate the Lower Bollinger Band (BB_DOWN)
            return SMA, BB_UP, BB_DOWN

        def Factor_MACD(df_close, short_span=12, long_span=26, signal_span=9):
            """
            Calculate the Moving Average Convergence Divergence (MACD) indicator.

            :param df_close: Pandas Series of close prices.
            :param short_span: Span for the short-term Exponential Moving Average (EMA).
            :param long_span: Span for the long-term EMA.
            :param signal_span: Span for the signal line EMA.
            :return: A tuple (macd_line, macd_signal_line, macd_histogram).
            """
            ema_short = df_close.ewm(span=short_span, adjust=False).mean() # Calculate short-term EMA (fast line)
            ema_long = df_close.ewm(span=long_span, adjust=False).mean() # Calculate long-term EMA (slow line)
            macd_line = ema_short - ema_long # Calculate MACD line (difference between the short-term EMA and long-term EMA)
            macd_signal_line = macd_line.ewm(span=signal_span, adjust=False).mean() # Calculate Signal line (EMA of the MACD line)
            macd_histogram = macd_line - macd_signal_line # Calculate MACD histogram (difference between MACD line and Signal line)
            return macd_line, macd_signal_line, macd_histogram

        def Factor_ADX(high_prices, low_prices, close_prices, lookback_period=14):
            """
            Calculate the Average Directional Index (ADX) for a given set of price series.

            ADX is a technical analysis indicator used to quantify trend strength. The higher the ADX value, 
            the stronger the trend. The ADX is calculated based on the moving averages of the price range expansion 
            over a given period of time.

            :param high_prices: Pandas Series representing the high prices of an asset.
            :param low_prices: Pandas Series representing the low prices of an asset.
            :param close_prices: Pandas Series representing the closing prices of an asset.
            :param lookback_period: The number of periods to consider for calculating the ADX. Default is 14.
            :return: A tuple containing the ADX, Positive Directional Indicator (+DI), and Negative Directional Indicator (-DI).
            """
            # Calculate the differences between the consecutive highs and lows
            delta_high = high_prices.diff()
            delta_low = low_prices.diff()
            # Identify where the +DM and -DM are positive or negative, respectively
            dm_positive = np.where(delta_high > delta_low, np.maximum(delta_high, 0), 0)
            dm_negative = np.where(delta_low > delta_high, np.maximum(-delta_low, 0), 0)
            # Calculate the True Range (TR)
            tr1 = high_prices - low_prices
            tr2 = abs(high_prices - close_prices.shift(1))
            tr3 = abs(low_prices - close_prices.shift(1))
            true_range = np.maximum.reduce([tr1, tr2, tr3])
            # Calculate the Average True Range (ATR)
            atr = pd.Series(true_range).rolling(window=lookback_period).mean() 
            # Calculate smoothed +DI and -DI
            di_positive = 100 * pd.Series(dm_positive).rolling(window=lookback_period).mean() / atr
            di_negative = 100 * pd.Series(dm_negative).rolling(window=lookback_period).mean() / atr
            # Calculate the Directional Movement Index (DX)
            directional_index = abs(di_positive - di_negative) / (di_positive + di_negative) * 100  
            ADX = directional_index.rolling(window=lookback_period).mean() # Calculate the Average Directional Index (ADX)
            return np.array(ADX), np.array(di_positive), np.array(di_negative)
        
        def Factor_RSI(close_prices, period=14):
            """
            Calculate the Relative Strength Index (RSI) for a given series of closing prices.

            RSI is a momentum oscillator that measures the speed and change of price movements.
            RSI oscillates between zero and 100. Traditionally, and according to Wilder, RSI is considered 
            overbought when above 70 and oversold when below 30.

            :param close_prices: Pandas Series representing the closing prices of an asset.
            :param period: The lookback period for calculating RSI. Default is 14.
            :return: A Pandas Series representing the RSI values.
            """
            price_changes = close_prices.diff()  # Calculate the daily price changes
            # Separate positive and negative price changes
            gains = price_changes.clip(lower=0)
            losses = -price_changes.clip(upper=0)
            # Calculate the Exponential Moving Average (EMA) of gains and losses
            avg_gain = gains.ewm(com=period - 1, min_periods=period).mean()
            avg_loss = losses.ewm(com=period - 1, min_periods=period).mean()
            relative_strength = avg_gain / avg_loss # Calculate the Relative Strength (RS)
            RSI = 100 - (100 / (1 + relative_strength)) # Calculate the Relative Strength Index (RSI)
            return RSI
        
        df['RET_1d'],df['RET_1w'],df['RET_1m'],df['RET_3m'],df['RET_6m'],df['RET_12m'] = Factor_Return(df['Close'])
        df = Factor_Volatility(df)
        df['Volume_1d'],df['Volume_1w'],df['Volume_1m'],df['Volume_3m'],df['Volume_6m'],df['Volume_12m'] = Factor_Volume(df['Volume'])
        df = Factor_Volume_Volatility(df)
        df['SMA'],df['BB_UP'],df['BB_DOWN'] = Factor_Bollinger_Bands(df['Close'])
        df['macd_line'],df['macd_signal_line'],df['macd_histogram'] = Factor_MACD(df['Close'])
        df['ADX'],df['di_positive'],df['di_negative'] = Factor_ADX(df['High'],df['Low'],df['Close'])
        df['RSI'] = Factor_RSI(df['Close'])
        return df
    
    def Merge_Dataframe(self, stock_df, macro_df):
        """
        Merge two dataframes on their indices using a left join.

        This function merges a stock-related dataframe and a macroeconomic dataframe 
        based on their indices. The merge is performed as a left join, meaning all 
        rows from the stock dataframe are included in the result, and the matching 
        rows from the macro dataframe are added where available.

        :param stock_df: DataFrame containing stock-related data.
        :param macro_df: DataFrame containing macroeconomic data.
        :return: A merged DataFrame containing both stock and macroeconomic data.
        """
        # Merge the two dataframes on their indices using a left join
        combined_df = stock_df.merge(macro_df, left_index=True, right_index=True, how="left")
        return combined_df

    def Deal_Missing_data(self, df):
        """
        Handle missing data in a DataFrame by applying forward fill and then 
        filling any remaining missing values with the column's mean.

        :param df: DataFrame in which to handle missing data.
        :return: DataFrame with missing data handled.
        """
        df = df.drop(df.index[:252])
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill() # Fill missing values by forward filling
        df = df.fillna(df.mean()) # Fill remaining missing values with column means

        return df

    # def Reduce_Dimension(self, df, correlation_threshold=0.9):
    #     """
    #     Reduce the dimensions of a DataFrame by dropping highly correlated columns.

    #     This function identifies and removes columns that have a correlation higher than the 
    #     specified threshold, which can help in addressing multicollinearity in the data.

    #     :param df: DataFrame whose dimensions are to be reduced.
    #     :param correlation_threshold: Threshold for identifying high correlation. Columns with 
    #                                 correlation greater than this threshold will be dropped.
    #     :return: DataFrame with reduced dimensions.
    #     """
    #     correlation_matrix = df.corr().abs()
    #     upper_tri_indices = np.triu_indices_from(correlation_matrix, k=1)
    #     drop_columns = set()
    #     for i, j in zip(*upper_tri_indices):
    #         if correlation_matrix.iloc[i, j] > correlation_threshold:
    #             drop_columns.add(correlation_matrix.columns[j])
    #     df = df.drop(columns=drop_columns)
    #     return df
    
    def Regularization(self, df):
        scaler = MinMaxScaler()
        columns_to_scale = df.columns[1:]
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        return df
