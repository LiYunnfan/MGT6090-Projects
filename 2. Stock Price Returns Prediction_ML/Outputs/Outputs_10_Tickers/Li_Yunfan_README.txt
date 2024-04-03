### Python Package Versions
This project is compatible with Python version 3.7.x. The key Python packages and dependencies used in the code include:
- `Pandas`
- `Numpy`
- `Sklearn` (including specific modules such as `DecisionTreeClassifier`, `Lasso`, `LinearRegression`, etc.)
- `Backtrader`
- `Yfinance`
- `Pyfolio`
- `Torch`
- `Matplotlib`
- `Pandas Datareader`

*Note: Please ensure you have these packages installed in your Python environment. 
The exact version compatibility is not specified in the code. 
It's recommended to use the latest versions of these packages that are compatible with Python 3.7.x.*

### Compilation and Execution Instructions
1. **Dataset Processing (`Dataset_Processing.py`):**
   - This script processes the dataset required for the models. 

2. **Model Training (`Model.py`):**
   - This script is responsible for training machine learning models.

3. **Backtesting (`BackTrader.py`):**
   - Use this script for backtesting the models with historical data.

4. **Testing (`Test.py`):**
   - This script is used for testing the models with all other three python file Dataset_Processing.py, Model.py and BackTrader.py.
   - Run using `python Test.py` or  `python3 Test.py`
   - Outputs from this script will include performance metrics of the models, trading reports of stocks.

*Please ensure all scripts are in the same directory and have access to any required data files.*