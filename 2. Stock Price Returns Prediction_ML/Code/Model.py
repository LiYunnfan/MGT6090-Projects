import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression, Lasso, Ridge,LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from Li_Yunfan_Dataset_Processing import *
import warnings
warnings.filterwarnings("ignore")

class Model():
    def __init__(self, dataset, ticker):
        self.dataset = dataset
        self.ticker = ticker
        self.NN_regression = None
        self.NN_classification = None
        self.y_test_dict = {}
        self.y_pred_dict = {}
        self.metrics = {}
        self.model = {}
    
    def Runall(self):
        models = [
            self.Method_LinearRegression(),
            self.Method_LassoRegression(),
            self.Method_RidgeRegression(),
            self.Method_LogisticRegression(),
            self.Method_Random_Forest(),
            self.Method_Adaboost(),
            self.Method_NeuralNetworkRegression(),
            self.Method_NeuralNetworkClassification()]
        return models

    def Decision_making(self, data, threshold = 0):
        """
        Determine buy, sell, or hold decisions based on the given threshold.

        :param data: A list containing price data.
        :param threshold: The threshold used to make buy or sell decisions. Default is 0.

        :return: A list of decisions, where 1 represents buy(rise), 0 represents sell(fall).
        """
        # decisions = [1 if RET > threshold else (-1 if RET <= -threshold else 0) for RET in data]
        decisions = [1 if RET > threshold else -1 for RET in data]
        return decisions
    
    def Dataset_split_regression(self):
        y = self.dataset['RET_1d']
        X = self.dataset.drop(columns=['RET_1d'])
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)
        return X_train, X_test, y_train, y_test
    
    def Dataset_split_classificaion(self):
        y = self.dataset['RET_1d']
        y = self.Decision_making(y)
        X = self.dataset.drop(columns=['RET_1d'])
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)
        return X_train, X_test, y_train, y_test

    def Metrics(self, model_type):
        """
        A function to obtain classification result metrics
        :param model_name: a string denoting the name of the ML model
        """
        self.metrics[f'{self.ticker}_{model_type}'] = {"Accuracy": metrics.accuracy_score(self.y_test_dict[model_type], self.y_pred_dict[model_type]),
                                    "F1 Score": metrics.f1_score(self.y_test_dict[model_type], self.y_pred_dict[model_type]),
                                    "Precision": metrics.precision_score(self.y_test_dict[model_type], self.y_pred_dict[model_type]),
                                    "Recall": metrics.recall_score(self.y_test_dict[model_type], self.y_pred_dict[model_type]),
                                    "ROC AUC Score": metrics.roc_auc_score(self.y_test_dict[model_type], self.y_pred_dict[model_type])}
        print(f'{self.ticker}_{model_type} Matrics Saved!')
    
    def Matrics_to_csv(self,file_path):
        df = pd.DataFrame(self.metrics).T
        # Check if the file already exists
        try:
            with open(file_path, 'x') as f:
                df.to_csv(f)
        except FileExistsError:
            # If the file exists, append the data without writing the headers
            with open(file_path, 'a') as f:
                df.to_csv(f, header=False)

    def Method_LinearRegression(self):
        """
        Perform Linear Regression on the dataset and calculate accuracy scores for training and testing sets.

        This method splits the dataset into training and testing sets, trains a Linear Regression model,
        makes predictions, and calculates accuracy scores for both training and testing data.

        :return: None
        """
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = self.Dataset_split_regression()
        # Creating a Linear Regression model and fitting it to the training data
        model = LinearRegression()
        model.fit(X_train, y_train)
        # Predicting the target variable for the training set and making buy/sell/hold decisions
        y_train_pred = model.predict(X_train)
        y_train_pred = self.Decision_making(y_train_pred)
        y_train = self.Decision_making(y_train)
        # Predicting the target variable for the testing set and making buy/sell/hold decisions
        y_test_pred = model.predict(X_test)
        y_test_pred = self.Decision_making(y_test_pred)
        y_test = self.Decision_making(y_test)
        # Calculating the matrics
        self.y_test_dict['LinearRegression'] = y_test
        self.y_pred_dict['LinearRegression'] = y_test_pred
        self.model['LinearRegression'] = model
        self.Metrics('LinearRegression')

        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        # Printing the accuracy scores for both training and testing sets
        print('LinearRegression:')
        print('Train: Linear Regression Accuracy:', accuracy_train)
        print('Test: Linear Regression Accuracy:', accuracy_test)
        print('\n')
        return model
        
    def Method_LassoRegression(self, alpha = 0.5):
        """
        Perform Lasso Regression on the dataset and calculate accuracy scores for training and testing sets.

        This method splits the dataset into training and testing sets, trains a Lasso Regression model,
        makes predictions, and calculates accuracy scores for both training and testing data.

        :return: None
        """
        # Splitting the data into training and testing sets
        # X_train, X_test, y_train, y_test = self.Dataset_split_regression()
        X_train, X_test, y_train, y_test = self.Dataset_split_regression()
        # Creating a Lasso Regression model with a specified alpha (L1 regularization parameter)
        model = Lasso(alpha=alpha)
        # Fitting the model to the training data
        model.fit(X_train, y_train)
        # Predicting the target variable for the training set and making buy/sell/hold decisions
        y_train_pred = model.predict(X_train)
        y_train_pred = self.Decision_making(y_train_pred)
        y_train = self.Decision_making(y_train)
        # Predicting the target variable for the testing set and making buy/sell/hold decisions
        y_test_pred = model.predict(X_test)
        y_test_pred = self.Decision_making(y_test_pred)
        y_test = self.Decision_making(y_test)
        # Calculating the matrics
        self.y_test_dict['LassoRegression'] = y_test
        self.y_pred_dict['LassoRegression'] = y_test_pred
        self.model['LassoRegression'] = model
        self.Metrics('LassoRegression')

        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        # Printing the accuracy scores for both training and testing sets
        print('Lasso Regression:')
        print('Train: Lasso Regression Accuracy:', accuracy_train)
        print('Test: Lasso Regression Accuracy:', accuracy_test)
        print('\n')
        return model

    def Method_RidgeRegression(self,alpha = 0.5):
        """
        Perform Ridge Regression on the dataset and calculate accuracy scores for training and testing sets.

        This method splits the dataset into training and testing sets, trains a Ridge Regression model,
        makes predictions, and calculates accuracy scores for both training and testing data.

        :return: None
        """
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = self.Dataset_split_regression()
        # Creating a Ridge Regression model
        model = Ridge(alpha=alpha)
        # Fitting the model to the training data
        model.fit(X_train, y_train)
        # Predicting the target variable for the training set and making buy/sell/hold decisions
        y_train_pred = model.predict(X_train)
        y_train_pred = self.Decision_making(y_train_pred)
        y_train = self.Decision_making(y_train)
        # Predicting the target variable for the testing set and making buy/sell/hold decisions
        y_test_pred = model.predict(X_test)
        y_test_pred = self.Decision_making(y_test_pred)
        y_test = self.Decision_making(y_test)

    #     weights = model.coef_
    #     feature_names = ['RET_1w', 'RET_1m', 'RET_3m', 'RET_6m', 'RET_12m',
    #    'Volatility_1w', 'Volatility_1m', 'Volatility_3m', 'Volatility_6m',
    #    'Volatility_12m', 'Volume_1d', 'Volume_1w', 'Volume_1m', 'Volume_3m',
    #    'Volume_6m', 'Volume_12m', 'Volume_Volatility_1w',
    #    'Volume_Volatility_1m', 'Volume_Volatility_3m', 'Volume_Volatility_6m',
    #    'Volume_Volatility_12m', 'SMA', 'BB_UP', 'BB_DOWN', 'macd_line',
    #    'macd_signal_line', 'macd_histogram', 'ADX', 'di_positive',
    #    'di_negative', 'RSI', 'USREC', 'DFF', 'UMCSENT', 'UNRATE'] 
    #     df_features = pd.DataFrame({
    #         'Feature Name': feature_names,
    #         'Weight': weights
    #         })
    #     df_sorted = df_features.sort_values(by='Weight', ascending=False).reset_index(drop=True)
    #     pd.set_option('display.max_rows', None)
    #     print(df_sorted)
        
        # Calculating the matrics
        self.y_test_dict['RidgeRegression'] = y_test
        self.y_pred_dict['RidgeRegression'] = y_test_pred
        self.model['RidgeRegression'] = model
        self.Metrics('RidgeRegression')

        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        # Printing the accuracy scores for both training and testing sets
        print('Ridge Regression:')
        print('Train: Ridge Regression Accuracy:', accuracy_train)
        print('Test: Ridge Regression Accuracy:', accuracy_test)
        print('\n')
        return model


    def Method_LogisticRegression(self):
        """
        Perform Logistic Regression on the dataset and calculate accuracy scores for training and testing sets.

        This method splits the dataset into training and testing sets, trains a Logistic Regression model,
        makes predictions, and calculates accuracy scores for both training and testing data.

        :return: None
        """
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = self.Dataset_split_classificaion()
        # Creating a Logistic Regression model
        model = LogisticRegression()
        # Fitting the model to the training data
        model.fit(X_train, y_train)
        # Predicting the target variable for the training set and making buy/sell/hold decisions
        y_train_pred = model.predict(X_train)
        # Predicting the target variable for the testing set and making buy/sell/hold decisions
        y_test_pred = model.predict(X_test)
        # Calculating the matrics
        self.y_test_dict['LogisticRegression'] = y_test
        self.y_pred_dict['LogisticRegression'] = y_test_pred
        self.model['LogisticRegression'] = model
        self.Metrics('LogisticRegression')
        
        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        # Printing the accuracy scores for both training and testing sets
        print('Logistic Regression:')
        print('Train: Logistic Regression Accuracy:', accuracy_train)
        print('Test: Logistic Regression Accuracy:', accuracy_test)
        print('\n')

        return model
    
    def Method_Random_Forest(self):
        """
        Perform Random Forest on the dataset and calculate accuracy scores for training and testing sets.

        This method splits the dataset into training and testing sets, trains a Random Forest model,
        makes predictions, and calculates accuracy scores for both training and testing data.

        :return: None
        """
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = self.Dataset_split_classificaion()
        # Creating a Random Forest Regression model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Fitting the model to the training data
        model.fit(X_train, y_train)
        # Predicting the target variable for the training set and making buy/sell/hold decisions
        y_train_pred = model.predict(X_train)
        # Predicting the target variable for the testing set and making buy/sell/hold decisions
        y_test_pred = model.predict(X_test)
        # Calculating the matrics
        self.y_test_dict['Random_Forest'] = y_test
        self.y_pred_dict['Random_Forest'] = y_test_pred
        self.model['Random_Forest'] = model
        self.Metrics('Random_Forest')
        
        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        # Printing the accuracy scores for both training and testing sets
        print('Random_Forest:')
        print('Train: Random_Forest Accuracy:', accuracy_train)
        print('Test: Random_Forest Accuracy:', accuracy_test)
        print('\n')

        return model
    
    def Method_Adaboost(self):
        """
        Perform Adaboost on the dataset and calculate accuracy scores for training and testing sets.

        This method splits the dataset into training and testing sets, trains a Adaboost model,
        makes predictions, and calculates accuracy scores for both training and testing data.

        :return: None
        """
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = self.Dataset_split_classificaion()
        # Creating a Adaboost model
        base_estimator = DecisionTreeClassifier(max_depth=1)
        model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100,learning_rate=1,random_state=42)
        # Fitting the model to the training data
        model.fit(X_train, y_train)
        # Predicting the target variable for the training set and making buy/sell/hold decisions
        y_train_pred = model.predict(X_train)
        # Predicting the target variable for the testing set and making buy/sell/hold decisions
        y_test_pred = model.predict(X_test)
        # Calculating the matrics
        self.y_test_dict['AdaBoost'] = y_test
        self.y_pred_dict['AdaBoost'] = y_test_pred
        self.model['AdaBoost'] = model
        self.Metrics('AdaBoost')
        
        accuracy_train = accuracy_score(y_train, y_train_pred)
        accuracy_test = accuracy_score(y_test, y_test_pred)
        # Printing the accuracy scores for both training and testing sets
        print('AdaBoost:')
        print('Train: AdaBoost Accuracy:', accuracy_train)
        print('Test: AdaBoost Accuracy:', accuracy_test)
        print('\n')

        return model

    def Method_NeuralNetworkRegression(self, lr = 1e-2, epochs = 1500):

        class NN_regression(nn.Module):
            """
            Define a neural network architecture for regression.
            The network consists of three linear layers with ReLU activations and dropout for regularization.
            """
            def __init__(self):
                super(NN_regression, self).__init__()
                self.relu = nn.ReLU()  # ReLU activation function
                self.dropout = nn.Dropout(p=0.2)  # Dropout layer with a probability of 0.2
                # Fully connected layers
                self.linear1 = nn.Linear(35, 40) 
                self.linear2 = nn.Linear(40, 20)  
                self.linear3 = nn.Linear(20, 1)   
            def forward(self, x):
                """
                Define the forward pass of the neural network.
                """
                x = self.linear1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.linear2(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.linear3(x)
                return x
        def Deal_with_y_NNR(y, y_pred):
            """
            Calculate accuracy of the model's predictions.
            This function converts continuous predictions into discrete decisions using a custom Decision_making function and computes the accuracy.
            :param y: Actual target values.
            :param y_pred: Predicted target values from the model.
            :return: Computed accuracy of the predictions.
            """
            y_ = self.Decision_making(y)
            y_pred_ = self.Decision_making(y_pred)
            return y_,y_pred_
        
        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = self.Dataset_split_regression()  
        # Convert data to PyTorch tensors
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        # Initialize variables for tracking the best performance
        min_loss = float('inf')
        best_epoch = -1
        best_accuracy = 0.0
        # Model training
        self.NN_regression = NN_regression()  # Instantiate the neural network
        criterion = nn.MSELoss()  # Mean Squared Error Loss
        optimizer = optim.SGD(self.NN_regression.parameters(), lr=lr)  # Stochastic Gradient Descent optimizer
        epochs = epochs  # Number of training epochs
        for epoch in range(epochs):
            y_train_pred = self.NN_regression(X_train)  # Forward pass on training data
            y_train_pred = y_train_pred.squeeze(1)
            train_loss = criterion(y_train_pred, y_train)  # Compute training loss
            optimizer.zero_grad()  # Clear previous gradients
            train_loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            self.NN_regression.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                y_test_pred = self.NN_regression(X_test)  # Forward pass on testing data
                y_test_pred = y_test_pred.squeeze(1)
                test_loss = criterion(y_test_pred, y_test)  # Compute testing loss
                # Check if current epoch yields the minimum loss
                if test_loss < min_loss:
                    min_loss = test_loss.item()
                    best_epoch = epoch
                    y_test_best,y_pred_best = Deal_with_y_NNR(y_test, y_test_pred)
                    best_accuracy = accuracy_score(y_test_best, y_pred_best)
                    best_model = NN_regression() 
                    best_model.load_state_dict(self.NN_regression.state_dict()) 
        # Calculating the matrics
        self.y_test_dict['NeuralNetworkRegression'] = y_test_best
        self.y_pred_dict['NeuralNetworkRegression'] = y_pred_best
        self.model['NeuralNetworkRegression'] = best_model
        self.Metrics('NeuralNetworkRegression')
        # Output the best performance metrics
        print('Neural Network Regression:')
        print(f'Test: Best Epoch: {best_epoch+1}, Min Loss: {min_loss:.4f}, Corresponding Accuracy: {best_accuracy:.4f}')
        print('\n')
        ''' 
        For Train:
            # if (epoch+1) % 10 == 0:
            # accuracy_train = Evaluate_accuracy(y_train,y_train_pred)
            # print(f'Epoch [{epoch+1}/{epochs}]')
            # print(f"Train Loss: {loss.item():.4f}, Train Accuracy: {accuracy_train:.4f}")
        For Test:
            # if (epoch+1) % 10 == 0:
            # print(f"Test Loss: {loss.item():.4f}, Test Accuracy: {accuracy_test:.4f}")
        '''

        return best_model

    def Method_NeuralNetworkClassification(self, lr = 1e-2, epochs = 1500):

        class NN_classification(nn.Module):
            """
            Define a neural network architecture for regression.
            The network consists of three linear layers with ReLU activations and dropout for regularization.
            """
            def __init__(self):
                super(NN_classification, self).__init__()
                self.relu = nn.ReLU()  # ReLU activation function
                self.sigmoid = nn.Sigmoid()
                self.dropout = nn.Dropout(p=0.2)  # Dropout layer with a probability of 0.2
                # Fully connected layers
                self.linear1 = nn.Linear(35, 40) 
                self.linear2 = nn.Linear(40, 20)  
                self.linear3 = nn.Linear(20, 1)  
            def forward(self, x):
                """
                Define the forward pass of the neural network.
                """
                x = self.linear1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.linear2(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.linear3(x)
                x = self.sigmoid(x)
                return x
        def Dear_with_y_NNC(y, y_pred):
            """
            Calculate accuracy of the model's predictions.
            This function converts continuous predictions into discrete decisions using a custom Decision_making function and computes the accuracy.
            :param y: Actual target values.
            :param y_pred: Predicted target values from the model.
            :return: Computed accuracy of the predictions.
            """
            y_ = 2 * y - 1
            y_pred_ = 2 * y_pred - 1
            y_ = self.Decision_making(y)
            y_pred_ = self.Decision_making(y_pred)
            return y_,y_pred_
        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = self.Dataset_split_classificaion()  
        y_train = (np.array(y_train) + 1)/2
        y_test = (np.array(y_test) + 1)/2
        # Convert data to PyTorch tensors
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        # Initialize variables for tracking the best performance
        min_loss = float('inf')
        best_epoch = -1
        best_accuracy = 0.0
        # Model training
        self.NN_classification = NN_classification()  # Instantiate the neural network
        criterion = nn.BCELoss()  # Mean Squared Error Loss
        optimizer = optim.Adam(self.NN_classification.parameters(), lr=lr)  # Stochastic Gradient Descent optimizer
        epochs = epochs  # Number of training epochs
        for epoch in range(epochs):
            y_train_pred = self.NN_classification(X_train)  # Forward pass on training data
            y_train_pred = y_train_pred.squeeze(1)
            train_loss = criterion(y_train_pred, y_train)  # Compute training loss
            optimizer.zero_grad()  # Clear previous gradients
            train_loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            self.NN_classification.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                y_test_pred = self.NN_classification(X_test)  # Forward pass on testing data
                y_test_pred = y_test_pred.squeeze(1)
                test_loss = criterion(y_test_pred, y_test)  # Compute testing loss
                # Check if current epoch yields the minimum loss
                if test_loss < min_loss:
                    min_loss = test_loss.item()
                    best_epoch = epoch
                    y_test_best,y_pred_best = Dear_with_y_NNC(y_test, y_test_pred)
                    best_accuracy = accuracy_score(y_test_best, y_pred_best)
                    best_model = NN_classification() 
                    best_model.load_state_dict(self.NN_classification.state_dict()) 
        # Calculating the matrics
        self.y_test_dict['NeuralNetworkClassification'] = y_test_best
        self.y_pred_dict['NeuralNetworkClassification'] = y_pred_best
        self.model['NeuralNetworkClassification'] = best_model
        self.Metrics('NeuralNetworkClassification')
        # Output the best performance metrics
        print('Neural Network Classification:')
        print(f'Test: Best Epoch: {best_epoch+1}, Min Loss: {min_loss:.4f}, Corresponding Accuracy: {best_accuracy:.4f}')
        print('\n')
        
        ''' 
        For Train:
            # if (epoch+1) % 10 == 0:
            # accuracy_train = Evaluate_accuracy(y_train,y_train_pred)
            # print(f'Epoch [{epoch+1}/{epochs}]')
            # print(f"Train Loss: {loss.item():.4f}, Train Accuracy: {accuracy_train:.4f}")
        For Test:
            # if (epoch+1) % 10 == 0:
            # print(f"Test Loss: {loss.item():.4f}, Test Accuracy: {accuracy_test:.4f}")
        '''
        return best_model

