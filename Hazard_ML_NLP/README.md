# Predicting Corporate Defaults Using Machine Learning and Hazard Models

This project implements and evaluates various machine learning models to predict corporate defaults. Using Python and Jupyter Notebooks, we explore models including Logistic Regression, LASSO Logistic Regression, Ridge Logistic Regression, K-Nearest Neighbor (KNN), Random Forest, Survival Random Forest, Gradient Boosted Trees (XGBOOST and LIGHTGBM), and an Artificial Neural Network (ANN) for their ability to predict corporate defaults accurately. Additionally, we extend our analysis to sentiment analysis on financial texts, applying the best-performing model to analyze SEC 10-K filings.

## Installation

To set up this project, you need to have Python installed on your system. We recommend using an Anaconda environment to manage your packages and dependencies. After installing Anaconda, create a new environment and install the required packages:

```bash
conda create --name default_prediction python=3.8
conda activate default_prediction
pip install jupyter numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm nltk tensorflow transformers scikit-survival
```

## Usage

To use this project, follow these steps:

1. Activate the Python environment:
   ```bash
   conda activate default_prediction
   ```
2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Navigate to the `*.ipynb` file within the Jupyter Notebook UI and open it.
4. Run the cells in the notebook to train models, evaluate their performance, and perform sentiment analysis.

## Models Included

- **Logistic Regression**: Basic ML model for binary classification tasks.
- **LASSO and Ridge Logistic Regression**: Logistic regression with regularization to prevent overfitting.
- **K-Nearest Neighbor (KNN)**: A non-parametric method used for classification.
- **Random Forest and Survival Random Forest**: Ensemble methods that use multiple decision trees to improve prediction accuracy.
- **Gradient Boosted Trees (XGBOOST and LIGHTGBM)**: Advanced ensemble techniques known for their performance and speed.
- **Artificial Neural Network (ANN)**: A deep learning model to capture complex nonlinear relationships.
- **Sentiment Analysis**: Utilizes the best performing model to analyze the sentiment of financial texts, specifically SEC 10-K filings.

## Contributing

Your contributions are welcome! Please follow these steps to contribute:

1. Fork the project repository.
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -am 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

For significant changes, please open an issue first to discuss what you would like to change.
