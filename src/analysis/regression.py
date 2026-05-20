import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from src.utils.logger import setup_logger

logger = setup_logger("regression")

class PricePredictor:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, df: pd.DataFrame, features: list = ['year', 'mileage']):
        """
        Trains a linear regression model to predict price.
        """
        X = df[features]
        y = df['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        logger.info(f"Model trained. MAE: {mae:.2f}, R2: {r2:.2f}")
        return mae, r2

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts prices for given features."""
        return self.model.predict(X)
