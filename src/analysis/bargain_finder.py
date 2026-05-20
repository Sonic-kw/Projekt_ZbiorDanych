import pandas as pd
from src.analysis.regression import PricePredictor
from src.utils.logger import setup_logger

logger = setup_logger("bargain_finder")

class BargainFinder:
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
        self.predictor = PricePredictor()

    def find_bargains(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies bargains where Actual Price < Predicted Price * (1 - Threshold).
        """
        # Train predictor on the current dataset
        self.predictor.train(df)
        
        # Predict prices for all listings
        predicted_prices = self.predictor.predict(df[['year', 'mileage']])
        
        df_with_pred = df.copy()
        df_with_pred['predicted_price'] = predicted_prices
        
        # Bargain condition: Actual < Predicted * (1 - threshold)
        bargains = df_with_pred[
            df_with_pred['price'] < df_with_pred['predicted_price'] * (1 - self.threshold)
        ]
        
        logger.info(f"Found {len(bargains)} potential bargains.")
        return bargains
