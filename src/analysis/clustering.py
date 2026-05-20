import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.utils.logger import setup_logger

logger = setup_logger("clustering")

class MarketSegmenter:
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Depreciation and Exploitation coefficients.
        Depreciation: Price / (Current Year - Year)
        Exploitation: Price / Mileage
        """
        import datetime
        current_year = datetime.datetime.now().year
        
        df = df.copy()
        # Avoid division by zero
        df['age'] = (current_year - df['year']).clip(lower=1)
        df['depreciation_coeff'] = df['price'] / df['age']
        df['exploitation_coeff'] = df['price'] / df['mileage'].clip(lower=1)
        
        return df[['depreciation_coeff', 'exploitation_coeff']]

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Clusters the motorcycles based on the coefficients.
        """
        features = self.prepare_features(df)
        scaled_features = self.scaler.fit_transform(features)
        clusters = self.model.fit_predict(scaled_features)
        
        logger.info(f"Clustering completed with {self.n_clusters} clusters.")
        return pd.Series(clusters, index=df.index)
