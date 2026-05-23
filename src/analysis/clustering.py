import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from src.utils.logger import setup_logger

logger = setup_logger("clustering")

class MarketSegmenter:
    def __init__(
        self,
        n_clusters: int = 3,
        algorithm: str = "auto",
        dbscan_eps: float = 0.7,
        dbscan_min_samples: int = 12,
    ):
        self.n_clusters = n_clusters
        self.algorithm = algorithm.lower()
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.scaler = StandardScaler()
        self.last_metrics = {}

        supported_algorithms = {"auto", "kmeans", "dbscan"}
        if self.algorithm not in supported_algorithms:
            supported = ", ".join(sorted(supported_algorithms))
            raise ValueError(f"Unsupported clustering algorithm '{algorithm}'. Use one of: {supported}.")

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

    @staticmethod
    def _calculate_silhouette(features: np.ndarray, labels: np.ndarray) -> float | None:
        # For DBSCAN we ignore noise points (-1) in quality metric calculation.
        mask = labels != -1
        filtered_features = features[mask]
        filtered_labels = labels[mask]

        if len(filtered_features) < 3:
            return None
        if np.unique(filtered_labels).size < 2:
            return None

        return float(silhouette_score(filtered_features, filtered_labels))

    def _fit_kmeans(self, scaled_features: np.ndarray) -> tuple[np.ndarray, float | None]:
        effective_clusters = min(self.n_clusters, len(scaled_features))
        if effective_clusters < 1:
            raise ValueError("Not enough samples to perform clustering.")

        model = KMeans(n_clusters=effective_clusters, random_state=42, n_init="auto")
        labels = model.fit_predict(scaled_features)
        silhouette = self._calculate_silhouette(scaled_features, labels)
        return labels, silhouette

    def _fit_dbscan(self, scaled_features: np.ndarray) -> tuple[np.ndarray, float | None]:
        model = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        labels = model.fit_predict(scaled_features)
        silhouette = self._calculate_silhouette(scaled_features, labels)
        return labels, silhouette

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Clusters the motorcycles based on the coefficients.
        """
        features = self.prepare_features(df)
        scaled_features = self.scaler.fit_transform(features)

        kmeans_labels, kmeans_silhouette = self._fit_kmeans(scaled_features)
        dbscan_labels, dbscan_silhouette = self._fit_dbscan(scaled_features)

        selected_algorithm = self.algorithm
        selected_labels = kmeans_labels
        selected_silhouette = kmeans_silhouette

        if self.algorithm == "auto":
            kmeans_score = kmeans_silhouette if kmeans_silhouette is not None else float("-inf")
            dbscan_score = dbscan_silhouette if dbscan_silhouette is not None else float("-inf")
            if dbscan_score > kmeans_score:
                selected_algorithm = "dbscan"
                selected_labels = dbscan_labels
                selected_silhouette = dbscan_silhouette
            else:
                selected_algorithm = "kmeans"
        elif self.algorithm == "dbscan":
            selected_algorithm = "dbscan"
            selected_labels = dbscan_labels
            selected_silhouette = dbscan_silhouette
        else:
            selected_algorithm = "kmeans"

        self.last_metrics = {
            "selected_algorithm": selected_algorithm,
            "selected_silhouette": selected_silhouette,
            "kmeans_silhouette": kmeans_silhouette,
            "dbscan_silhouette": dbscan_silhouette,
            "kmeans_cluster_count": int(np.unique(kmeans_labels).size),
            "dbscan_cluster_count": int(np.unique(dbscan_labels[dbscan_labels != -1]).size),
            "dbscan_noise_count": int((dbscan_labels == -1).sum()),
        }

        logger.info(
            "Clustering completed using %s | silhouette=%.4f | kmeans=%.4f | dbscan=%.4f",
            selected_algorithm,
            selected_silhouette if selected_silhouette is not None else float("nan"),
            kmeans_silhouette if kmeans_silhouette is not None else float("nan"),
            dbscan_silhouette if dbscan_silhouette is not None else float("nan"),
        )
        return pd.Series(selected_labels, index=df.index)
