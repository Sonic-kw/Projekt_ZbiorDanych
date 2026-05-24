import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("filter")

class DataFilter:
    REQUIRED_COLUMNS = ["brand", "model", "year", "mileage", "price"]

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_count - len(df)} duplicate records.")
        return df

    @staticmethod
    def remove_incomplete(df: pd.DataFrame, required_columns: list | None = None) -> pd.DataFrame:
        required_columns = required_columns or DataFilter.REQUIRED_COLUMNS
        present_required_columns = [col for col in required_columns if col in df.columns]
        initial_count = len(df)
        if present_required_columns:
            df = df.dropna(subset=present_required_columns)
        for col in present_required_columns:
            if df[col].dtype in [np.float64, np.int64]:
                df = df[df[col] > 0]
        
        logger.info(f"Removed {initial_count - len(df)} incomplete records.")
        return df

    @staticmethod
    def remove_outliers_iqr(df: pd.DataFrame, columns: list, threshold: float = 1.5) -> pd.DataFrame:
        """
        Removes outliers using the Interquartile Range (IQR) method.
        """
        df_cleaned = df.copy()
        for col in columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            initial_len = len(df_cleaned)
            df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
            logger.info(f"Removed {initial_len - len(df_cleaned)} outliers from column {col}.")
            
        return df_cleaned
