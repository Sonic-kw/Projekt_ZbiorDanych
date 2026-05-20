import pandas as pd
from typing import Dict
from src.utils.logger import setup_logger

logger = setup_logger("normalizer")

class DataNormalizer:
    def __init__(self, brand_mapping: Dict[str, str] = None):
        # Default mapping for common brand variations
        self.brand_mapping = brand_mapping or {
            "Yamaha Motor": "Yamaha",
            "Honda Motor": "Honda",
            "Kawasaki Heavy Industries": "Kawasaki",
            "Suzuki Motor": "Suzuki",
            "BMW Motorrad": "BMW",
            "KTM AG": "KTM",
            "Ducati": "Ducati",
            "Triumph": "Triumph"
        }

    def normalize_brands(self, df: pd.DataFrame, column: str = "brand") -> pd.DataFrame:
        """
        Normalizes brand names based on the mapping.
        """
        df = df.copy()
        df[column] = df[column].replace(self.brand_mapping)
        logger.info("Brand names normalized.")
        return df
