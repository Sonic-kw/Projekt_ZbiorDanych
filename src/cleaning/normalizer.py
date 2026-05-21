import pandas as pd
from typing import Dict
from src.utils.logger import setup_logger

logger = setup_logger("normalizer")

class DataNormalizer:
    def __init__(self, brand_mapping: Dict[str, str] = None):
        # Default mapping for common brand variations and names scraped from titles.
        default_mapping = {
            "Yamaha Motor": "Yamaha",
            "Honda Motor": "Honda",
            "Kawasaki Heavy Industries": "Kawasaki",
            "Suzuki Motor": "Suzuki",
            "BMW Motorrad": "BMW",
            "Bmw": "BMW",
            "KTM AG": "KTM",
            "Ktm": "KTM",
            "Ducati": "Ducati",
            "Triumph": "Triumph",
            "Triumph Motorcycles": "Triumph",
            "Harley Davidson": "Harley-Davidson",
            "Harley-Davidson Motor Company": "Harley-Davidson",
            "Moto Guzzi": "Moto Guzzi",
            "Royal Enfield": "Royal Enfield",
            "MV Agusta": "MV Agusta",
            "Cf Moto": "CFMOTO",
            "CF Moto": "CFMOTO",
            "CFMOTO": "CFMOTO",
            "QJ Motor": "QJMOTOR",
            "QJMotor": "QJMOTOR",
            "Qjmotor": "QJMOTOR",
            "Can Am": "Can-Am",
            "Can-Am": "Can-Am",
            "Indian Motorcycle": "Indian",
            "Piaggio & C. SpA": "Piaggio",
            "Peugeot Motocycles": "Peugeot",
            "Zero Motorcycles": "Zero",
            "Husqvarna Motorcycles": "Husqvarna",
            "Gas Gas": "GASGAS",
            "GasGas": "GASGAS",
        }
        self.brand_mapping = {
            self._mapping_key(key): value
            for key, value in (brand_mapping or default_mapping).items()
        }
        self.multiword_brand_prefixes = {
            ("moto", "guzzi"): "Moto Guzzi",
            ("royal", "enfield"): "Royal Enfield",
            ("mv", "agusta"): "MV Agusta",
            ("cf", "moto"): "CFMOTO",
            ("qj", "motor"): "QJMOTOR",
            ("gas", "gas"): "GASGAS",
            ("harley", "davidson"): "Harley-Davidson",
            ("indian", "motorcycle"): "Indian",
        }

    @staticmethod
    def _mapping_key(value: str) -> str:
        return " ".join(str(value).strip().lower().replace("-", " ").split())

    def _normalize_single_brand(self, value):
        if pd.isna(value):
            return value

        cleaned = " ".join(str(value).strip().split())
        if not cleaned:
            return cleaned

        mapped = self.brand_mapping.get(self._mapping_key(cleaned))
        if mapped:
            return mapped

        # Keep well-known all-uppercase brands readable after scraping.
        upper_brands = {"BMW", "KTM", "CFMOTO", "QJMOTOR", "GASGAS", "KXD", "ASIX"}
        if cleaned.upper() in upper_brands:
            return cleaned.upper()

        return cleaned

    def _normalize_multiword_brands(self, df: pd.DataFrame, brand_column: str, model_column: str) -> pd.DataFrame:
        if model_column not in df.columns:
            return df

        for (first_token, second_token), canonical_brand in self.multiword_brand_prefixes.items():
            brand_matches = df[brand_column].map(self._mapping_key) == first_token
            model_tokens = df[model_column].fillna("").astype(str).str.strip().str.split().str[0].str.lower()
            matches = brand_matches & (model_tokens == second_token)
            if not matches.any():
                continue

            df.loc[matches, brand_column] = canonical_brand
            df.loc[matches, model_column] = (
                df.loc[matches, model_column]
                .astype(str)
                .str.replace(rf"^\s*{second_token}\b\s*", "", case=False, regex=True)
                .str.strip()
            )
            df.loc[matches & (df[model_column] == ""), model_column] = "Unknown"

        return df

    def normalize_brands(self, df: pd.DataFrame, column: str = "brand") -> pd.DataFrame:
        """
        Normalizes brand names based on the mapping.
        """
        if column not in df.columns:
            logger.warning(f"Column {column} missing. Brand normalization skipped.")
            return df.copy()

        df = df.copy()
        df[column] = df[column].apply(self._normalize_single_brand)
        df = self._normalize_multiword_brands(df, column, "model")
        logger.info("Brand names normalized.")
        return df
