import pandas as pd
import numpy as np
from src.cleaning.normalizer import DataNormalizer
from src.cleaning.filter import DataFilter

def test_normalize_brands():
    df = pd.DataFrame({'brand': ['Yamaha Motor', 'honda motor', 'Bmw', 'Unknown Brand']})
    normalizer = DataNormalizer()
    result = normalizer.normalize_brands(df)
    assert result['brand'].iloc[0] == 'Yamaha'
    assert result['brand'].iloc[1] == 'Honda'
    assert result['brand'].iloc[2] == 'BMW'
    assert result['brand'].iloc[3] == 'Unknown Brand'

def test_normalize_multiword_brands_from_title_split():
    df = pd.DataFrame({
        'brand': ['Moto', 'Royal', 'MV', 'CF', 'Harley'],
        'model': ['Guzzi V7', 'Enfield Classic 350', 'Agusta Brutale', 'Moto 700MT', 'Davidson Sportster']
    })
    normalizer = DataNormalizer()
    result = normalizer.normalize_brands(df)

    assert result['brand'].tolist() == ['Moto Guzzi', 'Royal Enfield', 'MV Agusta', 'CFMOTO', 'Harley-Davidson']
    assert result['model'].tolist() == ['V7', 'Classic 350', 'Brutale', '700MT', 'Sportster']

def test_remove_duplicates():
    df = pd.DataFrame({'a': [1, 1, 2], 'b': [3, 3, 4]})
    filter_tool = DataFilter()
    result = filter_tool.remove_duplicates(df)
    assert len(result) == 2

def test_remove_incomplete():
    df = pd.DataFrame({
        'brand': ['A', 'B', 'C'],
        'price': [100, np.nan, 300],
        'year': [2020, 2021, 0]
    })
    filter_tool = DataFilter()
    result = filter_tool.remove_incomplete(df)
    # Row 1 has NaN price, Row 2 has 0 year
    assert len(result) == 1
    assert result['brand'].iloc[0] == 'A'

def test_remove_outliers_iqr():
    df = pd.DataFrame({'price': [100, 110, 120, 10000]}) # 10000 is a clear outlier
    filter_tool = DataFilter()
    result = filter_tool.remove_outliers_iqr(df, ['price'])
    assert len(result) == 3
    assert 10000 not in result['price'].values
