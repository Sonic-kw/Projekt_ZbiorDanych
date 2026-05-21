import pandas as pd
from src.analysis.regression import PricePredictor
from src.analysis.clustering import MarketSegmenter
from src.analysis.statistics import MarketAnalysis
from src.analysis.bargain_finder import BargainFinder

def test_price_predictor():
    # Create a linear relationship: price = 1000 * year - 2 * mileage
    df = pd.DataFrame({
        'year': [2010, 2015, 2020, 2025],
        'mileage': [50000, 30000, 10000, 1000],
        'price': [10000, 20000, 30000, 40000]
    })
    predictor = PricePredictor()
    mae, r2 = predictor.train(df)
    
    # Check if model can predict
    test_df = pd.DataFrame({'year': [2018], 'mileage': [20000]})
    prediction = predictor.predict(test_df)
    assert len(prediction) == 1
    assert prediction[0] > 0

def test_market_segmenter():
    df = pd.DataFrame({
        'price': [10000, 50000, 100000],
        'year': [2010, 2015, 2020],
        'mileage': [50000, 20000, 5000]
    })
    segmenter = MarketSegmenter(n_clusters=2)
    clusters = segmenter.fit_predict(df)
    assert len(clusters) == 3
    assert set(clusters).issubset({0, 1})

def test_percentile_summary_includes_numeric_columns():
    df = pd.DataFrame({
        'brand': ['Honda', 'Yamaha'],
        'price': [10000, 20000],
        'mileage': [5000, 15000],
        'year': [2018, 2020],
    })
    summary = MarketAnalysis.calculate_percentile_summary(df)

    assert 'price' in summary.index
    assert summary.loc['price', 'p50'] == 15000
    assert summary.loc['mileage', 'missing_pct'] == 0

def test_localize_columns_for_report_exports():
    df = pd.DataFrame({
        'brand': ['Honda'],
        'price_delta_pct': [-25.0],
        'mileage_delta_pct': [-30.0],
    })
    localized = MarketAnalysis.localize_columns(df)

    assert 'Marka' in localized.columns
    assert 'Cena względem średniej (%)' in localized.columns
    assert 'Przebieg względem średniej (%)' in localized.columns

def test_brand_regression_coefficients():
    df = pd.DataFrame({
        'brand': ['Honda'] * 8 + ['Yamaha'] * 8,
        'year': [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023] * 2,
        'mileage': [8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000] * 2,
        'price': [10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000] * 2,
    })
    coefficients = MarketAnalysis.calculate_brand_regression_coefficients(df, min_records=4)

    assert set(coefficients['brand']) == {'Honda', 'Yamaha'}
    assert (coefficients['price_year_slope'] > 0).all()
    assert (coefficients['usage_km_per_year'] > 0).all()

def test_bargain_finder_uses_peer_price_and_mileage():
    df = pd.DataFrame({
        'brand': ['Honda'] * 4,
        'model': ['CB500'] * 4,
        'year': [2020] * 4,
        'mileage': [10000, 11000, 9000, 4000],
        'price': [20000, 21000, 19000, 12000],
    })
    finder = BargainFinder(threshold=0.2, min_group_size=3)
    bargains = finder.find_bargains(df)

    assert len(bargains) == 1
    assert bargains.iloc[0]['price'] == 12000
    assert bargains.iloc[0]['mileage_delta_pct'] < -20
