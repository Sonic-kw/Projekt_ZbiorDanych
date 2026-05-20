import pandas as pd
from src.analysis.regression import PricePredictor
from src.analysis.clustering import MarketSegmenter

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
