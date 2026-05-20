import pandas as pd
from src.scraper.crawler import OtoMotoCrawler
from src.cleaning.normalizer import DataNormalizer
from src.cleaning.filter import DataFilter
from src.analysis.statistics import MarketAnalysis
from src.analysis.regression import PricePredictor
from src.analysis.clustering import MarketSegmenter
from src.analysis.bargain_finder import BargainFinder
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def main():
    # 1. Load Configuration
    config = load_config()
    
    # Initialize Logger with debug flag from config
    logger = setup_logger("main")
    
    # 2. Scraping
    logger.info("Starting scraping process...")
    crawler = OtoMotoCrawler()
    raw_df = crawler.scrape()
    crawler.save_to_csv(raw_df, config['paths']['raw_data'])
    
    # 3. Cleaning
    logger.info("Starting data cleaning...")
    normalizer = DataNormalizer()
    df = normalizer.normalize_brands(raw_df)
    
    data_filter = DataFilter()
    df = data_filter.remove_duplicates(df)
    df = data_filter.remove_incomplete(df, config['cleaning']['required_columns'])
    df = data_filter.remove_outliers_iqr(
        df, 
        columns=['price', 'mileage', 'year'], 
        threshold=config['cleaning']['outlier_threshold']
    )
    
    # Save processed data
    df.to_csv(config['paths']['processed_data'], index=False)
    logger.info(f"Processed data saved to {config['paths']['processed_data']}")
    
    # 4. Analysis
    logger.info("Starting market analysis...")
    
    # Statistics & Visualizations
    stats = MarketAnalysis()
    shares = stats.calculate_market_shares(df)
    logger.info(f"Market Shares:\n{shares}")
    
    # Advanced Visualizations
    stats.plot_market_structure(df)
    stats.plot_distributions(df)
    stats.plot_regression_analysis(df, x_col='year')
    stats.plot_regression_analysis(df, x_col='mileage')
    stats.plot_correlation_matrix(df)
    
    # Clustering
    segmenter = MarketSegmenter(n_clusters=config['analysis']['clustering_clusters'])
    df['cluster'] = segmenter.fit_predict(df)
    stats.plot_clusters(df)
    
    # Bargain Finding
    finder = BargainFinder(threshold=config['analysis']['bargain_threshold'])
    bargains = finder.find_bargains(df)
    
    logger.info(f"Analysis complete. Found {len(bargains)} bargains.")
    logger.info("Bargains preview:\n" + str(bargains.head()))

if __name__ == "__main__":
    main()
