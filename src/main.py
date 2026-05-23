import pandas as pd
from pathlib import Path
from src.scraper.crawler import OtoMotoCrawler
from src.cleaning.normalizer import DataNormalizer
from src.cleaning.filter import DataFilter
from src.analysis.statistics import MarketAnalysis
from src.analysis.clustering import MarketSegmenter
from src.analysis.bargain_finder import BargainFinder
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


VALID_DATA_SOURCES = {"scrape", "raw", "processed"}


def _get_data_source(config: dict) -> str:
    data_source = config.get("pipeline", {}).get("data_source", "scrape").lower()
    if data_source not in VALID_DATA_SOURCES:
        valid_values = ", ".join(sorted(VALID_DATA_SOURCES))
        raise ValueError(f"Unsupported pipeline.data_source='{data_source}'. Use one of: {valid_values}.")
    return data_source


def _read_existing_data(path: str, dataset_name: str) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot use existing {dataset_name} data. File not found: {path}")
    return pd.read_csv(data_path)


def get_data_source(config: dict) -> str:
    """Public wrapper used by tests and CLI integrations."""
    return _get_data_source(config)


def read_existing_data(path: str, dataset_name: str) -> pd.DataFrame:
    """Public wrapper used by tests and CLI integrations."""
    return _read_existing_data(path, dataset_name)


def _clean_raw_data(raw_df: pd.DataFrame, config: dict, logger) -> pd.DataFrame:
    logger.info("Starting data cleaning...")
    normalizer = DataNormalizer()
    df = normalizer.normalize_brands(raw_df)

    data_filter = DataFilter()
    df = data_filter.remove_duplicates(df)
    df = data_filter.remove_incomplete(df)
    df = data_filter.remove_outliers_iqr(
        df,
        columns=['price', 'mileage', 'year'],
        threshold=config['cleaning']['outlier_threshold']
    )

    Path(config['paths']['processed_data']).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config['paths']['processed_data'], index=False)
    logger.info(f"Processed data saved to {config['paths']['processed_data']}")
    return df


def prepare_dataset(config: dict, logger) -> pd.DataFrame:
    data_source = _get_data_source(config)

    if data_source == "processed":
        logger.info(f"Loading existing processed data from {config['paths']['processed_data']}.")
        return _read_existing_data(config['paths']['processed_data'], "processed")

    if data_source == "raw":
        logger.info(f"Loading existing raw data from {config['paths']['raw_data']}.")
        raw_df = _read_existing_data(config['paths']['raw_data'], "raw")
        return _clean_raw_data(raw_df, config, logger)

    logger.info("Starting scraping process...")
    crawler = OtoMotoCrawler()
    raw_df = crawler.scrape()
    crawler.save_to_csv(raw_df, config['paths']['raw_data'])
    return _clean_raw_data(raw_df, config, logger)


def main():
    config = load_config()
    logger = setup_logger("main")

    df = prepare_dataset(config, logger)

    logger.info("Starting market analysis...")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    stats = MarketAnalysis()
    shares = stats.calculate_market_shares(df)
    logger.info("Market share summary prepared.")
    logger.debug(f"Market share summary:\n{shares}")
    percentiles = stats.calculate_percentile_summary(df)
    stats.localize_columns(percentiles).to_csv(results_dir / "percentyle_numeryczne.csv")
    logger.info("Numeric percentile summary prepared.")
    logger.debug(f"Numeric percentile summary:\n{percentiles}")

    stats.plot_market_structure(df)
    stats.plot_numeric_diagnostics(df)
    stats.plot_distributions(df)
    stats.plot_brand_distributions(df)
    stats.plot_year_brand_heatmap(df)
    stats.plot_model_year_heatmap(df)
    stats.plot_regression_analysis(df, x_col='year')
    stats.plot_regression_analysis(df, x_col='mileage')
    stats.plot_relationship_facets(df, x_col='year')
    stats.plot_relationship_facets(df, x_col='mileage')
    stats.plot_correlation_matrix(df)
    brand_coefficients = stats.plot_depreciation_exploitation(
        df,
        n_clusters=config['analysis']['clustering_clusters']
    )
    stats.localize_columns(brand_coefficients).to_csv(results_dir / "wspolczynniki_marek.csv", index=False)

    segmenter = MarketSegmenter(
        n_clusters=config['analysis']['clustering_clusters'],
        algorithm=config['analysis'].get('clustering_algorithm', 'auto'),
        dbscan_eps=config['analysis'].get('dbscan_eps', 0.7),
        dbscan_min_samples=config['analysis'].get('dbscan_min_samples', 12),
    )
    df['cluster'] = segmenter.fit_predict(df)
    clustering_metrics = pd.DataFrame([segmenter.last_metrics])
    stats.localize_columns(clustering_metrics).to_csv(results_dir / "jakosc_klastrow.csv", index=False)
    stats.plot_clusters(df, clustering_method=segmenter.last_metrics.get("selected_algorithm", "kmeans"))

    finder = BargainFinder(threshold=config['analysis']['bargain_threshold'])
    bargains = finder.find_bargains(df)
    stats.localize_columns(bargains).to_csv(results_dir / "okazje_rynkowe.csv", index=False)
    stats.plot_bargain_candidates(bargains)

    logger.info(f"Analysis complete. Found {len(bargains)} potential bargains.")
    logger.debug("Bargains preview:\n" + str(bargains.head()))

if __name__ == "__main__":
    main()
