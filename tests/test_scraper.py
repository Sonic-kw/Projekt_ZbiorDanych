import pytest
import pandas as pd
from src.scraper.crawler import OtoMotoCrawler

@pytest.mark.integration
def test_scraper_honda_url():
    """
    Tests if the scraper can successfully extract data from the Honda motorcycles page.
    """
    crawler = OtoMotoCrawler()
    crawler.config['scraper']['start_url'] = "https://www.otomoto.pl/motocykle-i-quady/honda"
    crawler.config['scraper']['max_pages'] = 1
    
    try:
        df = crawler.scrape()
       
        assert isinstance(df, pd.DataFrame), "Scraper should return a pandas DataFrame"
        assert not df.empty, "DataFrame should not be empty for a valid URL"
        assert 'brand' in df.columns, "DataFrame should contain 'brand' column"
        assert 'price' in df.columns, "DataFrame should contain 'price' column"
        
        honda_count = len(df[df['brand'].str.contains('Honda', case=False, na=False)])
        print(f"Found {honda_count} Honda motorcycles.")
        
    finally:
        try:
            crawler.browser.stop()
        except:
            pass
