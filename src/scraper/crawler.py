import pandas as pd
from typing import List
from src.scraper.browser import OtoMotoBrowser
from src.scraper.parser import OtoMotoParser
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = setup_logger("crawler")

class OtoMotoCrawler:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = load_config(config_path)
        self.browser = OtoMotoBrowser(headless=True)
        self.parser = OtoMotoParser()

    def scrape(self) -> pd.DataFrame:
        self.browser.start()
        page = self.browser.get_page()
        
        start_url = self.config['scraper']['start_url']
        max_pages = self.config['scraper']['max_pages']
        
        all_listings = []
        for page_num in range(1, max_pages + 1):
            current_url = start_url
            if page_num > 1:
                separator = '&' if '?' in start_url else '?'
                current_url = f"{start_url}{separator}page={page_num}"

            logger.info(f"Scraping page {page_num}: {current_url}")
            logger.debug(f"Navigating to {current_url}...")
            page.goto(current_url, wait_until="networkidle")
            
            # Handle potential cookie consent
            try:
                consent_button = page.query_selector('button:has-text("Przejdź do serwisu")')
                if consent_button:
                    consent_button.click()
            except Exception:
                pass

            listings = self.parser.parse_listings(page)
            
            if not listings:
                if self.config.get('debug', False):
                    logger.warning(f"No listings found on page {page_num}. Saving HTML for diagnostics...")
                    import os
                    os.makedirs("data/raw", exist_ok=True)
                    with open(f"data/raw/debug_page_{page_num}.html", "w", encoding="utf-8") as f:
                        f.write(page.content())
                    logger.debug(f"Page HTML saved to data/raw/debug_page_{page_num}.html")
                
                logger.info(f"No listings found on page {page_num}. Reached the end of results.")
                break

            # Print a piece of the website in logs if debug is on
            if self.config.get('debug', False):
                html_snippet = page.content()[:1000] # First 1000 chars
                logger.debug(f"Page HTML snippet:\n{html_snippet}...")

            all_listings.extend(listings)
                
        self.browser.stop()
        
        df = pd.DataFrame(all_listings)
        logger.info(f"Successfully scraped {len(df)} listings.")
        return df

    def save_to_csv(self, df: pd.DataFrame, path: str):
        df.to_csv(path, index=False)
        logger.info(f"Data saved to {path}")
