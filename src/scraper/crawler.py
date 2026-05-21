import pandas as pd
from pathlib import Path
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

    @staticmethod
    def _ensure_output_dir(output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

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
                    debug_file = Path("results/debug") / f"debug_page_{page_num}.html"
                    self._ensure_output_dir(str(debug_file))
                    debug_file.write_text(page.content(), encoding="utf-8")
                    logger.debug(f"Page HTML saved to {debug_file}")
                
                logger.info(f"No listings found on page {page_num}. Reached the end of results.")
                break

            all_listings.extend(listings)
                
        self.browser.stop()
        
        df = pd.DataFrame(all_listings)
        logger.info(f"Successfully scraped {len(df)} listings.")
        return df

    def save_to_csv(self, df: pd.DataFrame, path: str):
        self._ensure_output_dir(path)
        df.to_csv(path, index=False)
        logger.info(f"Data saved to {path}")
