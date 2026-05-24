from playwright.sync_api import Page
from typing import List, Dict, Any
import re
from urllib.parse import urljoin
from src.utils.logger import setup_logger

logger = setup_logger("parser")

class OtoMotoParser:
    BASE_URL = "https://www.otomoto.pl"

    @staticmethod
    def parse_numeric(value_str: str) -> float:
        """Extracts numeric value from strings like '12 000 km' or '600 cm3'."""
        if not value_str:
            return 0.0
        cleaned = re.sub(r'[^\d]', '', value_str)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def parse_listings(self, page: Page) -> List[Dict[str, Any]]:
        """
        Parses motorcycle listings from the current page.
        """
        listings = []
        items = page.query_selector_all('article.ooa-zet1mn')
        logger.debug(f"Found {len(items)} potential listing items on the page.")
        
        for item in items:
            try:
                title_el = item.query_selector('h2.e123dwbo0 a')
                brand_model = title_el.inner_text().strip() if title_el else ""
                listing_href = title_el.get_attribute("href") if title_el else None
                listing_url = urljoin(self.BASE_URL, listing_href) if listing_href else ""
                listing_id_match = re.search(r"ID[0-9A-Za-z]+", listing_url)
                listing_id = listing_id_match.group(0) if listing_id_match else ""
                
                price_el = item.query_selector('h3.eg88ra81')
                price_text = price_el.inner_text().strip() if price_el else ""
                
                mileage_el = item.query_selector('dd[data-parameter="mileage"]')
                year_el = item.query_selector('dd[data-parameter="year"]')
                capacity_el = item.query_selector('dd[data-parameter="engine_capacity"]')
                
                mileage_text = mileage_el.inner_text().strip() if mileage_el else ""
                year_text = year_el.inner_text().strip() if year_el else ""
                capacity_text = capacity_el.inner_text().strip() if capacity_el else ""
                
                summary_el = item.query_selector('p.e1kj25my0')
                summary_text = summary_el.inner_text().strip() if summary_el else ""
                
                power = 0.0
                power_match = re.search(r'(\d+)\s*KM', summary_text)
                if power_match:
                    power = self.parse_numeric(power_match.group(1))
                
                parts = brand_model.split(' ', 1)
                brand = parts[0] if len(parts) > 0 else "Unknown"
                model = parts[1] if len(parts) > 1 else "Unknown"
                
                logger.debug(f"Found: {brand} {model} | Price: {price_text} | Year: {year_text}")

                listings.append({
                    "brand": brand,
                    "model": model,
                    "year": self.parse_numeric(year_text),
                    "mileage": self.parse_numeric(mileage_text),
                    "capacity": self.parse_numeric(capacity_text),
                    "power": power,
                    "price": self.parse_numeric(price_text),
                    "type": "Motorcycle",
                    "listing_id": listing_id,
                    "listing_url": listing_url,
                })
            except Exception as e:
                logger.error(f"Error parsing listing item: {e}")
                continue
                
        return listings
