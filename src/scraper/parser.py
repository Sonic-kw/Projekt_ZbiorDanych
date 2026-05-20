from playwright.sync_api import Page
from typing import List, Dict, Any
import re
from src.utils.logger import setup_logger

logger = setup_logger("parser")

class OtoMotoParser:
    @staticmethod
    def parse_price(price_str: str) -> float:
        """Extracts numeric price from string like '15 000 zł'."""
        if not price_str:
            return 0.0
        # Remove non-digit characters except for potential decimal separators
        cleaned = re.sub(r'[^\d]', '', price_str)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

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
        # Target the listing containers
        items = page.query_selector_all('article.ooa-zet1mn')
        logger.debug(f"Found {len(items)} potential listing items on the page.")
        
        for item in items:
            try:
                # Title/Brand/Model
                title_el = item.query_selector('h2.e123dwbo0 a')
                brand_model = title_el.inner_text().strip() if title_el else ""
                
                # Price
                price_el = item.query_selector('h3.eg88ra81')
                price_text = price_el.inner_text().strip() if price_el else ""
                
                # Specs from the dl list
                mileage_el = item.query_selector('dd[data-parameter="mileage"]')
                year_el = item.query_selector('dd[data-parameter="year"]')
                capacity_el = item.query_selector('dd[data-parameter="engine_capacity"]')
                
                mileage_text = mileage_el.inner_text().strip() if mileage_el else ""
                year_text = year_el.inner_text().strip() if year_el else ""
                capacity_text = capacity_el.inner_text().strip() if capacity_el else ""
                
                # Power is often in the summary text: <p class="e1kj25my0 ooa-nxfgg7">
                summary_el = item.query_selector('p.e1kj25my0')
                summary_text = summary_el.inner_text().strip() if summary_el else ""
                
                # Extract power from summary (e.g., "279 cm3 • 26 KM • ...")
                power = 0.0
                power_match = re.search(r'(\d+)\s*KM', summary_text)
                if power_match:
                    power = self.parse_numeric(power_match.group(1))
                
                # Simple split for brand/model
                parts = brand_model.split(' ', 1)
                brand = parts[0] if len(parts) > 0 else "Unknown"
                model = parts[1] if len(parts) > 1 else "Unknown"
                
                # Log found motorcycle only in DEBUG mode
                logger.debug(f"Found: {brand} {model} | Price: {price_text} | Year: {year_text}")

                listings.append({
                    "brand": brand,
                    "model": model,
                    "year": self.parse_numeric(year_text),
                    "mileage": self.parse_numeric(mileage_text),
                    "capacity": self.parse_numeric(capacity_text),
                    "power": power,
                    "price": self.parse_price(price_text),
                    "type": "Motorcycle"
                })
            except Exception as e:
                logger.error(f"Error parsing listing item: {e}")
                continue
                
        return listings
