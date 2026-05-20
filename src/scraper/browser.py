from playwright.sync_api import sync_playwright, Browser, Page
from typing import Optional
from src.utils.logger import setup_logger

logger = setup_logger("browser")

class OtoMotoBrowser:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context = None

    def start(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        logger.info("Browser started successfully.")

    def get_page(self) -> Page:
        if not self.context:
            raise RuntimeError("Browser context not initialized. Call start() first.")
        return self.context.new_page()

    def stop(self):
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        logger.info("Browser stopped.")
