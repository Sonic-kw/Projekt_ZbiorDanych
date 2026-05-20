import logging
import sys
import os

from src.utils.config_loader import load_config

def setup_logger(name: str = "otomoto_analysis", debug: bool = False):
    """
    Sets up a logger that outputs to both console and a file.
    """
    # Load debug flag from config
    config = load_config()
    debug = config.get('debug', False)

    level = logging.DEBUG if debug else logging.INFO
    
    # Set root logger level to ensure all loggers inherit it
    logging.getLogger().setLevel(level)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
 
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
 
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
 
        # File handler - save to results folder
        os.makedirs("results", exist_ok=True)
        file_handler = logging.FileHandler("results/analysis.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
 
    return logger

