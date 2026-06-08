import os
import logging
from datetime import datetime
from src.config import LOGS_DIR

def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a standard logger that outputs to both console and a log file.
    """
    logger = logging.getLogger(name)
    
    # If the logger already has handlers, it was already configured
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    log_file = os.path.join(LOGS_DIR, f"execution_{datetime.now().strftime('%Y-%m-%d')}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger
