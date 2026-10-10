import os
import sys
import logging
import json
import pandas as pd
from typing import Optional

def setup_logger(name: str = "AlphaReq", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a standardized, color-safe logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def safe_save_dataframe(df: pd.DataFrame, target_path: str, fallback_dir: Optional[str] = None) -> str:
    """
    Safely saves a pandas DataFrame to CSV, handling OS/OneDrive locks gracefully
    by falling back to a local temporary directory if needed.
    """
    logger = logging.getLogger("AlphaReq")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        df.to_csv(target_path, index=False)
        return target_path
    except (OSError, PermissionError) as e:
        logger.warning(f"Direct write to {target_path} failed ({e}). Attempting fallback directory...")
        fallback = fallback_dir or os.environ.get("TEMP", r"C:\Users\umert\AppData\Local\Temp\opencode")
        os.makedirs(fallback, exist_ok=True)
        alt_path = os.path.join(fallback, os.path.basename(target_path))
        df.to_csv(alt_path, index=False)
        logger.info(f"Successfully saved to fallback location: {alt_path}")
        return alt_path

def safe_save_json(data: dict, target_path: str, fallback_dir: Optional[str] = None) -> str:
    """
    Safely saves a dictionary to formatted JSON with fallback handling.
    """
    logger = logging.getLogger("AlphaReq")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        return target_path
    except (OSError, PermissionError) as e:
        logger.warning(f"Direct write to {target_path} failed ({e}). Attempting fallback...")
        fallback = fallback_dir or os.environ.get("TEMP", r"C:\Users\umert\AppData\Local\Temp\opencode")
        os.makedirs(fallback, exist_ok=True)
        alt_path = os.path.join(fallback, os.path.basename(target_path))
        with open(alt_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Successfully saved to fallback location: {alt_path}")
        return alt_path
