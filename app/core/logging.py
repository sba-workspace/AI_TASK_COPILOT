"""
Logging configuration for the application.
"""
import sys
from pathlib import Path

from loguru import logger as loguru_logger


# Configure Loguru logger
def setup_logger():
    """Configure the logger with custom format and level."""
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    # Remove default handler
    loguru_logger.remove()
    
    # Add custom handlers
    loguru_logger.add(
        sys.stderr,
        format=log_format,
        level="INFO",
        colorize=True,
    )
    
    # Optionally add file logging in production
    log_path = Path("logs")
    if not log_path.exists():
        log_path.mkdir(parents=True, exist_ok=True)
    
    loguru_logger.add(
        log_path / "app.log",
        format=log_format,
        level="INFO",
        rotation="10 MB",
        compression="zip",
        retention="1 month",
    )
    
    return loguru_logger


# Initialize logger
logger = setup_logger()