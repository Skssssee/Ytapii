import os
from typing import Optional

class Config:
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Rate limiting
    RATE_LIMIT_WINDOW: int = 2  # Seconds between requests
    MAX_REQUESTS_PER_MINUTE: int = 30
    
    # YouTube settings
    YTDLP_TIMEOUT: int = 30  # Seconds
    COOKIES_FILE: Optional[str] = "cookies.txt" if os.path.exists("cookies.txt") else None
    
    # Cache settings
    CACHE_TTL: int = 3600  # 1 hour
    MAX_CACHE_SIZE: int = 1000
    
    # Proxy settings (if needed)
    PROXY: Optional[str] = None
    
    # Download settings
    DOWNLOAD_DIR: str = "downloads"
    MAX_DOWNLOAD_SIZE: int = 500 * 1024 * 1024  # 500 MB
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "streaming_api.log"

config = Config()

# Create necessary directories
os.makedirs(config.DOWNLOAD_DIR, exist_ok=True)