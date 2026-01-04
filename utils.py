import re
import time
import hashlib
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse, parse_qs
import aiohttp
from datetime import datetime, timedelta

from config import config

class YouTubeUtils:
    @staticmethod
    def is_valid_youtube_url(url: str) -> bool:
        """Validate YouTube URL"""
        patterns = [
            r'^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$',
            r'^https?://(www\.)?youtube\.com/watch\?v=[\w-]{11}',
            r'^https?://(www\.)?youtube\.com/embed/[\w-]{11}',
            r'^https?://(www\.)?youtube\.com/shorts/[\w-]{11}',
            r'^https?://(www\.)?youtube\.com/live/[\w-]{11}',
            r'^https?://youtu\.be/[\w-]{11}'
        ]
        
        for pattern in patterns:
            if re.match(pattern, url, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract video ID from URL"""
        # Common patterns
        patterns = {
            'v': r'(?:v=|\/)([\w-]{11})',
            'embed': r'embed\/([\w-]{11})',
            'shorts': r'shorts\/([\w-]{11})',
            'live': r'live\/([\w-]{11})',
            'youtu.be': r'youtu\.be\/([\w-]{11})'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # Try parsing query parameters
        parsed = urlparse(url)
        if parsed.hostname and 'youtube.com' in parsed.hostname:
            query_params = parse_qs(parsed.query)
            if 'v' in query_params and query_params['v'][0]:
                return query_params['v'][0]
        
        return None
    
    @staticmethod
    def clean_title(title: str) -> str:
        """Clean video title for filename"""
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            title = title.replace(char, '')
        
        # Limit length
        if len(title) > 100:
            title = title[:97] + '...'
        
        return title.strip()
    
    @staticmethod
    def format_duration(seconds: int) -> str:
        """Format duration in seconds to HH:MM:SS"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"

class RateLimiter:
    """Simple rate limiter"""
    
    def __init__(self):
        self.requests = {}
        self.lock = asyncio.Lock()
    
    async def check_limit(self, client_ip: str) -> bool:
        """Check if client is within rate limits"""
        async with self.lock:
            current_time = time.time()
            
            # Clean old entries
            self.requests = {
                ip: ts for ip, ts in self.requests.items()
                if current_time - ts < config.RATE_LIMIT_WINDOW
            }
            
            # Check limit
            if client_ip in self.requests:
                return False
            
            # Update timestamp
            self.requests[client_ip] = current_time
            return True

class Cache:
    """Simple in-memory cache"""
    
    def __init__(self):
        self.cache = {}
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self.lock:
            if key in self.cache:
                timestamp, value = self.cache[key]
                if time.time() - timestamp < config.CACHE_TTL:
                    return value
                else:
                    del self.cache[key]
            return None
    
    async def set(self, key: str, value: Any):
        """Set value in cache"""
        async with self.lock:
            # Remove oldest if cache is full
            if len(self.cache) >= config.MAX_CACHE_SIZE:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
                del self.cache[oldest_key]
            
            self.cache[key] = (time.time(), value)
    
    async def delete(self, key: str):
        """Delete key from cache"""
        async with self.lock:
            if key in self.cache:
                del self.cache[key]

# Global instances
rate_limiter = RateLimiter()
cache = Cache()
youtube_utils = YouTubeUtils()