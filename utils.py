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
    
    @staticmethod
    def search_youtube_sync(query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Synchronous YouTube search using youtubesearchpython
        Returns list of video dictionaries
        """
        try:
            from youtubesearchpython import VideosSearch
            
            search = VideosSearch(query, limit=limit)
            data = search.result()

            if not data or "result" not in data:
                return []

            videos = []
            for v in data["result"]:
                # Extract video ID from link
                video_id = None
                if v.get("link"):
                    video_id = YouTubeUtils.extract_video_id(v["link"])
                
                videos.append({
                    "video_id": video_id,
                    "title": v.get("title", "No Title"),
                    "url": v.get("link", ""),
                    "duration": v.get("duration"),
                    "duration_formatted": YouTubeUtils.format_duration(
                        YouTubeUtils.parse_duration_string(v.get("duration", "0:00"))
                    ),
                    "thumbnail": v.get("thumbnails", [{}])[0].get("url") if v.get("thumbnails") else "",
                    "channel": v.get("channel", {}).get("name", "Unknown Channel"),
                    "view_count": v.get("viewCount", {}).get("short", "0 views") if isinstance(v.get("viewCount"), dict) else "0 views",
                    "upload_date": v.get("publishedTime", "Unknown"),
                })

            return videos

        except ImportError:
            print("Error: youtubesearchpython not installed. Install with: pip install youtubesearchpython")
            return []
        except Exception as e:
            print(f"YouTube search error: {e}")
            return []
    
    @staticmethod
    async def search_youtube_async(query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Asynchronous wrapper for YouTube search
        """
        # Run synchronous search in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            YouTubeUtils.search_youtube_sync, 
            query, limit
        )
    
    @staticmethod
    def parse_duration_string(duration_str: str) -> int:
        """
        Parse duration string (e.g., "1:23:45" or "5:30") to seconds
        """
        if not duration_str:
            return 0
        
        try:
            parts = duration_str.split(':')
            if len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:  # MM:SS
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            elif len(parts) == 1:  # SS
                return int(parts[0])
            else:
                return 0
        except:
            return 0

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
        self.hits = 0
        self.misses = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self.lock:
            if key in self.cache:
                timestamp, value = self.cache[key]
                if time.time() - timestamp < config.CACHE_TTL:
                    self.hits += 1
                    return value
                else:
                    del self.cache[key]
                    self.misses += 1
            else:
                self.misses += 1
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

# Convenience function for backward compatibility
def youtube_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Convenience function for synchronous YouTube search
    """
    return youtube_utils.search_youtube_sync(query, limit)