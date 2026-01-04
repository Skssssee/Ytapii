import asyncio
import time
import json
import os
import logging
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import yt_dlp
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    StreamingResponse, 
    RedirectResponse, 
    JSONResponse, 
    FileResponse
)
from fastapi.staticfiles import StaticFiles

from config import config
from utils import youtube_utils, rate_limiter, cache

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
_last_request_time = {}
_request_count = {}

# Lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("🚀 YouTube Streaming API Server starting...")
    logger.info(f"📁 Download directory: {config.DOWNLOAD_DIR}")
    logger.info(f"🌐 Server will run on: http://{config.HOST}:{config.PORT}")
    
    if config.COOKIES_FILE and os.path.exists(config.COOKIES_FILE):
        logger.info("🍪 Cookies file detected")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down YouTube Streaming API Server...")

# Create FastAPI app
app = FastAPI(
    title="YouTube Streaming API",
    version="2.0.0",
    description="High-performance YouTube streaming API server",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (if needed)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    if not await rate_limiter.check_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": "Please wait before making another request"
            }
        )
    
    # Process request
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add headers
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-RateLimit-Remaining"] = "unlimited"
    
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response

# YouTube downloader with enhanced options
class YouTubeDownloader:
    @staticmethod
    def get_ydl_options(video_type: str = "video", quality: str = "best"):
        """Get yt-dlp options"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'no_color': True,
            'extract_flat': False,
            'geo_bypass': True,
            'geo_bypass_country': 'IN',  # India
            'socket_timeout': config.YTDLP_TIMEOUT,
            'retries': 3,
            'fragment_retries': 3,
            'skip_unavailable_fragments': True,
            'concurrent_fragment_downloads': 4,  # Parallel downloads
            'throttledratelimit': 1024000,  # 1 MB/s limit
            'verbose': False,
        }
        
        # Add cookies if available
        if config.COOKIES_FILE and os.path.exists(config.COOKIES_FILE):
            ydl_opts['cookiefile'] = config.COOKIES_FILE
        
        # Format selection
        if video_type == "audio":
            ydl_opts['format'] = 'bestaudio/best'
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
        elif video_type == "video":
            if quality == "low":
                ydl_opts['format'] = 'best[height<=360][filesize<20M]/best[height<=240]/worst'
            elif quality == "medium":
                ydl_opts['format'] = 'best[height<=480][filesize<50M]/best[height<=360]/best'
            elif quality == "high":
                ydl_opts['format'] = 'best[height<=720][filesize<100M]/best[height<=480]/best'
            else:  # best
                ydl_opts['format'] = 'best[height<=1080][filesize<200M]/best[height<=720]/best'
        
        # Add proxy if configured
        if config.PROXY:
            ydl_opts['proxy'] = config.PROXY
        
        return ydl_opts
    
    @staticmethod
    async def get_stream_info(url: str, video_type: str = "video", quality: str = "best") -> Dict[str, Any]:
        """Get streaming information for YouTube URL"""
        try:
            video_id = youtube_utils.extract_video_id(url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")
            
            # Check cache first
            cache_key = f"{video_id}:{video_type}:{quality}"
            cached_data = await cache.get(cache_key)
            if cached_data:
                logger.info(f"🎯 Cache hit for {video_id}")
                return cached_data
            
            logger.info(f"🔍 Processing: {video_id} | Type: {video_type} | Quality: {quality}")
            
            ydl_opts = YouTubeDownloader.get_ydl_options(video_type, quality)
            
            # Use yt-dlp to get info
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if not info:
                    raise ValueError("Could not extract video info")
                
                result = {
                    'status': 'success',
                    'video_id': video_id,
                    'title': info.get('title', 'Unknown Title'),
                    'duration': info.get('duration', 0),
                    'thumbnail': info.get('thumbnail', ''),
                    'channel': info.get('channel', 'Unknown Channel'),
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'upload_date': info.get('upload_date', ''),
                }
                
                # Get direct stream URL
                if video_type == "audio":
                    # Find best audio format
                    audio_formats = []
                    for fmt in info.get('formats', []):
                        if fmt.get('acodec') != 'none' and fmt.get('vcodec') == 'none':
                            audio_formats.append(fmt)
                    
                    if audio_formats:
                        # Sort by quality
                        audio_formats.sort(
                            key=lambda x: x.get('abr', 0) or x.get('tbr', 0), 
                            reverse=True
                        )
                        best_audio = audio_formats[0]
                        
                        result.update({
                            'stream_url': best_audio['url'],
                            'type': 'audio',
                            'format': {
                                'ext': best_audio.get('ext', 'mp3'),
                                'abr': best_audio.get('abr', 128),
                                'filesize': best_audio.get('filesize'),
                                'format_note': best_audio.get('format_note', '')
                            }
                        })
                    else:
                        raise ValueError("No audio format found")
                
                elif video_type == "video":
                    # Find best video format
                    video_formats = []
                    for fmt in info.get('formats', []):
                        if fmt.get('vcodec') != 'none' and fmt.get('acodec') != 'none':
                            # Check filesize limit
                            filesize = fmt.get('filesize') or fmt.get('filesize_approx')
                            if filesize and filesize > config.MAX_DOWNLOAD_SIZE:
                                continue
                            video_formats.append(fmt)
                    
                    if video_formats:
                        # Sort by quality
                        video_formats.sort(
                            key=lambda x: (
                                x.get('height', 0) or 0,
                                x.get('width', 0) or 0,
                                x.get('fps', 0) or 0
                            ),
                            reverse=True
                        )
                        
                        best_video = video_formats[0]
                        result.update({
                            'stream_url': best_video['url'],
                            'type': 'video',
                            'format': {
                                'ext': best_video.get('ext', 'mp4'),
                                'height': best_video.get('height'),
                                'width': best_video.get('width'),
                                'fps': best_video.get('fps'),
                                'filesize': best_video.get('filesize'),
                                'format_note': best_video.get('format_note', '')
                            }
                        })
                    else:
                        raise ValueError("No suitable video format found")
                
                # Cache the result
                await cache.set(cache_key, result)
                
                return result
                
        except Exception as e:
            logger.error(f"❌ Error getting stream info: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'video_id': youtube_utils.extract_video_id(url) or 'unknown'
            }

# Create downloader instance
downloader = YouTubeDownloader()

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "YouTube Streaming API",
        "version": "2.0.0",
        "status": "active",
        "timestamp": time.time(),
        "endpoints": {
            "/stream/video": "Stream video (GET, params: url, quality)",
            "/stream/audio": "Stream audio (GET, params: url)",
            "/info": "Get video info (GET, params: url)",
            "/search": "Search videos (GET, params: q, limit)",
            "/formats": "Get available formats (GET, params: url)",
            "/download/video": "Download video (GET, params: url, quality)",
            "/download/audio": "Download audio (GET, params: url)",
            "/health": "Health check (GET)",
            "/stats": "API statistics (GET)"
        },
        "note": "All endpoints support CORS. Use ?url=YOUTUBE_URL parameter."
    }

@app.get("/stream/video")
async def stream_video(
    request: Request,
    url: str = Query(..., description="YouTube video URL"),
    quality: str = Query("best", description="Quality: low, medium, high, best")
):
    """
    Stream YouTube video directly
    Returns redirect to direct stream URL
    """
    # Validate URL
    if not youtube_utils.is_valid_youtube_url(url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    # Validate quality
    if quality not in ["low", "medium", "high", "best"]:
        raise HTTPException(status_code=400, detail="Invalid quality parameter")
    
    try:
        # Get stream info
        result = await downloader.get_stream_info(url, "video", quality)
        
        if result['status'] != 'success':
            raise HTTPException(status_code=500, detail=result.get('message', 'Stream error'))
        
        stream_url = result['stream_url']
        
        logger.info(f"🎬 Streaming video: {result.get('title', 'Unknown')} | Quality: {quality}")
        
        # Return redirect with proper headers
        response = RedirectResponse(url=stream_url, status_code=302)
        
        # Add headers for better streaming
        response.headers.update({
            "Accept-Ranges": "bytes",
            "Content-Type": "video/mp4",
            "Cache-Control": "public, max-age=7200",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Content-Length,Content-Range",
            "X-Video-Title": youtube_utils.clean_title(result.get('title', '')),
            "X-Video-Id": result.get('video_id', ''),
            "X-Stream-Url": stream_url[:100] + "..." if len(stream_url) > 100 else stream_url
        })
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream video error: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")

@app.get("/stream/audio")
async def stream_audio(
    request: Request,
    url: str = Query(..., description="YouTube video URL")
):
    """
    Stream YouTube audio directly
    Returns redirect to direct audio stream URL
    """
    if not youtube_utils.is_valid_youtube_url(url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    try:
        result = await downloader.get_stream_info(url, "audio")
        
        if result['status'] != 'success':
            raise HTTPException(status_code=500, detail=result.get('message', 'Stream error'))
        
        stream_url = result['stream_url']
        
        logger.info(f"🎵 Streaming audio: {result.get('title', 'Unknown')}")
        
        # Return redirect
        response = RedirectResponse(url=stream_url, status_code=302)
        
        # Audio headers
        response.headers.update({
            "Accept-Ranges": "bytes",
            "Content-Type": "audio/mpeg",
            "Cache-Control": "public, max-age=7200",
            "Access-Control-Allow-Origin": "*",
            "X-Audio-Title": youtube_utils.clean_title(result.get('title', '')),
            "X-Audio-Bitrate": str(result.get('format', {}).get('abr', 128)),
            "X-Video-Id": result.get('video_id', '')
        })
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream audio error: {e}")
        raise HTTPException(status_code=500, detail=f"Audio streaming error: {str(e)}")

@app.get("/info")
async def get_video_info(
    url: str = Query(..., description="YouTube video URL")
):
    """Get detailed video information"""
    if not youtube_utils.is_valid_youtube_url(url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    try:
        video_id = youtube_utils.extract_video_id(url)
        
        # Check cache
        cache_key = f"info:{video_id}"
        cached_info = await cache.get(cache_key)
        if cached_info:
            return cached_info
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,
        }
        
        if config.COOKIES_FILE and os.path.exists(config.COOKIES_FILE):
            ydl_opts['cookiefile'] = config.COOKIES_FILE
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            if not info:
                raise HTTPException(status_code=404, detail="Video not found")
            
            # Format response
            video_info = {
                'video_id': video_id,
                'title': info.get('title'),
                'description': info.get('description', '')[:500] + '...' if info.get('description') else '',
                'duration': info.get('duration'),
                'duration_formatted': youtube_utils.format_duration(info.get('duration', 0)),
                'thumbnail': info.get('thumbnail'),
                'channel': info.get('channel'),
                'channel_id': info.get('channel_id'),
                'view_count': info.get('view_count'),
                'like_count': info.get('like_count'),
                'upload_date': info.get('upload_date'),
                'categories': info.get('categories', []),
                'tags': info.get('tags', [])[:10],
                'age_limit': info.get('age_limit', 0),
                'is_live': info.get('is_live', False),
                'formats_count': len(info.get('formats', [])),
                'webpage_url': info.get('webpage_url'),
            }
            
            # Get available formats summary
            formats_summary = []
            for fmt in info.get('formats', []):
                if fmt.get('filesize') or fmt.get('filesize_approx'):
                    formats_summary.append({
                        'format_id': fmt.get('format_id'),
                        'ext': fmt.get('ext'),
                        'resolution': fmt.get('resolution', 'N/A'),
                        'filesize': fmt.get('filesize') or fmt.get('filesize_approx'),
                        'vcodec': fmt.get('vcodec', 'none'),
                        'acodec': fmt.get('acodec', 'none'),
                        'format_note': fmt.get('format_note', '')
                    })
            
            video_info['formats'] = formats_summary[:20]  # Limit to 20 formats
            
            # Cache the info
            await cache.set(cache_key, video_info)
            
            return video_info
            
    except Exception as e:
        logger.error(f"Info error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting info: {str(e)}")

@app.get("/search")
async def search_videos(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Number of results (1-50)")
):
    """Search YouTube videos"""
    if not q or len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="Search query too short")
    
    try:
        # Check cache
        cache_key = f"search:{q}:{limit}"
        cached_results = await cache.get(cache_key)
        if cached_results:
            return cached_results
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'default_search': f'ytsearch{limit}',
            'skip_download': True,
        }
        
        if config.COOKIES_FILE and os.path.exists(config.COOKIES_FILE):
            ydl_opts['cookiefile'] = config.COOKIES_FILE
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{limit}:{q}", download=False)
            
            results = []
            for entry in info.get('entries', []):
                if entry and entry.get('id'):
                    results.append({
                        'video_id': entry.get('id'),
                        'title': entry.get('title', 'No Title'),
                        'duration': entry.get('duration'),
                        'duration_formatted': youtube_utils.format_duration(entry.get('duration', 0)),
                        'thumbnail': entry.get('thumbnail'),
                        'channel': entry.get('channel'),
                        'view_count': entry.get('view_count'),
                        'upload_date': entry.get('upload_date'),
                        'url': f"https://youtube.com/watch?v={entry.get('id')}",
                    })
            
            response = {
                'query': q,
                'count': len(results),
                'results': results[:limit]
            }
            
            # Cache results
            await cache.set(cache_key, response)
            
            return response
            
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/formats")
async def get_available_formats(
    url: str = Query(..., description="YouTube video URL")
):
    """Get all available formats for a video"""
    if not youtube_utils.is_valid_youtube_url(url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    try:
        video_id = youtube_utils.extract_video_id(url)
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'listformats': True,
            'skip_download': True,
        }
        
        if config.COOKIES_FILE and os.path.exists(config.COOKIES_FILE):
            ydl_opts['cookiefile'] = config.COOKIES_FILE
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            formats = []
            for fmt in info.get('formats', []):
                if fmt.get('filesize') or fmt.get('filesize_approx'):
                    formats.append({
                        'format_id': fmt.get('format_id'),
                        'ext': fmt.get('ext'),
                        'resolution': fmt.get('resolution', 'N/A'),
                        'filesize': fmt.get('filesize') or fmt.get('filesize_approx'),
                        'filesize_mb': round((fmt.get('filesize') or fmt.get('filesize_approx') or 0) / (1024 * 1024), 2),
                        'vcodec': fmt.get('vcodec', 'none'),
                        'acodec': fmt.get('acodec', 'none'),
                        'format_note': fmt.get('format_note', ''),
                        'fps': fmt.get('fps'),
                        'tbr': fmt.get('tbr'),  # Average bitrate
                        'protocol': fmt.get('protocol', '')
                    })
            
            # Sort by resolution/filesize
            formats.sort(key=lambda x: (
                x.get('resolution', '0x0'),
                x.get('filesize', 0)
            ), reverse=True)
            
            return {
                'video_id': video_id,
                'title': info.get('title', 'Unknown'),
                'total_formats': len(formats),
                'formats': formats
            }
            
    except Exception as e:
        logger.error(f"Formats error: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting formats: {str(e)}")

@app.get("/download/video")
async def download_video(
    url: str = Query(..., description="YouTube video URL"),
    quality: str = Query("best", description="Quality: low, medium, high, best")
):
    """
    Download video file directly
    Returns the video file as attachment
    """
    if not youtube_utils.is_valid_youtube_url(url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    try:
        result = await downloader.get_stream_info(url, "video", quality)
        
        if result['status'] != 'success':
            raise HTTPException(status_code=500, detail=result.get('message', 'Download error'))
        
        stream_url = result['stream_url']
        video_title = youtube_utils.clean_title(result.get('title', 'video'))
        video_id = result.get('video_id', 'download')
        
        # Download file
        import aiohttp
        
        async def file_generator():
            async with aiohttp.ClientSession() as session:
                async with session.get(stream_url) as response:
                    # Set content length header
                    content_length = response.headers.get('Content-Length')
                    
                    # Stream content
                    async for chunk in response.content.iter_chunked(8192):
                        yield chunk
        
        filename = f"{video_title}_{video_id}.mp4"
        
        return StreamingResponse(
            file_generator(),
            media_type="video/mp4",
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Accept-Ranges': 'bytes',
                'Cache-Control': 'public, max-age=3600',
                'X-Video-Title': video_title,
                'X-Video-Id': video_id
            }
        )
        
    except Exception as e:
        logger.error(f"Download video error: {e}")
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")

@app.get("/download/audio")
async def download_audio(
    url: str = Query(..., description="YouTube video URL")
):
    """Download audio file directly"""
    if not youtube_utils.is_valid_youtube_url(url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    try:
        result = await downloader.get_stream_info(url, "audio")
        
        if result['status'] != 'success':
            raise HTTPException(status_code=500, detail=result.get('message', 'Download error'))
        
        stream_url = result['stream_url']
        audio_title = youtube_utils.clean_title(result.get('title', 'audio'))
        video_id = result.get('video_id', 'download')
        
        import aiohttp
        
        async def file_generator():
            async with aiohttp.ClientSession() as session:
                async with session.get(stream_url) as response:
                    async for chunk in response.content.iter_chunked(8192):
                        yield chunk
        
        filename = f"{audio_title}_{video_id}.mp3"
        
        return StreamingResponse(
            file_generator(),
            media_type="audio/mpeg",
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Accept-Ranges': 'bytes',
                'Cache-Control': 'public, max-age=3600',
                'X-Audio-Title': audio_title,
                'X-Video-Id': video_id
            }
        )
        
    except Exception as e:
        logger.error(f"Download audio error: {e}")
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "YouTube Streaming API",
        "version": "2.0.0",
        "cache_size": len(cache.cache),
        "rate_limits": len(rate_limiter.requests)
    }

@app.get("/stats")
async def api_statistics():
    """API statistics"""
    return {
        "total_requests": sum(_request_count.values()),
        "requests_by_endpoint": _request_count,
        "cache_hits": getattr(cache, 'hits', 0),
        "cache_misses": getattr(cache, 'misses', 0),
        "cache_size": len(cache.cache),
        "uptime": time.time() - getattr(app, 'start_time', time.time()),
        "rate_limited_ips": len(rate_limiter.requests)
    }

@app.get("/clear-cache")
async def clear_cache():
    """Clear all cache (admin endpoint)"""
    cache.cache.clear()
    return {"status": "success", "message": "Cache cleared"}

# WebSocket endpoint for real-time updates
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            
            # Handle different commands
            if data == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})
            elif data.startswith("info:"):
                video_url = data[5:]
                if youtube_utils.is_valid_youtube_url(video_url):
                    result = await downloader.get_stream_info(video_url, "video")
                    await websocket.send_json(result)
                else:
                    await websocket.send_json({"error": "Invalid URL"})
            else:
                await websocket.send_json({"type": "message", "text": f"Received: {data}"})
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "path": request.url.path
        }
    )

if __name__ == "__main__":
    # Record startup time
    app.start_time = time.time()
    
    # Print startup banner
    print("=" * 60)
    print("🎬 YOUTUBE STREAMING API SERVER v2.0.0")
    print("=" * 60)
    print(f"📁 Download directory: {config.DOWNLOAD_DIR}")
    print(f"🌐 Server URL: http://{config.HOST}:{config.PORT}")
    print(f"📚 Documentation: http://{config.HOST}:{config.PORT}/docs")
    print(f"📊 Health check: http://{config.HOST}:{config.PORT}/health")
    print("=" * 60)
    
    if config.COOKIES_FILE and os.path.exists(config.COOKIES_FILE):
        print("🍪 Cookies file: DETECTED")
    else:
        print("⚠️  Cookies file: NOT DETECTED (age-restricted videos may not work)")
    
    print("🚀 Starting server...")
    print("=" * 60)
    
    # Start server
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        access_log=True,
        timeout_keep_alive=30
    )