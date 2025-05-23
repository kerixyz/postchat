# utils/youtube_api.py

import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

def get_video_metadata(video_id):
    url = f"https://www.googleapis.com/youtube/v3/videos"
    params = {
        "id": video_id,
        "key": API_KEY,
        "part": "snippet,contentDetails,statistics"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    items = data.get("items", [])
    if not items:
        return None

    item = items[0]
    snippet = item["snippet"]
    content_details = item["contentDetails"]
    statistics = item["statistics"]

    # Parse ISO 8601 duration
    duration = parse_duration(content_details["duration"])

    return {
        "title": snippet["title"],
        "duration": duration,
        "thumbnail_url": snippet["thumbnails"]["medium"]["url"],
        "channel_title": snippet["channelTitle"],
        "published_at": snippet["publishedAt"],
        "view_count": int(statistics.get("viewCount", 0)),
        "language": snippet.get("defaultLanguage", "N/A")
    }

def parse_duration(iso_duration):
    import isodate
    try:
        duration = isodate.parse_duration(iso_duration)
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours}h {minutes}m {seconds}s"
    except:
        return "Unknown"
