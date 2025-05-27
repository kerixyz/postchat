from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import requests
import json
import pandas as pd
import random
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from utils.tc_scraper import download_twitch_chat
from utils.llm_processor import process_comments
from utils.llm_processor_v2 import batch_process_comments
from utils.llm_processor_v2 import compare_personas
from utils.twitch_api import get_vod_metadata
from utils.yt_scraper import download_youtube_chat
from utils.youtube_api import get_video_metadata

app = Flask(__name__)
app.config['DATA_DIR'] = 'data'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

def bucketize_chat(messages, bucket_minutes=5):
    def parse_time(t):
        h, m, s = map(int, t.split(":"))
        return timedelta(hours=h, minutes=m, seconds=s)

    buckets = defaultdict(list)

    for i, msg in enumerate(messages):
        t = parse_time(msg["time"]).total_seconds()
        bucket_seconds = int((t // (bucket_minutes * 60)) * (bucket_minutes * 60))
        bucket_key = str(timedelta(seconds=bucket_seconds))
        buckets[bucket_key].append((i, msg))

    results = []
    for key, entries in sorted(buckets.items()):
        indices, msgs = zip(*entries)
        themes = [m["category"] for m in msgs if m["category"]]
        most_common_theme = Counter(themes).most_common(1)
        theme = most_common_theme[0][0] if most_common_theme else None
        results.append({
            "time": key,
            "count": len(msgs),
            "first_index": indices[0],
            "theme": theme
        })

    return results

@app.route('/demo')
def demo():
    chat_messages = [
        {"time": "0:00:01", "user": "modBot", "message": "Stream is live! Welcome everyone 🎮", "category": None},
        {"time": "0:00:05", "user": "LaneChamp23", "message": "let's goooo, game 1 baby!", "category": None},
        {"time": "0:00:07", "user": "minionStealer", "message": "hope we see some spicy jungle plays", "category": None},
        {"time": "0:00:15", "user": "ClutchFlash", "message": "early invade??", "category": None},
        {"time": "0:01:21", "user": "GlacialTempo", "message": "clean leash, looking good", "category": None},
        {"time": "0:02:33", "user": "SnaxTime", "message": "ok real talk why does he always int lane", "category": None},
        {"time": "0:03:48", "user": "GamerMom", "message": "we still believe 🙏", "category": None},
        {"time": "0:05:09", "user": "wardpls", "message": "LOL he missed Q", "category": "Trolling"},
        {"time": "0:07:22", "user": "blueBuffThief", "message": "bro's playing with his monitor off 💀", "category": "Trolling"},
        {"time": "0:08:55", "user": "JustJungle", "message": "dude stop blaming your team", "category": None},
        {"time": "0:09:42", "user": "matchRecap", "message": "close fight midlane, unlucky flash", "category": None},
        {"time": "0:10:57", "user": "bushCamper", "message": "that gank was actually cracked", "category": None},
        {"time": "0:12:28", "user": "JungleGap88", "message": "he’s pinging like it's helping 💀", "category": None},
        {"time": "0:14:40", "user": "Sneakywarder", "message": "chat kinda wild rn 😭", "category": None},
        {"time": "0:16:02", "user": "RoamTiming", "message": "support diff confirmed", "category": None},
        {"time": "0:18:10", "user": "GameOver", "message": "L, next game", "category": None},

        # Game 2 Start
        {"time": "0:20:01", "user": "modBot", "message": "Game 2 loading... predictions in chat now!", "category": None},
        {"time": "0:21:15", "user": "CSIsOptional", "message": "can he finally win a lane? let's find out", "category": None},
        {"time": "0:22:41", "user": "MinimapLooker", "message": "look at that TP timing 😳", "category": None},
        {"time": "0:24:22", "user": "MutedYou", "message": "he’s been better this game so far tbh", "category": None},
        {"time": "0:25:19", "user": "toxicTilted", "message": "you are legit dogshit, uninstall", "category": "Profanity"},
        {"time": "0:27:36", "user": "Lanternbait", "message": "ok but why does he play like this?", "category": None},
        {"time": "0:28:44", "user": "TiltKing", "message": "this guy is such a washed up clown", "category": "Profanity"},
        {"time": "0:30:10", "user": "macroMind", "message": "good rotation mid", "category": None},
        {"time": "0:32:12", "user": "SupportMain7", "message": "nice roam!!", "category": None},
        {"time": "0:34:05", "user": "RedSideAce", "message": "actual comeback arc right now", "category": None},
        {"time": "0:36:48", "user": "positiveVibes", "message": "this team comp lowkey winnable tho", "category": None},
        {"time": "0:38:19", "user": "catnaplux", "message": "team diff tbh", "category": None},

        # Game 2 Late
        {"time": "0:41:07", "user": "GGNoLane", "message": "this team actually turned it around", "category": None},
        {"time": "0:43:29", "user": "MemeThrower", "message": "bro thinks he's Faker with that flash", "category": None},
        {"time": "0:45:02", "user": "glacialtempo", "message": "say less, you’re way better than LS", "category": "Other Streamers"},
        {"time": "0:47:11", "user": "KDA4Lyfe", "message": "not even Tarzaned would throw like that", "category": "Other Streamers"},
        {"time": "0:48:44", "user": "Pentakillerz", "message": "he actually hit a skillshot 😲", "category": None},
        {"time": "0:50:00", "user": "StreamChamps", "message": "gg wp", "category": None},
        {"time": "0:52:10", "user": "macroGap", "message": "gotta say, clean macro that game", "category": None},
        {"time": "0:54:22", "user": "LoLChef", "message": "chat what’s for dinner tonight?", "category": None},
        {"time": "0:56:01", "user": "CanyonEnjoyer", "message": "great jungle pathing, no troll", "category": None},
        {"time": "0:58:44", "user": "modBot", "message": "Thanks for watching! Stream ending in 60s.", "category": None},
        {"time": "0:59:55", "user": "FinalWordz", "message": "W stream, cya next time!", "category": None}
    ]
    themes = {
        "Profanity": {
            "title": "Cursing, Profanity, and Exclusionary Language",
            "color": "#3b82f6",
            "description": "This category highlights messages that focus on swearing, insults, and harmful or dehumanizing language.",
            "findings": [
                "Two profanity-laced messages appeared midstream, with one using explicit insults and another using demeaning language.",
                "Toxic comments were isolated but stood out due to their severity amid generally neutral conversation."
            ],
            "recommendations": [
                "Use profanity filters that escalate based on severity, allowing for lighter interventions on isolated outbursts.",
                "Pin or rotate reminders about respectful chat conduct at key points in the stream (e.g., post-death or between games)."
            ]
        },
        "Trolling": {
            "title": "Trolling",
            "description": "This category highlights messages that include baiting, fake tips, and sarcastic mockery meant to derail or agitate.",
            "color": "#facc15",
            "findings": [
                "Two sarcastic or mocking messages appeared during the first 10 minutes, mainly targeting gameplay mistakes.",
                "Trolling did not escalate or trigger follow-up negativity, but showed up during high-pressure moments."
            ],
            "recommendations": [
                "Watch for early trolling during initial gameplay dips, and consider gentle mod intervention to set the tone.",
                "Log repeat offenders silently and apply escalating timeouts only if trolling persists across games."
            ]
        },
        "Other Streamers": {
            "title": "Comparison with Other Content Creators",
            "color": "#8b5cf6",
            "description": "This category highlights comments about or comparisons to other streamers — often negative or drama-inducing.",
            "findings": [
                "Two late-stream comparisons referenced well-known creators in dismissive or competitive ways.",
                "Mentions were casual but risked inviting drama-based discussion near the end of the stream."
            ],
            "recommendations": [
                "Introduce moderation prompts that activate after known streamer names are mentioned in succession.",
                "Soft-moderate with emote-only or cooldown periods if side-by-side comparisons begin to dominate chat."
            ]
        }
    }

    chart_data = bucketize_chat(chat_messages, bucket_minutes=5)

    return render_template("demo.html", chat_messages=chat_messages, themes=themes, chart_data=chart_data)


@app.route('/results/<vod_id>')
def show_results(vod_id):
    # Load analysis data
    analysis_data = {
        'personas': load_json(f'personas/{vod_id}_personas.json'),
        'summaries': load_json(f'summaries/{vod_id}_summaries.json'),
        'video_info': load_json(f'metadata/{vod_id}_metadata.json')
    }

    # List of available icons
    icon_filenames = [
        'icon01.png', 'icon02.png', 'icon03.png', 'icon04.png',
        'icon05.png', 'icon06.png', 'icon07.png', 'icon08.png'
    ]

    selected_icons = random.sample(icon_filenames, 3)

    for persona, icon in zip(analysis_data['personas'], selected_icons):
        persona['icon'] = icon

    return render_template('results.html', vod_id=vod_id, **analysis_data)

@app.route('/download_twitch', methods=['POST'])
def download_twitch():
    try:
        twitch_url = request.form.get('twitch_url').strip()
        vod_id = twitch_url.split('/videos/')[-1].split('?')[0]

        success, message = download_twitch_chat(twitch_url, app.config['DATA_DIR'])
        if not success:
            return render_template('error.html', message=message)

        comments_path = os.path.join(app.config['DATA_DIR'], 'twitch_chat', f'{vod_id}.csv')
        comments_df = pd.read_csv(comments_path)
        # result = process_comments(comments_df)
        result = batch_process_comments(comments_df, batch_size=200, num_personas=3)

        personas = result.get("personas", [])
        summaries = {
            "overall_summary": " ".join(result.get("summaries", {}).values()),
            "total_messages": len(comments_df),
            "unique_users": comments_df['author_id'].nunique() if 'author_id' in comments_df.columns else "N/A"
        }

        # Fetch Twitch VOD metadata
        vod_metadata = get_vod_metadata(vod_id)
        if not vod_metadata:
            raise Exception("Unable to fetch VOD metadata")

        # Fix the thumbnail
        raw_url = vod_metadata["thumbnail_url"]
        thumbnail = raw_url.replace("%{width}", "640").replace("%{height}", "360")

        # Format the stream date
        created_at = vod_metadata.get("created_at")
        stream_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").strftime("%B %d, %Y") if created_at else "Unknown"

        # Build metadata
        video_info = {
            "title": vod_metadata["title"],
            "duration": vod_metadata["duration"],
            "thumbnail": thumbnail,
            "broadcaster": vod_metadata["user_name"],
            "stream_date": stream_date,
            "views": vod_metadata["view_count"],
            "language": vod_metadata.get("language", "N/A")
        }


        save_analysis(vod_id, {
            'personas': personas,  # Save the full LLM persona objects
            'summaries': summaries,
            'metadata': video_info
        })
        return redirect(url_for('show_results', vod_id=vod_id))
    
    except Exception as e:
        return render_template('error.html', message=str(e))

@app.route('/youtube')
def youtube_form():
    return render_template('youtube.html')  # Create a simple HTML form like your Twitch form

@app.route('/download_youtube', methods=['POST'])
def download_youtube():
    try:
        youtube_url = request.form.get('youtube_url').strip()
        video_id = youtube_url.split('v=')[-1].split('&')[0]

        success, message = download_youtube_chat(youtube_url, app.config['DATA_DIR'])
        if not success:
            return render_template('error.html', message=message)

        comments_path = os.path.join(app.config['DATA_DIR'], 'youtube_chat', f'{video_id}.csv')
        comments_df = pd.read_csv(comments_path)
        result = batch_process_comments(comments_df, batch_size=200, num_personas=3)

        personas = result.get("personas", [])
        summaries = {
            "overall_summary": " ".join(result.get("summaries", {}).values()),
            "total_messages": len(comments_df),
            "unique_users": comments_df['author_id'].nunique() if 'author_id' in comments_df.columns else "N/A"
        }

        video_metadata = get_video_metadata(video_id)
        if not video_metadata:
            raise Exception("Unable to fetch YouTube video metadata")

        video_info = {
            "title": video_metadata["title"],
            "duration": video_metadata["duration"],
            "thumbnail": video_metadata["thumbnail_url"],
            "broadcaster": video_metadata["channel_title"],
            "stream_date": video_metadata["published_at"],
            "views": video_metadata["view_count"],
            "language": video_metadata.get("language", "N/A")
        }

        save_analysis(video_id, {
            'personas': personas,
            'summaries': summaries,
            'metadata': video_info
        })
        return redirect(url_for('show_results', vod_id=video_id))

    except Exception as e:
        return render_template('error.html', message=str(e))

import os
import json
import requests
from flask import request, render_template

def list_supabase_metadata_files():
    base_url = os.getenv("SUPABASE_URL")
    bucket = os.getenv("SUPABASE_BUCKET")
    supabase_key = os.getenv("SUPABASE_API_KEY")

    if not base_url or not bucket:
        print("⚠️ SUPABASE_URL or SUPABASE_BUCKET not set")
        return []

    api_url = f"{base_url}/storage/v1/object/list/{bucket}"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, headers=headers, json={"prefix": "metadata/"})
        response.raise_for_status()
        items = response.json()
        return [item['name'] for item in items if item['name'].endswith('_metadata.json')]
    except Exception as e:
        print(f"⚠️ Failed to list files from Supabase: {e}")
        return []



@app.route('/compare', methods=['GET', 'POST'])
def compare():
    metadata_dir = os.path.join(app.config['DATA_DIR'], 'metadata')
    local_files = []

    # Get local files
    if os.path.exists(metadata_dir):
        local_files = [
            f for f in os.listdir(metadata_dir)
            if f.endswith('_metadata.json')
        ]

    # Get Supabase files
    supabase_files = list_supabase_metadata_files()

    # Merge and deduplicate
    all_files = sorted(set(local_files + supabase_files))
    vod_options = []

    for filename in all_files:
        vod_id = filename.replace('_metadata.json', '')
        metadata = load_json(f'metadata/{filename}')
        if metadata:  # Ensure non-empty metadata
            vod_options.append({
                "id": vod_id,
                "title": metadata.get("title", "Untitled"),
                "broadcaster": metadata.get("broadcaster", "Unknown"),
                "stream_date": metadata.get("stream_date", "Unknown"),
                "label": f"{metadata.get('title', 'Untitled')} — {metadata.get('broadcaster', '')} ({metadata.get('stream_date', '')})"
            })

    if request.method == 'POST':
        vod_a = request.form.get('vod_a')
        vod_b = request.form.get('vod_b')

        try:
            personas_a = load_json(f'personas/{vod_a}_personas.json')
            personas_b = load_json(f'personas/{vod_b}_personas.json')
            meta_a = load_json(f'metadata/{vod_a}_metadata.json')
            meta_b = load_json(f'metadata/{vod_b}_metadata.json')

            return render_template(
                'compare.html',
                vod_options=vod_options,
                vod_a=vod_a,
                vod_b=vod_b,
                meta_a=meta_a,
                meta_b=meta_b,
                personas_a=personas_a,
                personas_b=personas_b
            )

        except FileNotFoundError as e:
            return render_template('error.html', message=f"Missing analysis file: {e}")

    return render_template('compare.html', vod_options=vod_options)

def upload_to_supabase(local_path, remote_path):
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_API_KEY")
    bucket = os.getenv("SUPABASE_BUCKET", "postchat")

    full_url = f"{url}/storage/v1/object/{bucket}/{remote_path}"
    
    with open(local_path, 'rb') as file_data:
        res = requests.post(
            full_url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/octet-stream"
            },
            data=file_data
        )

    if res.status_code in [200, 201]:
        print(f"Uploaded {remote_path} to Supabase.")
    else:
        print(f"Upload failed: {res.status_code} - {res.text}")

def load_json(relative_path):
    # Try local first
    local_path = os.path.join(app.config['DATA_DIR'], relative_path)
    if os.path.exists(local_path):
        with open(local_path, 'r') as f:
            return json.load(f)

    # Supabase fallback
    base_url = os.getenv("SUPABASE_URL")
    bucket = os.getenv("SUPABASE_BUCKET")
    key = os.getenv("SUPABASE_API_KEY")

    if not base_url or not bucket:
        print("⚠️ SUPABASE_URL or SUPABASE_BUCKET not set")
        return {}

    remote_url = f"{base_url}/storage/v1/object/public/{bucket}/{relative_path}"
    try:
        res = requests.get(remote_url, headers={
            "apikey": key,
            "Authorization": f"Bearer {key}"
        })
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"⚠️ Failed to fetch from Supabase: {remote_url}\n{e}")
        return {}


def save_analysis(vod_id, data):
    for key in ['personas', 'summaries', 'metadata']:
        path = os.path.join(app.config['DATA_DIR'], key, f'{vod_id}_{key}.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data[key], f)

        upload_to_supabase(path, f"{key}/{vod_id}_{key}.json")


if __name__ == '__main__':
    for subdir in ['twitch_chat', 'youtube_chat', 'personas', 'summaries']:
        os.makedirs(os.path.join(app.config['DATA_DIR'], subdir), exist_ok=True)
    app.run(debug=True)
