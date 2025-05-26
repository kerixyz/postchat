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
        {"time": "0:00:01", "user": "GankPlz", "message": "ayo we live 🟢", "category": None},
        {"time": "0:00:02", "user": "biscuitbay", "message": "hi chat!!!", "category": None},
        {"time": "0:00:03", "user": "wardpls", "message": "LET’S GOOOO", "category": None},
        {"time": "0:00:05", "user": "comfyYordle", "message": "snacks secured", "category": None},
        {"time": "0:00:10", "user": "emojiCloud", "message": "🌟 streamer is online 🌟", "category": None},
        {"time": "0:00:15", "user": "autofilledagain", "message": "stream looking crispy today", "category": None},
        {"time": "0:08:38", "user": "glacialtempo", "message": "streamer already tilted 😭", "category": None},
        {"time": "0:08:46", "user": "wardpls", "message": "tower dive incoming", "category": None},
        {"time": "0:08:49", "user": "pokeflexer", "message": "CAT", "category": None},
        {"time": "0:08:52", "user": "GlitchQueen", "message": "This team is full of brain-dead monkeys", "category": "Profanity"},
        {"time": "0:00:00", "user": "generalspazzzz", "message": "Dude you’re so trash, uninstall", "category": "Profanity"},
        {"time": "0:08:56", "user": "VayneDiff", "message": "wait we’re losing to THAT team?", "category": None},
        {"time": "0:09:12", "user": "generalspazzzz", "message": "chat’s getting spicy rn", "category": None},
        {"time": "0:11:42", "user": "glacialtempo", "message": "nice flash lmao", "category": "Trolling"},
        {"time": "0:11:59", "user": "jerkstorerxD", "message": "bet he’s playing with his feet", "category": "Trolling"},
        {"time": "0:15:17", "user": "ThreshMain99", "message": "no shot bro is still trying", "category": "Trolling"},
        {"time": "0:15:20", "user": "grabbsmurf", "message": "lmao carried again", "category": "Trolling"},
        {"time": "0:15:24", "user": "Iron3Lifer", "message": "say “Jungle diff” if you’re real 😂", "category": "Trolling"},
        {"time": "0:15:27", "user": "BluewardKing", "message": "Yo you should queue ranked again, this time go AFK", "category": "Trolling"},
        {"time": "0:15:41", "user": "BusDriverLux", "message": "0/7 but he’s “learning” 😭", "category": "Trolling"},
        {"time": "0:15:55", "user": "LeagueMom", "message": "chat, y’all okay today??", "category": None},
        {"time": "0:20:30", "user": "snackattacktft", "message": "when you gonna collab with faker? 😂", "category": "Other Streamers"},
        {"time": "0:21:05", "user": "trashdiveTV", "message": "lol this guy plays like Nightblue but without the clout", "category": "Other Streamers"},
        {"time": "0:21:11", "user": "KDA4lyfe", "message": "wait didn’t that dude flame you last week?", "category": "Trolling"},
        {"time": "0:21:14", "user": "potionlord", "message": "bro thinks he’s tyler1", "category": "Other Streamers"},
        {"time": "0:21:17", "user": "KICKjax", "message": "say less, you’re way better than LS", "category": "Other Streamers"},
        {"time": "0:21:20", "user": "soloqonly", "message": "T1 viewers be wildin fr", "category": "Other Streamers"},
        {"time": "0:21:23", "user": "ZoningIsHard", "message": "he’s just a worse version of dantes", "category": "Other Streamers"},
        {"time": "0:21:27", "user": "9deaths1win", "message": "didn’t you ragequit when Tarzaned came in your stream?", "category": "Other Streamers"},
        {"time": "0:24:40", "user": "comfyYordle", "message": "i came for chill league not influencer beef 💀", "category": "Other Streamers"},
        {"time": "0:30:12", "user": "catnaplux", "message": "bye bye ", "category": None}
    ]

    themes = {
        "Profanity": {
            "title": "Cursing, Profanity, and Exclusionary Language",
            "color": "#3b82f6",
            "description": "This category highlights messages that focus on swearing, insults, and harmful or dehumanizing language.",
            "findings": [
                'Users used insults and dehumanizing phrases (e.g., "brain-dead monkeys", "you\'re so trash").',
                "Messages were aggressive and repeated within a short time, creating a hostile tone."
            ],
            "recommendations": [
                "Use moderation tools to automatically filter or flag profanity and personal attacks.",
                "Set clear chat rules and apply timeouts or bans for repeated toxic behavior."
            ]
        },
        "Trolling": {
            "title": "Trolling",
            "description": "This category highlights messages that include baiting, fake tips, and sarcastic mockery meant to derail or agitate.",
            "color": "#facc15",
            "findings": [
                'Several users posted baiting or sarcastic messages intended to provoke (e.g., “go AFK”, “0/7 but he’s ‘learning’”).',
                "Troll messages disrupted the tone of the chat and encouraged pile-on behavior."
            ],
            "recommendations": [
                "Enable keyword-based filters for phrases often used in sarcastic trolling or passive aggression.",
                "Address repeated trolling with escalating timeouts to reduce derailing behavior."
            ]
        },
        "Other Streamers": {
            "title": "Other Content Creators",
            "color": "#8b5cf6",
            "description": "This category highlights comments about or comparisons to other streamers — often negative or drama-inducing.",
            "findings": [
                'Chatters made passive-aggressive comparisons targeting other creators (e.g., “you’re way better than LS”).',
                "Comments encouraged revisited past drama, fueling ongoing tension."
            ],
            "recommendations": [
                "Monitor for indirect callouts or baiting comparisons, especially around known figures or communities.",
                "Use soft moderation (e.g., warning prompts or cooldowns) to prevent escalation without over-policing."
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

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    metadata_dir = os.path.join(app.config['DATA_DIR'], 'metadata')
    vod_options = []

    for filename in os.listdir(metadata_dir):
        if filename.endswith('_metadata.json'):
            vod_id = filename.replace('_metadata.json', '')
            metadata = load_json(f'metadata/{filename}')
            title = metadata.get("title", "Untitled")
            date_str = metadata.get("stream_date", "")[:10]
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
        print(f"✅ Uploaded {remote_path} to Supabase.")
    else:
        print(f"❌ Upload failed: {res.status_code} - {res.text}")

def load_json(relative_path):
    # Local check
    path = os.path.join(app.config['DATA_DIR'], relative_path)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)

    # Supabase fallback
    base_url = os.getenv("SUPABASE_URL")  
    if not base_url:
        print("⚠️ SUPABASE_URL not set")
        return {}

    remote_url = f"{base_url}/{relative_path}"
    try:
        res = requests.get(remote_url)
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
