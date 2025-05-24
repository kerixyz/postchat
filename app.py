from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import requests
import json
import pandas as pd
from datetime import datetime
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

@app.route('/results/<vod_id>')
def show_results(vod_id):
    analysis_data = {
        'personas': load_json(f'personas/{vod_id}_personas.json'),
        'summaries': load_json(f'summaries/{vod_id}_summaries.json'),
        'video_info': load_json(f'metadata/{vod_id}_metadata.json')
    }
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
            "unique_users": comments_df['user_id'].nunique() if 'user_id' in comments_df.columns else "N/A"
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
            "unique_users": comments_df['user_id'].nunique() if 'user_id' in comments_df.columns else "N/A"
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
