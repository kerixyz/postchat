import os
import pandas as pd
from chat_downloader import ChatDownloader

def download_youtube_chat(video_url, data_dir, max_messages=1000):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
        }
        downloader = ChatDownloader(headers=headers)
        chat = downloader.get_chat(
            url=video_url,
            max_messages=max_messages
        )

        # Extract YouTube video ID
        if "v=" in video_url:
            video_id = video_url.split("v=")[-1].split("&")[0]
        else:
            video_id = video_url.split("/")[-1]

        messages = []
        for message in chat:
            messages.append({
                'message_id': message.get('message_id'),
                'text': message.get('message'),
                'timestamp': message.get('timestamp'),
                'time_in_seconds': message.get('time_in_seconds'),
                'author_name': message.get('author', {}).get('name'),
                'author_id': message.get('author', {}).get('id')
            })

        chat_df = pd.DataFrame(messages)

        # --- Bot Filtering ---
        known_youtube_bots = [
            'nightbot', 'streamlabs', 'soundalerts', 'streamelements'
        ]

        chat_df = chat_df[
            (chat_df['author_id'].notna()) &
            (chat_df['author_id'] != '') &
            (~chat_df['author_name'].str.lower().isin(known_youtube_bots))
        ]

        output_dir = os.path.join(data_dir, 'youtube_chat')
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f'{video_id}.csv')
        chat_df.to_csv(output_path, index=False)

        return True, f"Downloaded {len(chat_df)} messages for video: {video_id}"

    except Exception as e:
        return False, f"Error: {str(e)}"
