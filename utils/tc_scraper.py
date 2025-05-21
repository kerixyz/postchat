import os
import pandas as pd
from chat_downloader import ChatDownloader

def download_twitch_chat(video_url, data_dir, max_messages=1000):
    try:
        downloader = ChatDownloader()
        chat = downloader.get_chat(video_url, max_messages=max_messages)
        
        # Extract VOD ID from URL
        vod_id = video_url.split('/videos/')[-1].split('?')[0]
        
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
        
        # Ensure output directory exists
        output_dir = os.path.join(data_dir, 'twitch_chat')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f'{vod_id}.csv')
        chat_df.to_csv(output_path, index=False)
        
        return True, f"Downloaded {len(chat_df)} messages for VOD: {vod_id}"
    
    except Exception as e:
        return False, f"Error: {str(e)}"
