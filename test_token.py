import requests
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")

def test_get_app_token():
    print("Testing Twitch OAuth Token Request...")

    if not CLIENT_ID or not CLIENT_SECRET:
        print("❌ Missing TWITCH_CLIENT_ID or TWITCH_CLIENT_SECRET in environment.")
        return

    url = "https://id.twitch.tv/oauth2/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials"
    }

    try:
        response = requests.post(url, data=data)
        response.raise_for_status()
        token = response.json().get("access_token")
        if token:
            print("✅ Success! Access Token:")
            print(token)
        else:
            print("⚠️ Request succeeded but no token found:")
            print(response.json())
    except requests.exceptions.RequestException as e:
        print("❌ Error during token request:")
        print(e)

if __name__ == "__main__":
    test_get_app_token()
