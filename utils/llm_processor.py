import json
from openai import OpenAI
from textblob import TextBlob
from config import OPENAI_API_KEY, OPENAI_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

# {
#     "name": "The Backseat Strategist",
#     "description": "...",
#     "share": "38%",
#     "sentiment_label": "Neutral",
#     "sentiment_percent": "60%",
#     "theme": "Gameplay Critique",
#     "feedback": ["You should’ve flanked", "Why not buy armor?", "Play safe next time"],
#     "key_feedback": [
#         {
#             "label": "Wants more engagement",
#             "comments": ["Streamer ignores chat", "Missed my donation"],
#             "recommendation": "Respond to chat more often during downtime."
#         }
#     ]
# }


def process_comments(comments_df, num_personas=3, sample_size=200):
    """Single-call processor for personas, summaries, and negativity handling."""

    col = 'message' if 'message' in comments_df.columns else 'text'
    comments = comments_df[col].dropna().tolist()

    sample_comments = comments[:sample_size]

    prompt = f"""
        You are an expert in sociolinguistics and behavioral analysis of live streaming communities, specializing in Twitch chat.

        Your task is to analyze the following Twitch chat messages and extract structured *viewer personas*. Use only the following grounded persona types from Schuck (2023):

        1. **System Alterer (SA)**: Viewers who attempt to influence or steer the streamer’s behavior or content (e.g., advice, critique, backseating).
        2. **Financial Sponsor (FS)**: Viewers motivated by financial support, including subs, donations, gifted memberships, or monetary encouragement.
        3. **Social Player (SP)**: Viewers driven by social interaction, fun, and engagement (e.g., games, emotes, memes, raffles, in-group bonding).

        ---

        ### Step-by-Step Reasoning

        **Step 1: Assign Every Message a Persona**

        Categorize every message into one of the three personas using the **best fit** approach. For each message, ask:

        - Is the message trying to **change, suggest, or direct** the stream or streamer? → Assign **SA**
        - Is it referring to **supporting the stream financially**? → Assign **FS**
        - Is it about **having fun, playing along, or engaging with chat/game elements**? → Assign **SP**

        **Every message must be assigned exactly one persona** based on dominant intent — no messages should be excluded.

        ---

        **Step 2: Group Messages by Persona**

        Once labeled, cluster messages into groups by persona type.

        ---

        **Step 3: Extract Persona Insights**

        For each of the 3 persona types that appear in the data, produce the following structured output:

        - `"name"`: Use one of: "System Alterer", "Financial Sponsor", or "Social Player"
        - `"description"`: 1–2 sentence summary of this persona’s behavior in this chat context
        - `"share"`: Approximate percentage (0–100%) of total messages attributed to this persona
        - `"sentiment_label"`: Dominant sentiment — "Positive", "Neutral", or "Negative"
        - `"sentiment_percent"`: Approximate percent of that group’s messages matching the dominant sentiment (e.g., "65%")
        - `"theme"`: Dominant tone or intent (e.g., Advice, Encouragement, Playfulness, Critique)
        - `"feedback"`: 3–5 representative messages from this persona group
        - `"key_feedback"`: Insights raised by this persona group. For each, include:
            - `"label"`: A short summary of the issue or theme
            - `"comments"`: Supporting chat message examples
            - `"recommendation"`: A short suggestion for the streamer

        ---

        **Step 4: Stay Grounded**

        Base all insights and categorizations only on the provided chat messages. Do not invent or extrapolate beyond what is shown.

        ---

        Input:
        Sample Messages:
        {json.dumps(sample_comments, indent=2)}

        Respond with a JSON object:
        {{
        "personas": [
            {{
            "name": "...",
            "description": "...",
            "share": "...",
            "theme": "...",
            "feedback": ["...", "..."],
            "key_feedback": [
                {{
                "label": "...",
                "comments": ["...", "..."],
                "recommendation": "..."
                }}
            ]
            }},
            ...
        ]
        }}
        """


    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You analyze Twitch chats and extract structured insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=3000
        )
        content = response.choices[0].message.content
        # print("🔍 Raw LLM content:\n", content)  # Log the full output for inspection

        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]
            parsed = json.loads(json_str)
            # print("✅ Parsed JSON:\n", json.dumps(parsed, indent=2))  # Pretty-print parsed output
            return parsed
        except Exception as e:
            print("❌ Error parsing JSON from LLM:", e)
            print("⚠️ Attempted JSON string:\n", json_str)
            return {}

    except Exception as e:
        print("LLM call failed:", e)
        return {}
    
    
