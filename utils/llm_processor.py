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
        You are an expert in sociolinguistic and behavioral analysis of live streaming communities, with a focus on Twitch chat.

        Your task is to analyze the following Twitch chat messages and extract structured *viewer personas*. Use a step-by-step process.

        Step 1: Identify {num_personas} distinct viewer personas based on recurring communication patterns, tone, goals, and social cues.

        Step 2: For each persona, generate the following structured data:
        - "name": A concise, descriptive label (e.g., "The Hype Booster", "The Backseat Strategist")
        - "description": 1–2 sentences describing their role or behavior in chat
        - "share": An estimated percentage (0–100%) representing how common this persona is in the chat sample
        - "sentiment_label": One of "Positive", "Neutral", or "Negative"
        - "sentiment_percent": An estimated proportion of this persona’s messages that carry the dominant sentiment (e.g., "75%")
        - "theme": The dominant intent or tone (e.g., Encouragement, Critique, Humor, Frustration)
        - "feedback": 3–5 representative chat messages from this persona group
        - "key_feedback": A list of notable concerns or insights
        - Each entry includes:
            - "label": a short summary of the issue
            - "comments": example messages reflecting that issue
            - "recommendation": a concise suggestion for the streamer to address this

        Step 3: Use only information grounded in the provided messages. Do not hallucinate or fabricate ungrounded interpretations.

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
    
    
