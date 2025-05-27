import json
from openai import OpenAI
from textblob import TextBlob
import pandas as pd
from collections import defaultdict
from difflib import SequenceMatcher
import re
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

        Your task is to analyze the following Twitch chat messages and extract structured *viewer personas*. Use only the grounded viewer types from Schuck (2023):

        1. **System Alterer (SA)**: Viewers who attempt to influence or steer the streamer’s behavior or content (e.g., advice, critique, backseating).
        2. **Financial Sponsor (FS)**: Viewers motivated by financial support, including subs, donations, gifted memberships, or monetary encouragement.
        3. **Social Player (SP)**: Viewers driven by social interaction, fun, and engagement (e.g., games, emotes, memes, raffles, in-group bonding).

        ---

        ### Step-by-Step Reasoning

        **Step 1: Assign Every Message a Persona**

        Categorize every message into one of the three viewer personas using a **best fit** approach. For each message, ask:

        - Is the message trying to **change, suggest, or direct** the streamer? → Assign **SA**
        - Does it refer to **supporting the stream financially**? → Assign **FS**
        - Is it about **having fun, reacting, or socializing with others**? → Assign **SP**

        **Every message must be assigned exactly one persona.**

        **Also determine each message’s dominant communication focus**:

        - **Streamer-focused**: Directed toward or about the streamer.
        - **Chat-focused**: Directed toward or engaging with other viewers or chat culture.
        - **Self-focused**: Primarily centered on the speaker themselves (e.g., donations, accomplishments, reactions).

        ---

        **Step 2: Group Messages by Persona**

        Once labeled, cluster messages into groups based on their assigned persona type. Consider the communication **focus** to help differentiate intent and reinforce distinctions.

        ---

        **Step 3: Extract Persona Insights**

        For each of the 3 persona types that appear in the data, generate the following structured summary:
                
        - `"name"`: One of "System Alterer", "Financial Sponsor", or "Social Player"
        - `"description"`: 1–2 sentence summary of this persona’s behavior and how it's distinct
        - `"share"`: Approximate percent (0–100) of total messages in this group
        - `"sentiment_label"`: Dominant sentiment — "Positive", "Neutral", or "Negative"
        - `"sentiment_percent"`: Percent of messages with that sentiment
        - `"theme"`: Dominant tone or function — e.g., "Hype", "Advice", "Praise", "Critique", "Playfulness"
        - `"focus"`: Dominant communication focus — one of: "Streamer-focused", "Chat-focused", or "Self-focused"
        - `"feedback"`: 3–5 representative messages from this persona group
        - `"key_feedback"`: A list of insights raised. For each, include:
            - `"label"`: Short summary of the issue or theme
            - `"comments"`: Sample supporting messages
            - `"recommendation"`: Suggestion for the streamer

        ---

        ### Step 4: Stay Grounded

        Base all classifications and insights **only** on the provided chat messages. Do not speculate or infer beyond the data.


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
            "sentiment_label": "...",
            "sentiment_percent": "...",
            "theme": "...",
            "focus": "...",
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
        print("🔍 Raw LLM content:\n", content)  # Log the full output for inspection

        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]
            parsed = json.loads(json_str)
            print("✅ Parsed JSON:\n", json.dumps(parsed, indent=2))  # Pretty-print parsed output
            return parsed
        except Exception as e:
            print("❌ Error parsing JSON from LLM:", e)
            print("⚠️ Attempted JSON string:\n", json_str)
            return {}

    except Exception as e:
        print("LLM call failed:", e)
        return {}
    

def batch_process_comments(comments_df, batch_size=200, num_personas=3):
    col = 'message' if 'message' in comments_df.columns else 'text'
    comments = comments_df[col].dropna().tolist()
    all_results = []

    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        batch_df = pd.DataFrame({col: batch})  # Maintain original column name
        result = process_comments(batch_df, num_personas=num_personas)
        if result and "personas" in result:
            all_results.extend(result["personas"])

    return aggregate_personas(all_results, max_personas=num_personas)


def normalize_name(name):
    return re.sub(r'\W+', '', name).lower()

def similar(a, b, threshold=0.75):
    return SequenceMatcher(None, a, b).ratio() >= threshold

def aggregate_personas(personas_list, max_personas=3):
    merged = []
    used = set()

    for i, p1 in enumerate(personas_list):
        if i in used:
            continue
        group = [p1]
        name1 = normalize_name(p1['name'])

        for j, p2 in enumerate(personas_list[i + 1:], start=i + 1):
            if j in used:
                continue
            name2 = normalize_name(p2['name'])
            if similar(name1, name2):
                group.append(p2)
                used.add(j)

        merged_persona = merge_persona_group(group)
        merged.append(merged_persona)
        
    merged.sort(key=lambda p: clean_percent(p['share']), reverse=True)
    return {"personas": merged[:max_personas]}


def clean_percent(value):
    if isinstance(value, str):
        return int(value.replace('%', '').strip())
    elif isinstance(value, (int, float)):
        return int(value)
    return 0  # default fallback

def merge_persona_group(group):
    name = group[0]['name']
    description = group[0]['description']
    theme = group[0]['theme']

    total_share = sum(clean_percent(p['share']) for p in group)
    avg_sentiment = sum(clean_percent(p['sentiment_percent']) for p in group) // len(group)
    dominant_sentiment = group[0]['sentiment_label']  # improve this later

    all_feedback = []
    key_feedback = []

    for p in group:
        all_feedback.extend(p.get('feedback', []))
        key_feedback.extend(p.get('key_feedback', []))

    return {
        "name": name,
        "description": description,
        "share": total_share,
        "sentiment_label": dominant_sentiment,
        "sentiment_percent": f"{avg_sentiment}%",
        "theme": theme,
        "feedback": list(set(all_feedback))[:5],
        "key_feedback": merge_key_feedback(key_feedback)
    }

def merge_key_feedback(feedback_items):
    merged = {}
    for item in feedback_items:
        label = item['label']
        if label not in merged:
            merged[label] = {
                "label": label,
                "comments": set(item['comments']),
                "recommendation": item['recommendation']
            }
        else:
            merged[label]['comments'].update(item['comments'])

    return [
        {
            "label": v['label'],
            "comments": list(v['comments'])[:3],
            "recommendation": v['recommendation']
        }
        for v in merged.values()
    ]

def compare_personas(personas_a, personas_b, similarity_threshold=0.75):
    from difflib import SequenceMatcher

    def normalize(name):
        return re.sub(r'\W+', '', name).lower()

    def similar(n1, n2):
        return SequenceMatcher(None, normalize(n1), normalize(n2)).ratio() >= similarity_threshold

    matched = []
    unique_a = []
    unique_b = personas_b.copy()

    for p1 in personas_a:
        match = None
        for p2 in unique_b:
            if similar(p1['name'], p2['name']):
                match = p2
                break
        if match:
            matched.append({
                "name": p1['name'],
                "a": p1,
                "b": match
            })
            unique_b.remove(match)
        else:
            unique_a.append(p1)

    return {
        "matched": matched,
        "unique_a": unique_a,
        "unique_b": unique_b
    }
