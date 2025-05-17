import openai
import os
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def detect_primary_intent(message):
    prompt = f"""
Classify the user's request into one of these high-level categories:

- fibre
- mobile
- unknown

Only return one word. No explanation. No punctuation.

User input: "{message}"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        intent = response.choices[0].message.content.strip().lower().split()[0]
        print(f"[DEBUG] Primary intent: {intent}")
        return intent if intent in ["fibre", "mobile"] else "unknown"
    except Exception as e:
        print(f"[ERROR] Primary intent detection failed: {e}")
        return "unknown"

def detect_sub_intent(message):
    prompt = f"""
Classify the user's intent into one of the following:

- new_line
- recontract
- unknown

Only return one word. No explanation. No punctuation.

User input: "{message}"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        intent = response.choices[0].message.content.strip().lower().split()[0]
        print(f"[DEBUG] Sub-intent: {intent}")
        return intent if intent in ["new_line", "recontract"] else "unknown"
    except Exception as e:
        print(f"[ERROR] Sub-intent detection failed: {e}")
        return "unknown"
