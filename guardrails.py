import openai
import os
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def is_off_topic(message):
    prompt = f"""
Determine if the following message is unrelated to choosing a Singtel broadband or mobile plan.

Reply with only "yes" or "no".

Message: "{message}"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip().lower()
        return result.startswith("yes")
    except Exception as e:
        print(f"[Guardrails Error]: {e}")
        return False  # fallback to not block

def is_salutation(message):
    prompt = f"""
Determine if the following message is just a salutation or casual greeting, like 'hello', 'hi', or 'good morning', with no real intent to explore Singtel broadband or mobile plans.

Only reply "yes" if the message is clearly just a standalone greeting — not if it mentions telcos, plans, or account status.

Respond with only "yes" or "no".

Message: "{message}"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip().lower()
        return result.startswith("yes")
    except Exception as e:
        print(f"[Guardrails Error - Salutation]: {e}")
        return False
