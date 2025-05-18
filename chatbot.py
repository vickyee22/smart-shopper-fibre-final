import gradio as gr
from guardrails import is_off_topic, is_salutation
import os
import requests
import datetime
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS")
INDEX_NAME = "smartshopper-index"

client = OpenAI(api_key=OPENAI_API_KEY)

def log_interaction(user_input, assistant_reply, profile):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_input": user_input,
        "assistant_reply": assistant_reply["content"],
        "profile": profile.copy()
    }
    with open("interaction_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def clarify_intent_with_llm(message, initial_intent):
    prompt = f"""
You are an AI assistant helping customers choose between Singtel mobile and fibre plans.

The user said: "{message}"
The system thinks the intent might be "{initial_intent}".

Please confirm which type of plan the user is referring to based on their input. If it is unclear, respond with "unknown".
Respond with only one word: "fibre", "mobile", or "unknown".
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Only respond with: fibre, mobile, or unknown."},
                {"role": "user", "content": prompt.strip()}
            ]
        )
        raw = response.choices[0].message.content.strip().lower()
        print(f"[DEBUG] Raw LLM response for clarification: {raw}")
        if raw in ["fibre", "mobile"]:
            return raw
        return "unknown"
    except Exception as e:
        print(f"[ERROR] clarify_intent_with_llm failed: {e}")
        return "unknown"

def update_profile_fields(message, existing_profile):
    import json
    system_prompt = (
        "Extract these fields from the user's message:\n"
        "- plan_type: fibre or mobile\n"
        "- current_provider: singtel or other (e.g., Starhub, M1, Circles are other)\n"
        "- relationship_status: new_line or recontract\n\n"
        "Return a JSON object with only the fields detected in this message. "
        "Ignore anything unrelated. Do not guess."
    )

    prompt = f"User said: \"{message}\"\n\nExisting profile: {json.dumps(existing_profile)}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    try:
        extracted = json.loads(response.choices[0].message.content)
        print(f"[DEBUG] Extracted profile fields: {extracted}")
        return extracted
    except Exception as e:
        print(f"[DEBUG] Failed to parse profile: {e}")
        reply = {
            "role": "assistant",
            "content": "Sorry, something went wrong while processing your request."
        }
        log_interaction(message, reply, context["profile"])
        return reply

def fetch_clarification_question(intent, sub_status, step):
    query = {
        "size": 1,
        "query": {
            "bool": {
                "must": [
                    {"term": {"metadata.intent": intent}},
                    {"term": {"metadata.sub_status": sub_status}},
                    {"term": {"metadata.sequence": step + 1}}
                ]
            }
        }
    }

    res = requests.get(
        f"{OPENSEARCH_HOST}/clarifications/_search",
        auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
        headers={"Content-Type": "application/json"},
        json=query
    )

    if res.status_code != 200:
        print(f"[DEBUG] Clarification fetch failed: {res.status_code}")
        return None

    hits = res.json().get("hits", {}).get("hits", [])
    if not hits:
        print("[DEBUG] No clarification question found.")
        return None

    return hits[0]["_source"]["text"]


def detect_emotion(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Classify the user's emotional tone as one of: neutral, frustration, or positive. "
                           "Treat complaints about price, speed, or dissatisfaction as frustration."
            },
            {
                "role": "user",
                "content": f'What is the emotional tone of: "{text}"?'
            }
        ]
    )
    emotion = response.choices[0].message.content.strip().lower()
    print(f"[DEBUG] Emotion detected: {emotion}")
    return emotion


CLARIFICATION_QUESTIONS = {
    "fibre": {
        "recontract": [
            "Are you experiencing issues with your current fibre plan?",
            "Do you need faster speed or better stability?",
            "Would you like to bundle with other services for more value?"
        ],
        "new_line": [
            "What's the size of your home or number of rooms?",
            "How many people will be using the internet at home?",
            "Are you a heavy user for gaming or streaming?"
        ]
    },
    "mobile": {
        "recontract": [
            "Are you looking to recontract with or without a new phone?",
            "Has your data or usage needs changed?",
            "What’s your priority: price, more data, or phone upgrade?"
        ],
        "new_line": [
            "How much data do you need per month?",
            "Do you prefer a contract or no contract?",
            "What’s your monthly budget?"
        ]
    }
}

user_context = {}

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return response.data[0].embedding

def detect_primary_intent_vector(message, threshold=0.5):
    vector = embed_text(message)
    query = {
        "size": 1,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": 1
                }
            }
        }
    }
    print(f"[DEBUG] Sending vector search for intent: {message}")
    res = requests.get(
        f"{OPENSEARCH_HOST}/{INDEX_NAME}/_search",
        auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
        headers={"Content-Type": "application/json"},
        json=query
    )
    print(f"[DEBUG] OpenSearch response status: {res.status_code}")
    if res.status_code != 200:
        print(f"[ERROR] OpenSearch response: {res.status_code}")
        return "unknown"
    hits = res.json().get("hits", {}).get("hits", [])
    if not hits:
        print("[DEBUG] No matches found.")
        return "unknown"

    print(f"[DEBUG] Vector match: {hits[0]['_source']['text']} | Score: {hits[0]['_score']}")
    if hits[0]["_score"] < threshold:
        print("[DEBUG] Match score too low.")
        return "unknown"
    return hits[0]["_source"]["metadata"]["intent"]

def chat(message, history):
    print(f"[DEBUG] Received message: {message}")
    user_id = "default_user"
    if user_id not in user_context:
        print("[DEBUG] Initializing new user context")
        user_context[user_id] = {
        "profile": {
            "plan_type": None,
            "current_provider": None,
            "relationship_status": None
        },
        "primary": None,
        "sub_status": None,
        "step": 0,
        "telco_clarified": False
    }
        open("interaction_log.jsonl", "w").close()
        print("[DEBUG] Log reset on context reinitialization.")

    context = user_context[user_id]

    # ✅ Salutation check first
    if is_salutation(message):
        reply = {
            "role": "assistant",
            "content": "Hi there! I’m here to help you find the best Singtel broadband or mobile plan. What are you looking for today?"
        }
        log_interaction(message, reply, context['profile'])
        return reply
        
    print(f"[PROFILE TRACKER] Profile before update: {context['profile']}")
    # print(f"[Context TRACKER] Primary after update: {primary}")
    # print(f"[Context TRACKER] Substatus after update: {sub_status}")

    # Guardrail check
    primary = context["primary"]
    missing = [k for k, v in context["profile"].items() if v is None]
    if missing:
        print(f"[PROFILE TRACKER] Profile before update: {context['profile']}")
        updates = update_profile_fields(message, context["profile"])
        print(f"[DEBUG] Extracted profile fields: {updates}")
        context["profile"].update(updates)
        print(f"[PROFILE TRACKER] Profile after update: {context['profile']}")

        if context["profile"]["plan_type"] and not context["primary"]:
            context["primary"] = context["profile"]["plan_type"]
        if context["profile"]["relationship_status"]:
            context["sub_status"] = context["profile"]["relationship_status"]
        if context["profile"]["current_provider"]:
            context["telco_clarified"] = True
        print(f"[PROFILE TRACKER] Profile after update: {context['profile']}")

        # Re-calculate missing after update to prevent re-asking already filled fields
        missing = [k for k, v in context["profile"].items() if v is None]

        if "plan_type" in missing and not context["primary"]:
            reply = {
                "role": "assistant",
                "content": "Are you looking for a broadband (fibre) plan or a mobile plan?"
            }
            log_interaction(message, reply, context["profile"])
            return reply

        if "current_provider" in missing and not context["telco_clarified"]:
            reply = {
                "role": "assistant",
                "content": "Are you currently with Singtel or switching from another provider?"
            }
            log_interaction(message, reply, context["profile"])
            return reply

        if "relationship_status" in missing and not context["sub_status"]:
            reply = {
                "role": "assistant",
                "content": "Are you signing up for a new line or recontracting an existing plan?"
            }
            log_interaction(message, reply, context["profile"])
            return reply

        print(f"[DEBUG] Updated profile: {context['profile']}")

        if not missing:
            print("[DEBUG] Profile complete. Skipping intent classification.")
        else:
            print("[DEBUG] No primary intent yet. Checking off-topic...")
            primary = detect_primary_intent_vector(message)
            print(f"[DEBUG] Initial vector intent: {primary}")
            print(f"[DEBUG] Ready to clarify intent using GPT...")
            primary = clarify_intent_with_llm(message, primary)
            print(f"[DEBUG] Final intent after clarify_intent_with_llm: {primary}")
            print(f"[DEBUG] Final intent after LLM clarification: {primary}")

            if primary == "unknown" and not context["primary"]:
                reply = {
                    "role": "assistant",
                    "content": "Got it. Are you referring to a broadband (fibre) plan or a mobile plan?"
                }
                log_interaction(message, reply, context['profile'])
                return reply

            if primary == "unknown":
                if is_off_topic(message):
                    print("[DEBUG] Detected off-topic, exiting.")
                    reply = {
                        "role": "assistant",
                        "content": "Apologies, I'm here specifically to help you explore Singtel broadband and mobile plans. Let me know how I can assist with that!"
                    }
                    log_interaction(message, reply, context['profile'])
                    return reply

            context["primary"] = primary

        emotion = detect_emotion(message).strip().lower()
        print(f"[DEBUG] Detected emotion: {emotion}")

        if "frustration" in emotion:
            tone = f"Sorry to hear that! Let's explore better {primary} options for you."
        elif "positive" in emotion:
            tone = f"Awesome! Let's help you find the right {primary} plan."
        else:
            tone = f"Thanks for sharing. You're looking for {primary} plans."

        reply = {
            "role": "assistant",
            "content": f"{tone} Are you currently with Singtel or switching from another provider?"
        }
        log_interaction(message, reply, context['profile'])
        # NOTE: Do not return here; proceed to clarification questions block below

    # Clarify if user is recontracting or new
    print(f"[DEBUG] Telco clarification not done yet. User message: {message}")
    if not context["sub_status"]:
        reply = {
            "role": "assistant",
            "content": "Are you signing up for a new line or recontracting an existing plan?"
        }
        log_interaction(message, reply, context["profile"])
        return reply

    # Proceed with clarification questions
    step = context["step"]
    primary = context["primary"]
    sub_status = context["sub_status"]
    # Gather asked questions to avoid repeating questions already answered
    asked_questions = set()
    for i in range(len(history) - 1):
        if history[i]["role"] == "assistant" and history[i+1]["role"] == "user":
            question = history[i]["content"].strip().lower().rstrip("?")
            asked_questions.add(question)

    while True:
        question = fetch_clarification_question(primary, sub_status, step)
        if not question:
            break
        if question.strip().lower().rstrip("?") not in asked_questions:
            context["step"] = step + 1
            reply = {"role": "assistant", "content": question}
            log_interaction(message, reply, context["profile"])
            return reply
        step += 1
    else:
        print('[DEBUG] No more clarification questions.')

    # All questions answered → final recommendation
    num_questions = context["step"]
    user_answers = [msg["content"] for msg in history if msg["role"] == "user"][-num_questions:]
    qna_pairs = "\n\n".join([
        f"A{i+1}: {user_answers[i]}" for i in range(len(user_answers))
    ])
    prompt = (
        f"A customer answered the following about their {sub_status.replace('_',' ')} {primary} plan needs:\n\n"
        f"{qna_pairs}\n\n"
        f"Recommend the most suitable Singtel plan with a short reason."
    )

    try:
        with open("prompts.json", "r") as f:
            system_prompt = json.load(f)["system_prompt"]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        reply = response.choices[0].message.content.strip()
        print(f"[DEBUG] Final GPT reply: {reply}")

    except Exception as e:
        reply = f"Error: {str(e)}"

    print("[DEBUG] Resetting user context")
    user_context[user_id] = {"primary": None, "sub_status": None, "step": 0, "telco_clarified": False}
    reply = {"role": "assistant", "content": reply}
    log_interaction(message, reply, context['profile'])
    return reply

# Gradio UI
gr.ChatInterface(
    fn=chat,
    title="Singtel Smart Shopper Assistant",
    type="messages"
).launch()
