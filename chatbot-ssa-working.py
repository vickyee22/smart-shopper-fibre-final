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
    with open("interaction_log_ssa.json", "a") as f:
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
        f"{OPENSEARCH_HOST}/clarifications-ssa/_search",
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
            "primary": None,
            "step": 0
        }
        open("interaction_log_ssa.json", "w").close()
        print("[DEBUG] Log reset on context reinitialization.")
    context = user_context[user_id]

    # Salutation check
    if is_salutation(message):
        reply = {
            "role": "assistant",
            "content": "Hi there! I’m here to help you find the best Singtel broadband or mobile plan. What are you looking for today?"
        }
        log_interaction(message, reply, {})
        return reply

    # Detect or confirm primary intent
    if context["primary"] is None:
        intent = detect_primary_intent_vector(message)
        intent = clarify_intent_with_llm(message, intent)
        print(f"[DEBUG] Detected intent: {intent}")
        if intent in ["fibre", "mobile"]:
            context["primary"] = intent
        else:
            if is_off_topic(message):
                print("[DEBUG] Detected off-topic input.")
                reply = {
                    "role": "assistant",
                    "content": "I'm here to assist with Singtel broadband and mobile plans. Let me know how I can help!"
                }
                log_interaction(message, reply, {})
                return reply
            reply = {
                "role": "assistant",
                "content": "Are you looking for a broadband (fibre) plan or a mobile plan?"
            }
            log_interaction(message, reply, {})
            return reply

    # Update profile after every answer
    if "profile" not in context:
        context["profile"] = {}

    print(f"[PROFILE TRACKER] Profile before update: {context['profile']}")
    updated = update_profile_fields(message, context["profile"])
    context["profile"].update(updated)
    print(f"[PROFILE TRACKER] Profile after update: {context['profile']}")

    # Fetch clarification question from vector DB
    step = context["step"]
    asked_questions = set()
    for i in range(len(history) - 1):
        if history[i]["role"] == "assistant" and history[i+1]["role"] == "user":
            asked_questions.add(history[i]["content"].strip().lower().rstrip("?"))

    while True:
        question = fetch_clarification_question(context["primary"], "new_line", step)
        if not question:
            break
        if question.strip().lower().rstrip("?") not in asked_questions:
            context["step"] += 1
            reply = {"role": "assistant", "content": question}
            log_interaction(message, reply, context.get("profile", {}))
            return reply
        step += 1

    reply = {"role": "assistant", "content": "Thanks! Based on your responses, I’ll help find the most suitable Singtel plan for you."}
    log_interaction(message, reply, context.get("profile", {}))

    # Save conversation summary for agent handoff
    try:
        # Extract all user responses in order
        user_answers = [msg["content"] for msg in history if msg["role"] == "user"]

        qna_pairs = "\n".join([
            f"{i+1}. {q}" for i, q in enumerate(user_answers)
        ])

        summary_prompt = (
            f"Summarize this customer conversation in a way that a live sales agent can take over smoothly.\n\n"
            f"User Profile: {json.dumps(context['profile'])}\n\n"
            f"Q&A:\n{qna_pairs}\n\n"
            f"Final Recommendation: {reply['content']}"
        )

        summary_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant creating handover summaries for customer support."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        conversation_summary = summary_response.choices[0].message.content.strip()

        summary_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_profile": context["profile"],
            "answers": user_answers,
            "final_recommendation": reply["content"],
            "summary": conversation_summary
        }
        with open("handoff_summary_ssa.json", "w") as f:
            json.dump(summary_entry, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to write handoff summary: {e}")

    # Recommendation logic
    try:
        with open("fibre_recommendation_matrix_ssa.json") as f:
            fibre_matrix = json.load(f)
        with open("BTL_Offers.json") as f:
            offer_details = json.load(f)

        profile = context.get("profile", {})
        matched_offer = None
        fallback_offer = None

        for offer in fibre_matrix:
            if (
                offer["intent"] == "fibre"
                and (offer["relationship_status"] == profile.get("relationship_status") or offer["relationship_status"] == "any")
                and (offer["home_size"] == profile.get("home_size") or offer["home_size"] == "any")
                and (offer["postal_code_prefix"] == profile.get("postal_code_prefix") or offer["postal_code_prefix"] == "any")
            ):
                matched_offer = offer
                break
            if offer["offerId"] == "10":  # fallback
                fallback_offer = offer

        if not matched_offer and fallback_offer:
            matched_offer = fallback_offer

        if matched_offer:
            plan_info = next((item for item in offer_details if item["offerId"] == matched_offer["offerId"]), {})
            recommendation = f"We recommend the {matched_offer['plan_name']}.\n{matched_offer['highlight']}\nLearn more: {matched_offer['link']}"
            reply = {"role": "assistant", "content": recommendation}
    except Exception as e:
        print(f"[ERROR] Failed to generate recommendation: {e}")

    return reply

# Gradio UI
gr.ChatInterface(
    fn=chat,
    title="Singtel Smart Shopper Assistant - SSA",
    type="messages"
).launch()
