import gradio as gr
from intent_classifier import detect_primary_intent
from guardrails import is_off_topic, is_salutation

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

def chat(message, history):
    user_id = "default_user"
    if user_id not in user_context:
        user_context[user_id] = {
            "primary": None,
            "sub_status": None,
            "step": 0,
            "telco_clarified": False
        }

    context = user_context[user_id]

    # Step 0: Handle greetings using GPT
    if is_salutation(message):
        return {
            "role": "assistant",
            "content": "Hi there! I’m here to help you find the best Singtel broadband or mobile plan. What are you looking for today?"
        }

    # Step 1: Detect primary intent
    if not context["primary"]:
        primary = detect_primary_intent(message)
        print(f"[DEBUG] Detected primary intent: {primary}")

        if primary != "unknown":
            context["primary"] = primary

            # GPT-based empathetic response
            try:
                import openai
                import json
                import os
                from dotenv import load_dotenv

                load_dotenv()
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                with open("prompts.json", "r") as f:
                    system_prompt = json.load(f)["system_prompt"]

                empathy_prompt = (
                    f"The user said: \"{message}\"\n\n"
                    "Respond as a helpful and empathetic Singtel sales assistant. Acknowledge their concern if any, "
                    "and smoothly ask: 'Are you currently with Singtel or switching from another provider?'"
                )

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": empathy_prompt}
                    ]
                )
                reply = response.choices[0].message.content.strip()
                print(f"[DEBUG] Empathy GPT response: {reply}")
                return {"role": "assistant", "content": reply}

            except Exception as e:
                return {
                    "role": "assistant",
                    "content": "Thanks for sharing. Are you currently with Singtel or switching from another provider?"
                }

        # Step 1.5: Guardrail check
        if is_off_topic(message):
            return {
                "role": "assistant",
                "content": "Apologies, I'm here specifically to help you explore Singtel broadband and mobile plans. Let me know how I can assist with that!"
            }

        return {
            "role": "assistant",
            "content": "Hi there! I'm your Singtel assistant. Are you looking for a broadband (fibre) or mobile plan today?"
        }

    # Step 2: Clarify telco status
    print(f"[DEBUG] Telco clarification not done yet. User message: {message}")
    if not context["telco_clarified"]:
        if "singtel" in message.lower():
            context["sub_status"] = "recontract"
        else:
            context["sub_status"] = "new_line"
        context["telco_clarified"] = True
        question = CLARIFICATION_QUESTIONS[context["primary"]][context["sub_status"]][0]
        return {
            "role": "assistant",
            "content": f"Got it — I understand where you're coming from. Let's explore some better options for you.\n\n{question}"
        }

    # Step 3: Ask clarification questions
    step = context["step"]
    primary = context["primary"]
    sub_status = context["sub_status"]
    questions = CLARIFICATION_QUESTIONS[primary][sub_status]
    context["step"] += 1

    if context["step"] < len(questions):
        return {"role": "assistant", "content": questions[context["step"]]}

    # Step 4: All questions answered → recommend plan
    user_answers = [msg["content"] for msg in history if msg["role"] == "user"][-len(questions):]
    qna_pairs = "\n\n".join([
        f"Q{i+1}: {questions[i]}\nA{i+1}: {user_answers[i]}" for i in range(len(questions))
    ])
    prompt = (
        f"A customer answered the following about their {sub_status.replace('_',' ')} {primary} plan needs:\n\n"
        f"{qna_pairs}\n\n"
        f"Recommend the most suitable Singtel plan with a short reason."
    )

    try:
        import openai
        import os
        import json
        from dotenv import load_dotenv

        load_dotenv()
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    user_context[user_id] = {
        "primary": None,
        "sub_status": None,
        "step": 0,
        "telco_clarified": False
    }

    return {"role": "assistant", "content": reply}

gr.ChatInterface(
    fn=chat,
    title="Singtel Smart Shopper Assistant",
    type="messages"
).launch()
