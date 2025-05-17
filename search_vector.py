
import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS")
INDEX_NAME = "smartshopper-index"

if not all([OPENAI_API_KEY, OPENSEARCH_HOST, OPENSEARCH_USER, OPENSEARCH_PASS]):
    raise ValueError("Missing one or more environment variables in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return response.data[0].embedding

def search_similar(query_text, top_k=3):
    query_vector = embed_text(query_text)

    query = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": top_k
                }
            }
        }
    }

    response = requests.get(
        f"{OPENSEARCH_HOST}/{INDEX_NAME}/_search",
        auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
        headers={"Content-Type": "application/json"},
        json=query
    )

    if response.status_code != 200:
        print("[ERROR]", response.status_code, response.text)
        return []

    hits = response.json()["hits"]["hits"]
    results = [
        {
            "score": hit["_score"],
            "text": hit["_source"]["text"],
            "intent": hit["_source"]["metadata"].get("intent"),
            "emotion": hit["_source"]["metadata"].get("emotion"),
            "response_prompt": hit["_source"]["metadata"].get("response_prompt")
        }
        for hit in hits
    ]
    return results

# Example usage
if __name__ == "__main__":
    query = input("Enter a user query: ")
    results = search_similar(query)

    if not results:
        print("No relevant match found.")
    else:
        print("Top matches:")
        for i, res in enumerate(results, 1):
            print(f"#{i}: [{res['score']:.4f}] {res['text']}")
            print(f"Intent: {res['intent']}, Emotion: {res['emotion']}")
            print(f"Prompt: {res['response_prompt']}")
            print("-" * 50)
