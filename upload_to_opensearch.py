import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env in project root
load_dotenv(dotenv_path=".env")

# Read credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS")
INDEX_NAME = "smartshopper-index"

# Debug: Check that all variables are loaded
print(f"[DEBUG] Auth: {(OPENSEARCH_USER, OPENSEARCH_PASS)}")
print(f"[DEBUG] OpenSearch Host: {OPENSEARCH_HOST}")

# Verify config
if not OPENAI_API_KEY or not OPENSEARCH_HOST or not OPENSEARCH_USER or not OPENSEARCH_PASS:
    raise ValueError("Missing configuration values from .env file.")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Embed the given text
def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return response.data[0].embedding

# Load input samples
jsonl_path = "ssa_examples.jsonl"  # Ensure this file is in the same folder
if not os.path.exists(jsonl_path):
    raise FileNotFoundError(f"Sample input file not found: {jsonl_path}")

with open(jsonl_path, "r") as f:
    for line in f:
        doc = json.loads(line)
        vector = embed_text(doc["text"])
        body = {
            "text": doc["text"],
            "embedding": vector,
            "metadata": doc["metadata"]
        }
        res = requests.post(
            f"{OPENSEARCH_HOST}/{INDEX_NAME}/_doc",
            auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
            headers={"Content-Type": "application/json"},
            json=body
        )
        print(f"Indexed: {doc['text']} - Status: {res.status_code}")
