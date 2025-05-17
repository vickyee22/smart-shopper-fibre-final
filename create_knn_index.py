import os
import requests
from dotenv import load_dotenv

# Explicitly load .env from current directory
load_dotenv(dotenv_path=".env")

host = os.getenv("OPENSEARCH_HOST")
auth = (os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASS"))
index_name = "smartshopper-index"

# Debug to verify values loaded correctly
print("[DEBUG] HOST:", host)
print("[DEBUG] AUTH:", auth)

if not host or not auth[0] or not auth[1]:
    raise ValueError("Missing OpenSearch configuration. Please check your .env file.")

index_url = f"{host}/{index_name}"
headers = {"Content-Type": "application/json"}

payload = {
    "settings": {
        "index": {
            "knn": True
        }
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1536,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "faiss"
                }
            },
            "metadata": {
                "properties": {
                    "intent": {"type": "keyword"},
                    "emotion": {"type": "keyword"},
                    "response_prompt": {"type": "text"}
                }
            }
        }
    }
}

# Create the index
res = requests.put(index_url, auth=auth, headers=headers, json=payload)
print("[RESPONSE]", res.status_code)
print(res.text)
