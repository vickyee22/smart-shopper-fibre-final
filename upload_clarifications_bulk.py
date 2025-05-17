
import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS")

bulk_file = "clarifications_bulk.json"
index_name = "clarifications"

# Ensure index exists (optional)
mapping = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "metadata": {
                "properties": {
                    "intent": {"type": "keyword"},
                    "sub_status": {"type": "keyword"},
                    "sequence": {"type": "integer"},
                    "emotion": {"type": "keyword"}
                }
            }
        }
    }
}

print("[DEBUG] Creating index if it doesn't exist...")
res = requests.put(
    f"{OPENSEARCH_HOST}/{index_name}",
    auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
    headers={"Content-Type": "application/json"},
    json=mapping
)
print(f"[DEBUG] Index creation response: {res.status_code}")

# Upload bulk file
with open(bulk_file, "rb") as f:
    bulk_data = f.read()

print("[DEBUG] Sending _bulk upload request...")
res = requests.post(
    f"{OPENSEARCH_HOST}/_bulk",
    auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
    headers={"Content-Type": "application/x-ndjson"},
    data=bulk_data
)

print(f"[DEBUG] Bulk upload response: {res.status_code}")
print(res.text)
