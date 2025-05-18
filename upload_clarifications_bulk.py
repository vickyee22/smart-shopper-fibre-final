import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS")

INDEX_MAP = {
    "clarifications_poc.json": "clarifications-poc",
    "clarifications_ssa.json": "clarifications-ssa",
    "fibre_recommendation_matrix_ssa.json": "fibre-recommendation-ssa",
    "mobile_recommendation_matrix_ssa.json": "mobile-recommendation-ssa",
    "BTL_Offers.json": "fibre-offers-ssa"
}

for bulk_file, index_name in INDEX_MAP.items():
    print(f"\n[DEBUG] Processing index: {index_name}")

    # Delete existing index
    print("[DEBUG] Deleting existing index...")
    res = requests.delete(
        f"{OPENSEARCH_HOST}/{index_name}",
        auth=(OPENSEARCH_USER, OPENSEARCH_PASS)
    )
    print(f"[DEBUG] Delete response: {res.status_code}")

    # Create new index with mapping
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
