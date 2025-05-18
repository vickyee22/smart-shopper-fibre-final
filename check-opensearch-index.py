import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("OPENSEARCH_HOST")
if not host.startswith("http"):
    host = f"https://{host}"
auth = HTTPBasicAuth(os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASS"))

res = requests.get(
    f"{host}/fibre-recommendation-ssa/_search",
    auth=auth,
    headers={"Content-Type": "application/json"},
    json={"size": 10}
)

print(res.json())