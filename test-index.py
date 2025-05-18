import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("OPENSEARCH_HOST")
auth = HTTPBasicAuth(os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASS"))

res = requests.get(
    f"{host}/fibre-recommendation-ssa/_search",
    auth=auth,
    headers={"Content-Type": "application/json"},
    params={"q": "offerId:b4"}
)

print(res.status_code)
print(res.text)