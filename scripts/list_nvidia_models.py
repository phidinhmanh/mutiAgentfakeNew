import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")
url = "https://integrate.api.nvidia.com/v1/models"
headers = {"Authorization": f"Bearer {api_key}"}

response = requests.get(url, headers=headers)
if response.status_code == 200:
    models = response.json().get("data", [])
    for m in models:
        if "qwen" in m["id"].lower() or "llama" in m["id"].lower():
            print(m["id"])
else:
    print(f"Error: {response.status_code} - {response.text}")
