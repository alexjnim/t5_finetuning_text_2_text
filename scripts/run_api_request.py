import pprint
import requests

url = "http://localhost:9000/api/v1.0/predictions"

example_input = {"text": "alex"}

response = requests.post(url, json=example_input, timeout=3600)

if response.status_code == 200:
    print("response.json():\n\n")
    pprint.pprint(response.json())
else:
    print(f"response code: {response.status_code}\n\n")
