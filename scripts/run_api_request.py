import pprint
import requests

url = "http://localhost:9000/api/v1.0/predictions"

example_input = {"text": "Boris Johnson may have misled Parliament over Partygate on four occasions, MPs investigating his conduct say. Evidence strongly suggests breaches of coronavirus rules would have been \"obvious\" to Mr Johnson, the privileges committee said in an update. The former PM was among those fined by police for breaking lockdown rules at gatherings in Downing Street."}

response = requests.post(url, json=example_input, timeout=3600)

if response.status_code == 200:
    print("response.json():\n\n")
    pprint.pprint(response.json())
else:
    print(f"response code: {response.status_code}\n\n")
