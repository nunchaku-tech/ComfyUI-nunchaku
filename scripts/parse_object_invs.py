import requests
import sphobjinv

url = "https://nunchaku.tech/docs/nunchaku/objects.inv"
response = requests.get(url)
response.raise_for_status()

with open("objects.inv", "wb") as f:
    f.write(response.content)

inv = sphobjinv.Inventory("objects.inv")

base_url = "https://nunchaku.tech/docs/nunchaku/"

for item in inv.objects:
    print(f"{item.name} -> {base_url}{item.uri} ({item.role})")
