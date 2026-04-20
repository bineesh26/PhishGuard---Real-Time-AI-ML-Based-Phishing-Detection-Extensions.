import requests

url1 = "google.com"
url2 = "https://google.com"

res1 = requests.post("http://127.0.0.1:8000/check", json={"url": url1}).json()
res2 = requests.post("http://127.0.0.1:8000/check", json={"url": url2}).json()

print(f"URL: {url1} -> Pred: {res1['prediction']}, Prob: {res1['probability']}")
print(f"URL: {url2} -> Pred: {res2['prediction']}, Prob: {res2['probability']}")
