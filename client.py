import requests

data = {"prompt" : "O Romeo, Romeo! wherefore "}

url = "http://localhost:8080/predict"

response = requests.post(url, json=data)

print(response.json())