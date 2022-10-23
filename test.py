import requests
import json

if __name__ == '__main__':
    api_url = "http://localhost:8000/predict"
    response = requests.get(api_url)
    print(json.loads(response.text)['Predictions'])