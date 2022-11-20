from typing import Optional
import requests
from steamship import SteamshipError


def complete(api_key: str, prompt: str, temperature: Optional[float] = 0.3) -> str:
    body = {
        "prompt": prompt,
        "model": "text-davinci-002",
        "max_tokens": 2048,
        "temperature": temperature,
        "n": 1,
        "echo": False,
        "stop": "\n"
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = "https://api.openai.com/v1/completions"
    res = requests.post(url, headers=headers, json=body)

    if not res.ok:
        raise SteamshipError(message=f"OpenAI response indicated an error. {res.text}")

    res_json = res.json()
    if res_json is None:
        raise SteamshipError(message="OpenAI response was not valid JSON")

    if "choices" in res_json and "text" in res_json["choices"]:
        return res_json["choices"]["text"]

    raise SteamshipError(message="Response format was unexpected")
