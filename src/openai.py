from enum import Enum
from typing import Optional, List, Dict
import requests
from pydantic import BaseModel
from steamship import SteamshipError

from build.deps.typing_extensions import Any


class OpenAIObject(str, Enum):
    LIST = 'list'
    EMBEDDING = 'embedding'
    COMPLETION = "text_completion"


class OpenAiCompletionChoice(BaseModel):
    text: str
    index: int
    logprops: Optional[Dict]
    finish_reason: Optional[str]

    class Config:
        extra = 'allow'



class OpenAiCompletion(BaseModel):
    object: OpenAIObject # 'text_completion'
    created: int
    model: str
    choices: List[OpenAiCompletionChoice]
    usage: Optional[Dict]

    class Config:
        extra = 'allow'


def complete(api_key: str, prompt: str, stop: str = "\n", temperature: Optional[float] = 0.3) -> str:
    body = {
        "prompt": prompt,
        "model": "text-davinci-002",
        "max_tokens": 2048,
        "temperature": temperature,
        "n": 1,
        "echo": False,
        "stop": stop
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
    print(res_json)
    if res_json is None:
        raise SteamshipError(message="OpenAI response was not valid JSON.")

    completion = OpenAiCompletion.parse_obj(res_json)

    if completion.choices and completion.choices[0].text:
        return completion.choices[0].text

    if completion.choices and len(completion.choices[0].text) == 0:
        print(prompt)
        raise SteamshipError(message="OpenAI responded with an empty response.")

    raise SteamshipError(message="Response format was unexpected.")
