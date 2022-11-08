"""Data model for OI"""
from typing import Optional, List

from steamship.base.model import CamelModel

class OiResponse(CamelModel):
    # The text of the response
    text: str

    # Tags describing the context in which this response is appropriate
    context: Optional[List[str]] = None

class OiPrompt(CamelModel):
    # The text of the response
    text: str

class OiIntent(CamelModel):
    # An intent should probably have a name
    handle: str

    # An intent has a list of prompts that match it
    prompts: List[OiPrompt] = None

    # An intent has a list of responses for different contexts
    responses: List[OiResponse] = None

class OiFeed(CamelModel):
    # Name of the feed
    handle: str

    # List of intents in the feed
    intents: List[OiIntent]

class OiQuestion(CamelModel):
    text: str
    context: Optional[List[str]]

class OiAnswer(CamelModel):
    top_response: OiResponse
