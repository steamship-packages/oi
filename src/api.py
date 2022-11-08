"""Description of your app."""
from typing import Optional, Type

from steamship.invocable import Config, create_handler, post, PackageService
from src.model import OiFeed, OiIntent, OiAnswer, OiPrompt, OiQuestion, OiResponse

class OiPackage(PackageService):
    """Example steamship Package."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def config_cls(self) -> Type[Config]:
        return Config

    @post("learn_intent")
    def learn_intent(self, intent: OiIntent = None) -> str:
        """Learn an intent."""
        return "I have learned: {intent}"

    @post("learn_intent")
    def learn_feed(self, feed: OiFeed = None) -> str:
        """Learn a whole feed of intents."""
        results = []
        for intent in feed.intents:
            results.append(self.learn_intent(intent))

    @post("query")
    def query(self, question: Optional[OiQuestion] = None) -> OiAnswer:
        """Query Oi with a question."""
        return OiAnswer(
            top_response=OiResponse(
                text="Here's your answer!"
            )
        )


handler = create_handler(OiPackage)
