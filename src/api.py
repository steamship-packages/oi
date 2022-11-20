"""Description of your app."""
import logging
from typing import Optional, Type, List

from steamship import EmbeddingIndex, File, SteamshipError
from steamship.invocable import Config, create_handler, post, PackageService

from model import OiFeed, OiIntent, OiAnswer, OiTrigger, OiQuestion, OiResponse

class OiPackageConfig(Config):
    openai_api_key: Optional[str] = None

class OiPackage(PackageService):
    """Example steamship Package."""

    config: OiPackageConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedder = self.client.use_plugin("openai-embedder", config={
            "model": "text-similarity-davinci-001",
            "dimensionality": 12288
        })
        self.index = EmbeddingIndex.create(
            client=self.client,
            handle="prompt-index",
            plugin_instance=self.embedder.handle,
            fetch_if_exists=True
        )

    def config_cls(self) -> Type[Config]:
        return OiPackageConfig

    @post("learn_intent")
    def learn_intent(self, intent: OiIntent = None) -> OiIntent:
        """Learn an intent."""
        if not intent:
            raise SteamshipError(message="Provided `intent` was None")
        if isinstance(intent, dict):
            intent = OiIntent.parse_obj(intent)
        return intent.save(self.client, self.index)

    @post("learn_feed")
    def learn_feed(self, feed: OiFeed = None) -> OiFeed:
        """Learn a whole feed of intents."""
        if isinstance(feed, dict):
            feed = OiFeed.parse_obj(feed)
        if not feed:
            raise SteamshipError(message="Provided `feed` was None")
        return feed.save(self.client, self.index)

    @post("query")
    def query(self, question: Optional[OiQuestion] = None) -> OiAnswer:
        """Query Oi with a question."""
        if isinstance(question, dict):
            question = OiQuestion.parse_obj(question)
        search_task = self.index.search(question.text, include_metadata=True)
        search_task.wait()

        if not search_task.output.items:
            return OiAnswer(top_response=None)
        else:
            # Get return the item
            top = search_task.output.items[0].value
            file_id = top.external_id

            # Get the file and turn it into an intent
            file = File.get(self.client, _id=file_id)
            matched_intent = OiIntent.from_steamship_file(file)

            # Find the best matching response from the context adn return it
            response = matched_intent.top_response(question.context)

            # Now we have to generate the return response.
            # 1. Fixed response
            # 2. Shuffled from a list of options
            # Along with possible prompt-based completion
            ret_response = response.complete_response(
                client=self.client,
                question=question,
                intent=matched_intent,
                openai_api_key=self.config.openai_api_key
            )

            return OiAnswer(top_response=ret_response)


handler = create_handler(OiPackage)
