"""Description of your app."""
from typing import Optional, Type, List

from steamship import EmbeddingIndex, File, SteamshipError
from steamship.invocable import Config, create_handler, post, PackageService

from src.model import OiFeed, OiIntent, OiAnswer, OiPrompt, OiQuestion, OiResponse

class OiPackage(PackageService):
    """Example steamship Package."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedder = self.client.use_plugin("nlpcloud-tagger", config={
            "task": "embeddings",
            "model": "paraphrase-multilingual-mpnet-base-v2",
            "dimensionality": 768
        })
        self.index = EmbeddingIndex.create(
            client=self.client,
            plugin_instance=self.embedder.handle,
        )

    def config_cls(self) -> Type[Config]:
        return Config

    @post("learn_intent")
    def learn_intent(self, intent: OiIntent = None) -> bool:
        """Learn an intent."""
        if not intent:
            raise SteamshipError(message="Provided `intent` was None")

        # Create a file that contains the responses
        response_file = intent.to_steamship_file(self.client)

        # Now add the prompts to the index, linking each item with the file
        intent.add_to_index(self.index, response_file.id)

        return True

    @post("learn_intent")
    def learn_feed(self, feed: OiFeed = None) -> List[str]:
        """Learn a whole feed of intents."""
        print(f"Learning {feed} ")
        if not feed:
            raise SteamshipError(message="Provided `feed` was None")

        results = [self.learn_intent(intent) for intent in feed.intents or []]
        return results

    @post("query")
    def query(self, question: Optional[OiQuestion] = None) -> OiAnswer:
        """Query Oi with a question."""

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
            return OiAnswer(top_response=response)


handler = create_handler(OiPackage)
