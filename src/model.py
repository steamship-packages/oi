"""Data model for OI"""
from typing import Optional, List

from steamship import File, Block, Tag, EmbeddingIndex, Steamship
from steamship.base.model import CamelModel

OI_RESPONSE = "oi-response"
OI_CONTEXT = "oi-context"

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

    def add_to_index(self, index: EmbeddingIndex, file_id: str):
        """Add all the prompts to the embedding index, associated with the file ID containing the results."""
        for prompt in self.prompts:
            print(f"Inserting {file_id}, {prompt.text}")
            index.insert(prompt.text, external_id=file_id)

        embed_task = index.embed()
        embed_task.wait()

        snapshot_task = index.create_snapshot()
        snapshot_task.wait()


    def to_steamship_file(self, client: Steamship) -> File.CreateRequest:
        blocks = []
        tags = []

        for response in self.responses:
            block = Block.CreateRequest(
                text=response.text,
                tags=[
                    Tag.CreateRequest(kind=OI_RESPONSE, start_idx=0, end_idx=len(response.text))
                ]
            )
            for tag in response.context or []:
                block.tags.append(Tag.CreateRequest(
                    kind=OI_CONTEXT, name=tag, start_idx=0, end_idx=len(response.text)
                ))
            blocks.append(block)

        file = File.create(
            client=client,
            blocks=blocks,
            tags=tags
        )
        return file



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
