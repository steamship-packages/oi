"""Data model for OI"""
from typing import Optional, List

from steamship import File, Block, Tag, EmbeddingIndex, Steamship
from steamship.base.model import CamelModel

OI_RESPONSE = "oi-response"
OI_CONTEXT = "oi-context"
OI_INTENT = "oi-intent"

class OiResponse(CamelModel):
    # The text of the response
    text: str

    # Tags describing the context in which this response is appropriate
    context: Optional[List[str]] = None

    @staticmethod
    def from_steamship_block(block) -> "Optional[OiResponse]":
        context: List[str] = []
        is_oi_response: bool = False

        for tag in block.tags or []:
            if tag.kind == OI_RESPONSE:
                is_oi_response = True
            elif tag.kind == OI_CONTEXT:
                context.append(tag.name)

        if not is_oi_response:
            return None
        return OiResponse(text=block.text, context=context)

    def to_steamship_block(self) -> Block.CreateRequest:
        block = Block.CreateRequest(
            text=self.text,
            tags=[
                Tag.CreateRequest(kind=OI_RESPONSE, start_idx=0, end_idx=len(self.text))
            ]
        )
        for tag in self.context or []:
            block.tags.append(Tag.CreateRequest(
                kind=OI_CONTEXT, name=tag, start_idx=0, end_idx=len(self.text)
            ))
        return block

    def score(self, other_context: Optional[List[str]] = None):
        """

        Higher is better.

        """
        if other_context is None:
            return -1 * len(self.context or [])

        matches = 0
        for item in self.context or []:
            if item in other_context:
                matches += 1
            else:
                # For now, if we're conditioned on context not provided, we consider it a hard "NO" match
                return 0
        return matches



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


    @staticmethod
    def from_steamship_file(file: File) -> "OiIntent":
        responses = [OiResponse.from_steamship_block(block) for block in file.blocks or []]
        responses = [r for r in responses if r is not None]

        handle = None
        for tag in file.tags or []:
            if tag.kind == OI_INTENT:
                handle = tag.name

        # Note: we're not storing the prompts here..
        # I think that's OK for now, but in a future version where the embedding indices are more tightly
        # woven into the Block & Tag structure, we probably should.
        return OiIntent(
            handle=handle,
            responses=responses
        )

    def top_response(self, other_context: Optional[List[str]] = None) -> Optional[OiResponse]:
        top_score = None
        top_response = None

        for response in self.responses or []:
            score = response.score(other_context)
            if top_score is None or score > top_score:
                top_score = score
                top_response = response

        return top_response




    def to_steamship_file(self, client: Steamship) -> File:
        file = File.create(
            client=client,
            blocks=[response.to_steamship_block() for response in self.responses],
            tags=[Tag.CreateRequest(
                kind=OI_INTENT,
                name=self.handle
            )]
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
    top_response: Optional[OiResponse]
