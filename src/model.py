"""Data model for OI"""
import logging
from enum import Enum
from random import choice
from typing import Optional, List

from steamship import File, Block, Tag, EmbeddingIndex, Steamship
from steamship.base.model import CamelModel

OI_RESPONSE = "oi-response"
OI_CONTEXT = "oi-context"
OI_INTENT = "oi-intent"

class OiResponseType(str, Enum):
    FIXED = "fixed"
    SHUFFLE = "shuffle"

class OiResponse(CamelModel):
    # The type of response
    type: Optional[OiResponseType] = None

    # The text of the response, for response_type = FIXED
    text: Optional[str]

    # The options of the response, for response_type = SHUFFLE
    text_options: Optional[List[str]]

    # Tags describing the context in which this response is appropriate
    context: Optional[List[str]] = None

    block_id: Optional[str] = None

    @staticmethod
    def from_steamship_block(block) -> "Optional[OiResponse]":
        context: List[str] = []
        is_oi_response: bool = False

        ret = OiResponse(contet=context, block_id=block.id)

        for tag in block.tags or []:
            if tag.kind == OI_RESPONSE:
                is_oi_response = True
                ret.text = block.text
                if tag.value and tag.value["text"]:
                    ret.text = tag.value["text"]
                if tag.value and tag.value["text_options"]:
                    ret.text = tag.value["text_options"]

            elif tag.kind == OI_CONTEXT:
                context.append(tag.name)

        if not is_oi_response:
            return None
        ret.context = context
        return ret

    def to_steamship_block(self) -> Block.CreateRequest:
        text = "(no text)"
        if self.text:
            text = self.text
        elif self.text_options:
            text = "\n".join(self.text_options)

        block = Block.CreateRequest(
            text=text,
            tags=[
                Tag.CreateRequest(kind=OI_RESPONSE, start_idx=0, end_idx=len(text), value={
                    "text": self.text,
                    "text_options": self.text_options
                })
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

    def complete_response(self):
        """Generate the complete response.

        A response can be fixed (e.g. `response.text`).
        But it can also be something generated, or one of a set of responses,
        """
        if self.type is None or self.type == OiResponseType.FIXED:
            return OiResponse(text=self.text, context=self.context, block_id=self.block_id)
        elif self.type == OiResponseType.SHUFFLE:
            if self.text_options:
                text = choice(self.text_options)
                return OiResponse(text=text, context=self.context, block_id=self.block_id)
            else:
                return OiResponse(text=self.text, context=self.context, block_id=self.block_id)


class OiPrompt(CamelModel):
    # The text of the response
    text: str

    embedding_id: Optional[str] = None


class OiIntent(CamelModel):
    # An intent should probably have a name
    handle: str

    # An intent has a list of prompts that match it
    prompts: List[OiPrompt] = None

    # An intent has a list of responses for different contexts
    responses: List[OiResponse] = None

    # The file ID to associate it with
    file_id: str = None

    def add_to_index(self, index: EmbeddingIndex, file_id: str) -> List[OiPrompt]:
        """Add all the prompts to the embedding index, associated with the file ID containing the results."""
        ret = []
        new_additions = []
        for prompt in self.prompts:
            if prompt.embedding_id is not None:
                logging.info(f"Skipping index embed of prompt: {prompt.embedding_id} / {prompt.text}")
                ret.append(prompt)
            else:
                logging.info(f"Adding index embed of prompt: {prompt.text}")
                res = index.insert(prompt.text, external_id=file_id)
                item = res.item_ids[0]
                prompt.embedding_id = item.id
                ret.append(prompt)
                new_additions.append(prompt)

        if len(new_additions):
            logging.info(f"Added {len(new_additions)} new additions so embedding.")
            embed_task = index.embed()
            embed_task.wait()

            logging.info(f"Added {len(new_additions)} new additions so snapshotting.")
            snapshot_task = index.create_snapshot()
            snapshot_task.wait()
        else:
            logging.info(f"Did not add any new additions; neither embedding nor snapshotting.")

        return ret

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
            responses=responses,
            file_id=file.id
        )

    def top_response(self, other_context: Optional[List[str]] = None) -> Optional[OiResponse]:
        print("context", other_context)
        print("responses", self.responses)
        top_score = None
        top_response = None

        for response in self.responses or []:
            score = response.score(other_context)
            if top_score is None or score > top_score:
                top_score = score
                top_response = response

        return top_response


    def to_steamship_file(self, client: Steamship) -> File:
        if self.file_id is None:
            # Create the file
            logging.info(f"Creating new intent file for {self.handle}: {self.file_id}")
            file = File.create(
                client=client,
                blocks=[response.to_steamship_block() for response in self.responses],
                tags=[Tag.CreateRequest(
                    kind=OI_INTENT,
                    name=self.handle
                )]
            )
            return file
        else:
            logging.info(f"Reloading intent file for {self.handle}: {self.file_id}")
            return File.get(client=client, _id=self.file_id)


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
