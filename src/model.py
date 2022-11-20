"""Data model for OI"""
import logging
from enum import Enum
from random import choice
from typing import Optional, List

from steamship import File, Block, Tag, EmbeddingIndex, Steamship, SteamshipError
from steamship.base.model import CamelModel
from steamship.utils.kv_store import KeyValueStore

from openai import complete

OI_RESPONSE = "oi-response"
OI_CONTEXT = "oi-context"
OI_INTENT = "oi-intent"

class OiResponseType(str, Enum):
    FIXED = "fixed"
    SHUFFLE = "shuffle"


class GptPrompt(CamelModel):
    """A Prompt for completion against GPT-3 or other system.

    Type the prompt with the following variables:

    {query_text} - The incoming user query
    {prompt_text} - The prompt this user query matched to
    {response_text} - The initial response that intent produced

    """
    handle: str

    # Text of the response
    text: str

    temperature: Optional[float] = None

    def save(self, client: Steamship):
        store = GptPrompt.get_store(client)
        store.set(self.handle, {
            "text": self.text,
            "temperature": self.temperature
        })

    @staticmethod
    def get_store(self, client: Steamship) -> KeyValueStore:
        store = KeyValueStore(client, store_identifier="PromptStore")
        return store

    @staticmethod
    def get_from_handle(self, client: Steamship, handle: str) -> "Optional[GptPrompt]":
        store = GptPrompt.get_store(client)
        obj = store.get(handle)
        if obj is None:
            return None

        return OiTrigger.parse_obj({
            "text": obj.get("text"),
            "temperature": obj.get("temperature")
        })

    def complete_response(
            self,
            question: "OiQuestion",
            intent: "OiIntent",
            response_text: str,
            api_key: str
    ):
        """Generate the complete response using the template."""
        compiled_prompt = self.text.format(**{
            "query_text": question.text,
            "prompt_text": prompt.text,
            "response_text": response_text
        })
        return complete(api_key=api_key, prompt=compiled_prompt, temperature=self.temperature)


class OiResponse(CamelModel):
    # The type of response
    type: Optional[OiResponseType] = None

    # The text of the response, for response_type = FIXED
    text: Optional[str]

    # The options of the response, for response_type = SHUFFLE
    text_options: Optional[List[str]]

    # The GPT-3 prompt to run to generate the final response
    prompt_handle: Optional[str]

    # Tags describing the context in which this response is appropriate
    context: Optional[List[str]] = None

    block_id: Optional[str] = None

    class Config:
        use_enum_values = False

    @staticmethod
    def from_steamship_block(block) -> "Optional[OiResponse]":
        context: List[str] = []
        is_oi_response: bool = False

        ret = OiResponse(contet=context, block_id=block.id)

        for tag in block.tags or []:
            if tag.kind == OI_RESPONSE:
                is_oi_response = True
                ret.text = block.text
                ret.block_id = tag.block_id
                if tag.value:
                    if tag.value["text"]:
                        ret.text = tag.value["text"]
                    if tag.value["text_options"]:
                        ret.text_options = tag.value["text_options"]
                    if tag.value["type"]:
                        ret.type = OiResponseType(tag.value["type"])

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
                    "text_options": self.text_options,
                    "type": self.type.value if self.type is not None else None
                })
            ]
        )
        for tag in self.context or []:
            block.tags.append(Tag.CreateRequest(
                kind=OI_CONTEXT, name=tag, start_idx=0, end_idx=len(text)
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

    def complete_response(
            self,
            client: Steamship,
            question: "OiQuestion",
            intent: "OiIntent",
            openai_api_key: str
    ):
        """Generate the complete response.

        A response can be fixed (e.g. `response.text`).
        But it can also be something generated, or one of a set of responses,
        """

        # First, either select the fixed output text or a shuffled one.
        if self.type is None or self.type == OiResponseType.FIXED:
            output_text = self.text
        elif self.type == OiResponseType.SHUFFLE:
            if self.text_options:
                output_text = choice(self.text_options)
            else:
                output_text = self.text

        # Next, if we should pass it through a prompt, do it.
        if self.prompt_handle is not None:
            prompt = GptPrompt.get_from_handle(client, self.prompt_handle)
            if prompt is None:
                raise SteamshipError(message=f"Unable to locate completion prompt: {self.prompt_handle}")
            output_text = prompt.complete_response(
                question=question,
                prompt=prompt,
                intent=intent,
                api_key=openai_api_key
            )

        return OiResponse(text=output_text, context=self.context, block_id=self.block_id)


class OiTrigger(CamelModel):
    # The text of the response
    text: str

    embedding_id: Optional[str] = None


class OiIntent(CamelModel):
    # An intent should probably have a name
    handle: str

    # The triggers which match this fact.
    triggers: List[OiTrigger] = None

    # The responses which can come from this fact.
    responses: List[OiResponse] = None

    # The file ID to associate it with
    file_id: str = None

    def add_to_index(self, index: EmbeddingIndex, file_id: str) -> List[OiTrigger]:
        """Add all the triggers to the embedding index, associated with the file ID containing the results."""
        ret = []
        new_additions = []
        for trigger in self.triggers:
            if trigger.embedding_id is not None:
                logging.info(f"Skipping index embed of trigger: {trigger.embedding_id} / {trigger.text}")
                ret.append(trigger)
            else:
                logging.info(f"Adding index embed of trigger: {trigger.text}")
                res = index.insert(trigger.text, external_id=file_id)
                item = res.item_ids[0]
                trigger.embedding_id = item.id
                ret.append(trigger)
                new_additions.append(trigger)

        if len(new_additions):
            logging.info(f"Added {len(new_additions)} new additions so embedding.")
            embed_task = index.embed()
            embed_task.wait()

            logging.info(f"Added {len(new_additions)} new additions so snapshotting.")
            index.create_snapshot()
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

        # Note: we're not storing the trigger here..
        # I think that's OK for now, but in a future version where the embedding indices are more tightly
        # woven into the Block & Tag structure, we probably should.
        return OiIntent(
            handle=handle,
            responses=responses,
            file_id=file.id
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

    def save(self, client: Steamship, index: EmbeddingIndex) -> "OiIntent":
        # Create a file that contains the responses
        response_file = self.to_steamship_file(client)

        # Now add the triggers to the index, linking each item with the file
        triggers = self.add_to_index(index, response_file.id)

        self.file_id = response_file.id
        self.triggers = triggers
        self.responses = OiIntent.from_steamship_file(response_file).responses

        return self


class OiFeed(CamelModel):
    # Name of the feed
    handle: str

    # List of intents in the feed
    intents: Optional[List[OiIntent]]

    # List of prompts in the feed
    prompts: Optional[List[GptPrompt]]

    def save(self, client: Steamship, index: EmbeddingIndex) -> "OiFeed":
        logging.info(f"Saving feed {self.handle} ")
        if self.intents:
            intents = [intent.save(client, intent, index) for intent in self.intents or []]
            self.intents = intents
        if self.prompts:
            prompts = [prompt.save(client) for prompt in self.prompts or []]
            self.prompts = prompts

        return self

class OiQuestion(CamelModel):
    text: str
    context: Optional[List[str]]

class OiAnswer(CamelModel):
    top_response: Optional[OiResponse]
