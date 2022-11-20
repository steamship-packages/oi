"""Unit tests for the package."""
import logging
from typing import Optional

import pytest
from steamship import Steamship, PackageInstance

from openai import complete
from src.model import OiQuestion, OiFeed, OiIntent, OiTrigger, OiResponse, OiResponseType, GptPrompt
from src.api import OiPackage
import string
import random

# If this isn't present, Localstack won't show logs
logging.getLogger().setLevel(logging.INFO)

def random_name() -> str:
    """Returns a random name suitable for a handle that has low likelihood of colliding with another.

    Output format matches test_[a-z0-9]+, which should be a valid handle.
    """
    letters = string.digits + string.ascii_letters
    return f"test-{''.join(random.choice(letters) for _ in range(10))}".lower()  # noqa: S311


HOW_TO_REBASE = OiIntent(
    handle="how-to-rebase",
    triggers=[
        OiTrigger(text="how do i reset to upstream main"),
        OiTrigger(text="i want to reset my branch")
    ],
    responses=[
        OiResponse(
            text="""git checkout master\ngit pull upstream master\ngit reset --hard upstream/master"""
        )
    ]
)

HOW_TO_GET_IN_OFFICE = OiIntent(
    handle="how-to-get-in-office",
    triggers=[
        OiTrigger(text="how do i unlock the office door"),
        OiTrigger(text="how do i get in the office")
    ],
    responses=[
        OiResponse(
            text="""Punch the code 1 2 3 4 on the front door of the ground floor"""
        ),
        OiResponse(
            text="""Ring the front desk at 555-1212""",
            context=["#afterhours"]
        )
    ]
)

INTENT_WITH_OPTIONS = OiIntent(
    handle="whats-for-dinner",
    triggers=[
        OiTrigger(text="what's for dinner?")
    ],
    responses=[
        OiResponse(
            type=OiResponseType.SHUFFLE,
            text_options=[
                "Pizza",
                "Hamburger"
            ]
        )
    ]
)

INTENT_WITH_GPT = OiIntent(
    handle="tell-a-joke",
    triggers=[
        OiTrigger(text="tell me a fact about baseball"),
        OiTrigger(text="tell me a fact about america"),
        OiTrigger(text="teach me something about america")
    ],
    responses=[
        OiResponse(
            prompt_handle="pass-through",
            text="""Below is a list of the most interesting simple facts in the world.

Q: Tell me a fact about baseball
A: A baseball game has nine innings.

Q: Teach me something about pianos.
A: A piano has 88 keys.
            
Q: What do you know about America?
A: The capital is in Washington DC.

Q: {question_text}
A: """,
        )
    ]
)

PASS_THROUGH_PROMPT = GptPrompt(
    handle="pass-through",
    text="""{response_text}""",
    stop="\n\n"
)

TEST_FEED = OiFeed(
    handle="test-feed",
    intents=[
        HOW_TO_REBASE,
        HOW_TO_GET_IN_OFFICE,
        INTENT_WITH_OPTIONS,
        INTENT_WITH_GPT
    ],
    prompts=[
        PASS_THROUGH_PROMPT
    ]
)


GLOBAL_OI: Optional[OiPackage] = None


def test_context_matching():
    assert HOW_TO_GET_IN_OFFICE.responses[0] == HOW_TO_GET_IN_OFFICE.top_response()
    assert HOW_TO_GET_IN_OFFICE.responses[1] == HOW_TO_GET_IN_OFFICE.top_response(["#afterhours"])


def test_context_shuffle():
    values = set()
    for i in range(10):
        r = INTENT_WITH_OPTIONS.responses[0].complete_response()
        values.add(r.text)
    assert len(values) == 2
    assert INTENT_WITH_OPTIONS.responses[0].text_options[0] in values
    assert INTENT_WITH_OPTIONS.responses[0].text_options[1] in values


@pytest.fixture
def oi():
    global GLOBAL_OI
    if GLOBAL_OI:
        return GLOBAL_OI
    client = Steamship(workspace=random_name())
    GLOBAL_OI = OiPackage(client=client)
    resp = GLOBAL_OI.learn_feed(feed=TEST_FEED)
    assert isinstance(resp, OiFeed)
    assert len(resp.intents) == 4
    for intent in resp.intents:
        assert intent.file_id is not None
        assert len(intent.responses) > 0
        for response in intent.responses:
            assert response.block_id is not None
        assert len(intent.triggers) > 0
        for prompt in intent.triggers:
            assert prompt.embedding_id is not None
    return GLOBAL_OI


testdata = [
    ("How do I rebase?", [], HOW_TO_REBASE.responses[0].text),
    ("How do I fix main?", [], HOW_TO_REBASE.responses[0].text),
    ("I'm stuck outside the office", [], HOW_TO_GET_IN_OFFICE.responses[0].text),
    ("I'm stuck outside the office", ["#afterhours"], HOW_TO_GET_IN_OFFICE.responses[1].text),
]

@pytest.mark.parametrize("question,context,expected", testdata)
def test_response(oi: OiPackage, question, context, expected):
    """You can test your app like a regular Python object."""
    a = oi.query(question=OiQuestion(text=question, context=context))
    assert a is not None
    assert a.top_response is not None
    assert a.top_response.text is not None
    assert a.top_response.text == expected

def test_gpt(oi: OiPackage):
    """You can test your app like a regular Python object."""
    a = oi.query(question=OiQuestion(text="Tell me something about France"))
    assert a is not None
    assert a.top_response is not None
    assert a.top_response.text is not None

def test_generate(oi: OiPackage):
    """You can test your app like a regular Python object."""
    client = Steamship(workspace=random_name())
    oi = OiPackage(client=client)
    prompt = "Tell me something"
    res = complete(oi.config.openai_api_key, prompt)
    assert res is not None

