"""Unit tests for the package."""
import logging
from typing import Optional

import pytest
from steamship import Steamship, PackageInstance

from src.model import OiQuestion, OiFeed, OiIntent, OiPrompt, OiResponse, OiResponseType
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
    prompts=[
        OiPrompt(text="how do i reset to upstream main"),
        OiPrompt(text="i want to reset my branch")
    ],
    responses=[
        OiResponse(
            text="""git checkout master\ngit pull upstream master\ngit reset --hard upstream/master"""
        )
    ]
)

HOW_TO_GET_IN_OFFICE = OiIntent(
    handle="how-to-get-in-office",
    prompts=[
        OiPrompt(text="how do i unlock the office door"),
        OiPrompt(text="how do i get in the office")
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
    prompts=[
        OiPrompt(text="what's for dinner?")
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

TEST_FEED = OiFeed(
    handle="test-feed",
    intents=[
        HOW_TO_REBASE,
        HOW_TO_GET_IN_OFFICE,
        INTENT_WITH_OPTIONS
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
    assert len(resp.intents) == 3
    for intent in resp.intents:
        assert intent.file_id is not None
        assert len(intent.responses) > 0
        for response in intent.responses:
            assert response.block_id is not None
        assert len(intent.prompts) > 0
        for prompt in intent.prompts:
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
