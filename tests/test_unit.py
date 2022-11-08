"""Unit tests for the package."""

from steamship import Steamship

from src.model import OiQuestion, OiFeed, OiIntent, OiPrompt, OiResponse
from src.api import OiPackage
import string
import random

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
        )
    ]
)

TEST_FEED = OiFeed(
    handle="test-feed",
    intents=[
        HOW_TO_REBASE,
        HOW_TO_GET_IN_OFFICE
    ]
)


def test_response():
    """You can test your app like a regular Python object."""
    client = Steamship(workspace=random_name())
    oi = OiPackage(client=client)

    resp = oi.learn_feed(feed=TEST_FEED)
    assert resp == [True, True] # A silly return type, but I'm trying to do this in half a day :)

    q = OiQuestion(text="How do I rebase?")
    a = oi.query(question=q)
    print(a)
    assert a is not None
    assert a.top_response.text == HOW_TO_REBASE.responses[0].text

