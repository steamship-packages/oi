"""Unit tests for the package."""

from steamship import Steamship

from src.model import OiQuestion
from src.api import OiPackage


def test_response():
    """You can test your app like a regular Python object."""
    client = Steamship()
    oi = OiPackage(client=client)

    q = OiQuestion(text="How do I rebase?")
    a = oi.query(question=q)
    assert a is not None

