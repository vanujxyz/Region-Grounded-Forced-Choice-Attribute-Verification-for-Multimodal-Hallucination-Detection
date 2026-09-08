"""TRD §13 — demo claim splitting and the demo-only antonym boundary."""

import pytest

from src.demo import contrasting_attribute, split_claim
from src.modules.attribute import load_antonyms


@pytest.mark.parametrize(
    "claim,obj,attr,kind",
    [
        ("the sky is sunny", "sky", "sunny", "state"),
        ("the sky is gloomy", "sky", "gloomy", "state"),
        ("the grass is green", "grass", "green", "state"),
        ("the mountain is tall", "mountain", "tall", "state"),
    ],
)
def test_split_state_claims(claim, obj, attr, kind):
    assert split_claim(claim) == (obj, attr, kind)


@pytest.mark.parametrize(
    "claim,obj,attr",
    [
        ("the man is sitting", "man", "sit"),
        ("the dog is running", "dog", "run"),
    ],
)
def test_split_action_claims(claim, obj, attr):
    o, a, kind = split_claim(claim)
    assert (o, a) == (obj, attr)
    assert kind == "action"


def test_split_adjectival_claim():
    o, a, kind = split_claim("a red apple")
    assert (o, a, kind) == ("apple", "red", "state")


def test_unparseable_claim_raises():
    with pytest.raises(ValueError, match="could not extract"):
        split_claim("aardvark")


def test_contrasting_attribute_from_the_map():
    m = load_antonyms()
    assert contrasting_attribute("sunny", m) == "gloomy"
    assert contrasting_attribute("clean", m) == "dirty"


def test_unmapped_attribute_refuses_to_invent_a_contrast():
    """TRD §13: the demo must not fabricate an opposite."""
    with pytest.raises(ValueError, match="will not"):
        contrasting_attribute("zzz_not_an_attribute", {})


def test_demo_map_is_the_same_file_as_dext_but_a_separate_use():
    """The map is shared; the rule is that the EVALUATION path never uses it.

    Cells A-D take their options from the dataset pairs. Only Cell D-ext (an
    explicitly-labelled control) and the demo consult the antonym map.
    """
    import inspect

    from src.modules import attribute

    src = inspect.getsource(attribute.predict_pair)
    assert "antonym" not in src.lower(), (
        "predict_pair (cells A-D) must never consult the antonym map"
    )
