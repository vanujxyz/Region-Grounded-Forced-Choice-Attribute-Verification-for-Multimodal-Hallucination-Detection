"""SHROOM-Vis: the dataset contract, and the artifacts it must NOT have.

The point of a second dataset is to test the method, not to be flattered by it.
These tests assert the properties that make the comparison meaningful -- above
all that SHROOM-Vis does not reproduce AMBER's id-ordering artifact, which
would let a strategy that never opens an image score 1.0.
"""

import json
import re

import pytest

from src.config import ROOT
from src.data import shroom
from src.data.pairs import ACTION_RE, STATE_RE

pytestmark = pytest.mark.skipif(
    not shroom.ANNOTATIONS_PATH.exists(),
    reason="SHROOM-Vis not built; run scripts/shroom_build.py",
)


def test_annotations_and_queries_are_parallel_and_aligned():
    anns = json.loads(shroom.ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    qs = json.loads(shroom.QUERY_ALL_PATH.read_text(encoding="utf-8"))
    assert len(anns) == len(qs)
    for i, (a, q) in enumerate(zip(anns, qs), start=1):
        assert a["id"] == i, f"annotations[{i - 1}] holds id {a['id']}"
        assert q["id"] == i


def test_every_question_parses_with_ambers_regexes():
    for q in shroom.load_questions():
        rx = STATE_RE if q.qtype.endswith("state") else ACTION_RE
        m = rx.match(q.query)
        assert m is not None, f"{q.id}: {q.query!r} does not parse"
        # The object must be a single token: AMBER's regexes are non-greedy on
        # both fields, so a two-word object silently steals a word from the
        # attribute.
        assert " " not in m.group("obj"), f"{q.id}: multi-word object {m.group('obj')!r}"


def test_pairs_are_clean_and_balanced():
    pairs = shroom.load_attr_pairs()
    assert len(pairs) * 2 == len(shroom.load_questions())
    for p in pairs:
        assert p.positive_attr != p.negative_attr
        assert p.options == sorted(p.options)
        assert p.positive_id != p.negative_id


def test_gold_is_exactly_balanced():
    """One yes and one no per pair, so forced choice gets no free prior edge
    that the baseline lacks -- the same property AMBER has (D-020)."""
    qs = shroom.load_questions()
    yes = sum(1 for q in qs if q.truth == "yes")
    assert yes * 2 == len(qs), f"{yes} yes of {len(qs)} questions"


def test_the_id_ordering_artifact_is_absent():
    """AMBER gives the true attribute the lower id in 2774/2774 pairs, so
    "answer yes to the lower id" scores 1.0000 without opening an image
    (README.md). SHROOM-Vis must not reproduce that."""
    pairs = shroom.load_attr_pairs()
    lower = sum(1 for p in pairs if p.positive_id < p.negative_id)
    rate = lower / len(pairs)
    assert 0.45 < rate < 0.55, (
        f"positive attribute holds the lower id in {rate:.4f} of pairs; a "
        "seeded coin flip should put this near 0.5"
    )


def test_no_attribute_is_a_word_of_its_own_object():
    """`Is the red car red?` is unanswerable by construction."""
    for p in shroom.load_attr_pairs():
        obj_words = set(p.obj.split())
        for attr in (p.positive_attr, p.negative_attr):
            assert not (obj_words & set(attr.split())), f"{p.obj} / {attr}"


def test_splits_are_disjoint_and_by_image():
    splits = shroom.load_splits()
    assert not (set(splits["dev"]) & set(splits["test"]))
    assert len(splits["dev"]) + len(splits["test"]) == len(shroom.all_images())


def test_reading_the_test_split_is_gated(monkeypatch):
    monkeypatch.delenv(shroom.TEST_ENV_VAR, raising=False)
    with pytest.raises(PermissionError):
        shroom.get_split("test")


def test_pair_meta_agrees_with_the_reconstructed_pairs():
    """pair_meta.json is provenance; the evaluation path rebuilds the pairs by
    parsing queries. A disagreement means the build is stale."""
    meta = shroom.load_pair_meta()
    pairs = {p.positive_id: p for p in shroom.load_attr_pairs()}
    assert set(meta) == {str(k) for k in pairs}
    for pid, m in meta.items():
        p = pairs[int(pid)]
        assert (m["obj"], m["positive_attr"], m["negative_attr"]) == (
            p.obj,
            p.positive_attr,
            p.negative_attr,
        )
        assert m["negative_id"] == p.negative_id


def test_out_of_vocabulary_objects_are_present_and_labelled():
    """SHROOM-Vis deliberately reaches outside AMBER's closed 340-object
    vocabulary; in-vocab and out-of-vocab must be separable at analysis time."""
    meta = shroom.load_pair_meta()
    vocab = set(json.loads((ROOT / "data" / "amber" / "relation.json").read_text(encoding="utf-8")))
    for m in meta.values():
        assert m["in_amber_vocab"] == (m["obj"] in vocab)
    out = sum(1 for m in meta.values() if not m["in_amber_vocab"])
    assert out > 0, "no out-of-vocabulary objects; the vocab test is vacuous"


def test_filenames_carry_no_index_the_pipeline_could_read():
    """Nothing downstream may parse an ordering out of a SHROOM filename."""
    for name in shroom.all_images():
        assert not re.fullmatch(r"AMBER_\d+\.jpg", name)
