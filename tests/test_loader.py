"""TRD §14 — test_loader.py: all counts in §3; id<->index alignment."""

import json

import pytest

from src.data import loader
from src.data.loader import (
    EXPECTED_COUNTS,
    N_IMAGES,
    N_RECORDS,
    N_RELATION_KEYS,
    Question,
    all_images,
    assert_data_integrity,
    load_questions,
)


def test_startup_assertions_pass():
    assert_data_integrity()


def test_annotations_length():
    with loader.ANNOTATIONS_PATH.open(encoding="utf-8") as fh:
        assert len(json.load(fh)) == N_RECORDS


def test_query_all_length():
    with loader.QUERY_ALL_PATH.open(encoding="utf-8") as fh:
        assert len(json.load(fh)) == N_RECORDS


def test_relation_keys():
    with loader.RELATION_PATH.open(encoding="utf-8") as fh:
        assert len(json.load(fh)) == N_RELATION_KEYS


def test_id_index_alignment():
    with loader.ANNOTATIONS_PATH.open(encoding="utf-8") as fh:
        annotations = json.load(fh)
    with loader.QUERY_ALL_PATH.open(encoding="utf-8") as fh:
        queries = json.load(fh)
    for q in queries:
        assert annotations[q["id"] - 1]["id"] == q["id"]


def test_total_questions():
    assert len(load_questions()) == N_RECORDS


@pytest.mark.parametrize("qtype,count", sorted(EXPECTED_COUNTS.items()))
def test_qtype_counts(qtype, count):
    qs = load_questions(qtype)
    assert len(qs) == count
    assert all(q.qtype == qtype for q in qs)


def test_question_schema():
    q = load_questions("discriminative-attribute-state")[0]
    assert isinstance(q, Question)
    assert isinstance(q.id, int)
    assert q.image.startswith("AMBER_") and q.image.endswith(".jpg")
    assert q.query.endswith("?")
    assert q.truth in ("yes", "no")


def test_generative_truth_is_a_list():
    assert isinstance(load_questions("generative")[0].truth, list)


def test_discriminative_truths_are_yes_no():
    for qtype in EXPECTED_COUNTS:
        if qtype == "generative":
            continue
        assert {q.truth for q in load_questions(qtype)} <= {"yes", "no"}


def test_unknown_qtype_raises():
    with pytest.raises(ValueError):
        load_questions("not-a-real-qtype")


def test_all_images_numeric_order():
    images = all_images()
    assert len(images) == N_IMAGES
    indices = [int(loader.IMAGE_RE.match(n).group(1)) for n in images]
    assert indices == list(range(1, N_IMAGES + 1))
    assert images[0] == "AMBER_1.jpg"
    assert images[-1] == "AMBER_1004.jpg"


def test_all_images_matches_question_images():
    assert set(all_images()) == {q.image for q in load_questions()}
