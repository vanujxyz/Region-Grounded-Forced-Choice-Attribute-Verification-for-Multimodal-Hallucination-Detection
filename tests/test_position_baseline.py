"""D-009 — the AMBER pair-ordering artifact, and the guards against leaking it."""

import inspect

import pytest

from src.data import pairs as pairs_mod
from src.data.pairs import AttrPair, load_attr_pairs
from src.eval.metrics import accuracy
from src.modules import position_baseline
from src.modules.position_baseline import audit_id_ordering, predict


def test_true_attribute_always_has_the_lower_id():
    """The artifact itself: 2774/2774, zero exceptions."""
    ps = load_attr_pairs()
    a = audit_id_ordering(ps)
    assert a["n_pairs"] == 2774
    assert a["positive_has_lower_id"] == 2774
    assert a["fraction"] == 1.0


def test_pair_ids_are_always_adjacent():
    """Stronger than stated: the id gap is always exactly 1."""
    assert audit_id_ordering(load_attr_pairs())["distinct_id_gaps"] == [1]


def test_position_only_baseline_scores_100_percent():
    """A model-free, image-free detector achieves perfect accuracy."""
    recs = predict(load_attr_pairs())
    assert len(recs) == 5548
    assert accuracy(recs) == 1.0


def test_baseline_is_labelled_as_an_artifact():
    assert position_baseline.IS_BENCHMARK_ARTIFACT is True
    assert "ARTIFACT" in position_baseline.LABEL
    assert all(r["benchmark_artifact"] is True for r in predict(load_attr_pairs()[:5]))


# --------------------------------------------------------------------------
# D-009 consequence 2: option order must never derive from question id.
# --------------------------------------------------------------------------
def test_option_order_is_invariant_to_question_id():
    """Rebuilding a pair with ids swapped/scrambled must not change the options."""
    for p in load_attr_pairs()[:500]:
        swapped = AttrPair(
            image=p.image,
            obj=p.obj,
            positive_attr=p.positive_attr,
            negative_attr=p.negative_attr,
            positive_id=p.negative_id,   # ids deliberately swapped
            negative_id=p.positive_id,
        )
        assert swapped.options == p.options


def test_option_order_is_invariant_to_huge_id_perturbation():
    for p in load_attr_pairs()[:200]:
        perturbed = AttrPair(
            image=p.image, obj=p.obj,
            positive_attr=p.positive_attr, negative_attr=p.negative_attr,
            positive_id=999_999, negative_id=1,
        )
        assert perturbed.options == p.options


def test_option_order_depends_only_on_attribute_text():
    """Options are a pure function of the two attribute strings."""
    a = AttrPair(image="AMBER_1.jpg", obj="sky", positive_attr="sunny",
                 negative_attr="gloomy", positive_id=1005, negative_id=1006)
    b = AttrPair(image="AMBER_9.jpg", obj="sky", positive_attr="gloomy",
                 negative_attr="sunny", positive_id=7, negative_id=3)
    assert a.options == b.options == ["gloomy", "sunny"]


def test_options_property_does_not_reference_ids():
    """Static guard: the ordering code cannot see an id even by accident."""
    src = inspect.getsource(AttrPair.options.fget)
    assert "_id" not in src, "option ordering must not reference any id field"


def test_pair_polarity_is_decided_by_truth_not_by_id():
    """The real risk D-009 creates: assigning positive/negative by id order.

    If _build_pairs ever picked the positive attribute as "the lower id" instead
    of "the one whose gold answer is yes", every cell would inherit the artifact
    silently. Guard it statically -- the two are indistinguishable on real AMBER
    data precisely because the artifact is 100%, so a behavioural test cannot
    catch it.
    """
    src = inspect.getsource(pairs_mod._build_pairs)
    assert 'm[1] == "yes"' in src, "polarity must be selected on the truth field"
    assert 'm[1] == "no"' in src, "polarity must be selected on the truth field"
    for forbidden in ("positive_id <", "positive_id >", "min(", "max(", "sorted(members"):
        assert forbidden not in src, (
            f"pairs.py compares ids ({forbidden!r}) when deciding pair polarity; "
            "polarity must come from the gold answer only"
        )


def test_options_never_derive_from_id_anywhere_in_src():
    """Codebase-wide: no module orders options by id (D-009 consequence 2)."""
    import pathlib as _pl

    root = _pl.Path(inspect.getsourcefile(pairs_mod)).resolve().parents[2]
    offenders = []
    for path in sorted((root / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pattern in ("key=lambda o: o.positive_id", "sort(key=lambda a: a.id)"):
            if pattern in text:
                offenders.append(f"{path.name}: {pattern}")
    assert not offenders, f"option ordering derived from ids: {offenders}"


@pytest.mark.parametrize("pos,neg", [("sunny", "gloomy"), ("gloomy", "sunny")])
def test_options_identical_regardless_of_which_is_positive(pos, neg):
    p = AttrPair(image="AMBER_1.jpg", obj="sky", positive_attr=pos,
                 negative_attr=neg, positive_id=1, negative_id=2)
    assert p.options == ["gloomy", "sunny"]
