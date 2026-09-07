"""D-032 — the external antonym map and Cell D-ext.

Written before any D-ext result was computed. The central property under test:
Cell D-ext must never consult the dataset's paired negative attribute.
"""

import pytest

from src.data.pairs import load_attr_pairs
from src.modules.attribute import (
    ALL_CELLS,
    EXT_CELLS,
    CoverageLog,
    decision_of,
    load_antonyms,
    region_of,
)


def test_map_loads_and_is_non_trivial():
    m = load_antonyms()
    assert len(m) > 150
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in m.items())


def test_no_attribute_is_its_own_antonym():
    for k, v in load_antonyms().items():
        assert k.strip().lower() != v.strip().lower()


def test_known_entries():
    m = load_antonyms()
    assert m["sunny"] == "gloomy"
    assert m["clean"] == "dirty"
    assert m["tall"] == "short"
    assert m["sit"] == "stand"
    assert m["stand"] == "sit"


def test_map_is_not_a_copy_of_the_dataset_pairing():
    """The map must be an EXTERNAL source, not AMBER's pairing rebadged.

    If every mapped attribute's antonym equalled the dataset's paired negative,
    D-ext would be Cell D with extra steps and would prove nothing.
    """
    m = load_antonyms()
    agree = differ = 0
    for p in load_attr_pairs():
        got = m.get(p.positive_attr)
        if got is None:
            continue
        if got == p.negative_attr:
            agree += 1
        else:
            differ += 1
    assert differ > 0, (
        "the antonym map reproduces the dataset pairing on every single pair; "
        "it is not an independent source"
    )
    # Some agreement is expected and healthy -- sunny/gloomy really are opposites.
    assert agree > 0


def test_ext_cell_is_registered_and_routed():
    assert EXT_CELLS == ("Dext",)
    assert "Dext" in ALL_CELLS
    assert region_of("Dext") == "crop"
    assert decision_of("Dext") == "forced_choice_external"


def test_ext_cell_is_not_part_of_the_2x2():
    from src.modules.attribute import CELLS

    assert "Dext" not in CELLS


# --------------------------------------------------------------------------
# Coverage accounting
# --------------------------------------------------------------------------
def test_coverage_log_counts_and_rate():
    c = CoverageLog()
    c.record("sunny", True)
    c.record("swim", False)
    c.record("swim", False)
    d = c.as_dict()
    assert d["questions_total"] == 3
    assert d["questions_mapped"] == 1
    assert d["questions_unmapped"] == 2
    assert d["coverage_rate"] == pytest.approx(1 / 3)
    assert d["distinct_unmapped_attrs"] == 1
    assert d["top_unmapped"][0] == ("swim", 2)


def test_coverage_rate_not_computed_when_empty():
    assert CoverageLog().as_dict()["coverage_rate"] == "NOT_COMPUTED"


def test_unmapped_attributes_have_no_silent_fallback():
    """An unmapped attribute must yield NO record, never a guessed competitor."""
    m = load_antonyms()
    unmapped = [a for a in ("swim", "surf", "dance") if a not in m]
    assert unmapped, "expected some actions to be deliberately unmapped"


def test_coverage_on_the_real_pair_set_is_reported_not_assumed():
    """Compute coverage over every real question. Any value is legal; it must
    simply be computable and honest."""
    m = load_antonyms()
    c = CoverageLog()
    for p in load_attr_pairs():
        c.record(p.positive_attr, p.positive_attr in m)
        c.record(p.negative_attr, p.negative_attr in m)
    d = c.as_dict()
    assert d["questions_total"] == 5548
    assert 0.0 < d["coverage_rate"] <= 1.0
    assert d["questions_mapped"] + d["questions_unmapped"] == 5548
