"""TRD §0 / D-031 — the detection cache.

The property that matters: a cache may make a run faster; it must never make it
different. Every test here defends that.
"""

import json

import pytest

from src.config import load_config
from src.modules.crop_cache import CropCache, _settings_fingerprint


@pytest.fixture
def cfg():
    return load_config()


def _cache(cfg, tmp_path, **kw):
    return CropCache(cfg, path=tmp_path, **kw)


def test_roundtrip(cfg, tmp_path):
    c = _cache(cfg, tmp_path)
    assert c.get("AMBER_1.jpg", "sky") is None
    c.put("AMBER_1.jpg", "sky", (1, 2, 3, 4), False, 0.42)
    hit = c.get("AMBER_1.jpg", "sky")
    assert hit["box"] == [1, 2, 3, 4]
    assert hit["fell_back"] is False
    assert hit["detector_score"] == 0.42


def test_survives_reload(cfg, tmp_path):
    a = _cache(cfg, tmp_path)
    a.put("AMBER_1.jpg", "sky", (1, 2, 3, 4), False, 0.42)
    a.save()
    b = _cache(cfg, tmp_path)
    assert b.get("AMBER_1.jpg", "sky")["box"] == [1, 2, 3, 4]


def test_fallback_entries_round_trip(cfg, tmp_path):
    """A fallback is a legitimate cached outcome, with a null score."""
    a = _cache(cfg, tmp_path)
    a.put("AMBER_2.jpg", "unicorn", (0, 0, 100, 100), True, None)
    a.save()
    hit = _cache(cfg, tmp_path).get("AMBER_2.jpg", "unicorn")
    assert hit["fell_back"] is True
    assert hit["detector_score"] is None


def test_keys_are_distinct_per_phrase(cfg, tmp_path):
    c = _cache(cfg, tmp_path)
    c.put("AMBER_1.jpg", "sky", (1, 1, 2, 2), False, 0.1)
    c.put("AMBER_1.jpg", "grass", (3, 3, 4, 4), False, 0.2)
    assert c.get("AMBER_1.jpg", "sky")["box"] == [1, 1, 2, 2]
    assert c.get("AMBER_1.jpg", "grass")["box"] == [3, 3, 4, 4]


# --------------------------------------------------------------------------
# The fingerprint: any detector setting that can change a detection
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field,value",
    [
        ("threshold", 0.25),
        ("nms_iou", 0.7),
        ("revision", "0" * 40),
        ("model_id", "some/other-model"),
        ("prompt_template", "a picture of a {phrase}"),
    ],
)
def test_fingerprint_changes_with_every_detector_setting(cfg, field, value):
    import copy

    other = copy.deepcopy(cfg)
    other["detector"][field] = value
    assert _settings_fingerprint(other) != _settings_fingerprint(cfg), (
        f"changing detector.{field} did not change the cache fingerprint; "
        "a stale entry could be reused across different settings"
    )


def test_fingerprint_is_stable_for_identical_settings(cfg):
    import copy

    assert _settings_fingerprint(copy.deepcopy(cfg)) == _settings_fingerprint(cfg)


def test_different_settings_use_different_files(cfg, tmp_path):
    import copy

    other = copy.deepcopy(cfg)
    other["detector"]["threshold"] = 0.25
    assert _cache(cfg, tmp_path).path != _cache(other, tmp_path).path


def test_mismatched_fingerprint_is_refused(cfg, tmp_path):
    """A file whose fingerprint disagrees must raise, never be silently used."""
    c = _cache(cfg, tmp_path)
    c.put("AMBER_1.jpg", "sky", (1, 2, 3, 4), False, 0.42)
    c.save()
    with c.path.open(encoding="utf-8") as fh:
        blob = json.load(fh)
    blob["fingerprint"] = "deadbeefdeadbeef"
    with c.path.open("w", encoding="utf-8") as fh:
        json.dump(blob, fh)
    with pytest.raises(RuntimeError, match="Refusing to use it"):
        _cache(cfg, tmp_path)


def test_corrupt_cache_raises_rather_than_being_ignored(cfg, tmp_path):
    c = _cache(cfg, tmp_path)
    c.path.parent.mkdir(parents=True, exist_ok=True)
    c.path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unreadable"):
        _cache(cfg, tmp_path)


def test_disabled_cache_never_hits(cfg, tmp_path):
    c = _cache(cfg, tmp_path, enabled=False)
    c.put("AMBER_1.jpg", "sky", (1, 2, 3, 4), False, 0.42)
    assert c.get("AMBER_1.jpg", "sky") is None
    c.save()
    assert not c.path.exists()


def test_stats(cfg, tmp_path):
    c = _cache(cfg, tmp_path)
    c.put("AMBER_1.jpg", "sky", (1, 2, 3, 4), False, 0.42)
    c.get("AMBER_1.jpg", "sky")
    c.get("AMBER_9.jpg", "nothing")
    s = c.stats()
    assert s["hits"] == 1 and s["misses"] == 1
    assert s["hit_rate"] == 0.5
    assert s["entries"] == 1


def test_hit_rate_not_computed_when_unused(cfg, tmp_path):
    assert _cache(cfg, tmp_path).stats()["hit_rate"] == "NOT_COMPUTED"
