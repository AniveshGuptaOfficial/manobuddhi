"""Checks on the built splits. Skipped until `python data/build_splits.py` has run."""
import pytest

from engine.data.splits import BENCHMARKS, CALIB_N, SPLITS, load_split, manifest

pytestmark = pytest.mark.skipif(not (SPLITS / "manifest.json").exists(),
                                reason="run python data/build_splits.py first")


def test_all_benchmarks_and_splits_present():
    m = manifest()
    for name, (_, splits) in BENCHMARKS.items():
        assert set(m["benchmarks"][name]["splits"]) == set(splits), name


def test_calibration_sets_are_exactly_500():
    m = manifest()
    for name, b in m["benchmarks"].items():
        if "calib" in b["splits"]:
            assert b["splits"]["calib"]["n"] == CALIB_N, name


def test_no_leakage():
    m = manifest()
    assert m["leakage"]["train_or_calib_in_any_eval"] == 0
    assert all(b["cross_split_overlap"] == 0 for b in m["benchmarks"].values())


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_items_have_question_and_gold(name):
    for split in BENCHMARKS[name][1]:
        items = load_split(name, split)
        assert items, (name, split)
        assert all(i["question"].strip() and str(i["gold"]).strip() for i in items), (name, split)
        assert len({i["id"] for i in items}) == len(items), f"duplicate ids in {name}/{split}"
