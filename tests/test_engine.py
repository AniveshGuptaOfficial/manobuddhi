"""Engine unit tests (no DB): placeholder router, threshold, grading, stats."""
from engine.grading import grade, is_lure
from engine.interface import RouteRequest, RouterSettings
from engine.placeholder import get_router
from engine.stats import auroc, percentile
from engine.threshold import choose_threshold

HARD = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much is the ball? " * 3


def test_placeholder_is_deterministic():
    r, s = get_router(), RouterSettings()
    assert r.route(RouteRequest("hello", s)) == r.route(RouteRequest("hello", s))


def test_force_route_and_paths():
    r = get_router()
    slow = r.route(RouteRequest(HARD, RouterSettings(), force_route="escalated"))
    assert slow.route == "escalated" and slow.paths[-1].path == "slow" and slow.slow_answer
    fast = r.route(RouteRequest(HARD, RouterSettings(), force_route="fast"))
    assert fast.route == "fast" and all(p.path != "slow" for p in fast.paths)


def test_reuse_method_per_config():
    r = get_router()
    a = r.route(RouteRequest(HARD, RouterSettings(mode="single_model_nothink_think"), force_route="escalated"))
    b = r.route(RouteRequest(HARD, RouterSettings(mode="small_large"), force_route="escalated"))
    assert a.reuse_method == "kv_prefix" and b.reuse_method == "speculative_draft"


def test_threshold_budget():
    assert choose_threshold(RouterSettings(latency_budget_ms=500)) > choose_threshold(RouterSettings())
    assert choose_threshold(RouterSettings(threshold_mode="fixed", fixed_threshold=0.42)) == 0.42


def test_grading():
    item = {"answer_type": "numeric", "gold": "5", "meta": {"lure": "10"}}
    assert grade(item, "work...\nAnswer: 5") and not grade(item, "Answer: $0.10")
    assert is_lure(item, "Answer: 10")
    assert grade({"answer_type": "math_expr", "gold": r"\frac{1}{3}"}, r"\boxed{\dfrac{1}{3}}")
    assert grade({"answer_type": "yes_no", "gold": "no"}, "Answer: No.")
    assert grade({"answer_type": "multiple_choice", "gold": "B", "choices": []}, "Answer: B")
    assert grade({"answer_type": "text_aliases", "gold": "Paris", "aliases": ["Paris"]}, "Answer: It's Paris")
    assert grade({"answer_type": "free_text", "gold": "x"}, "y") is None
    # thinking-model endings without an "Answer:" line (F4 Config A smoke, gsm8k-test-517)
    assert grade(item, "so 2 x 9 = 5\n\n### Final Answer\n\n$$\n\\boxed{5}\n$$")
    assert grade(item, "total is 5 apples\n\n---\n$$")


def test_stats():
    assert auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert auroc([0.5, 0.5], [1, 0]) == 0.5 and auroc([1], [1]) is None
    assert percentile([1, 2, 3, 4, 5], 50) == 3
