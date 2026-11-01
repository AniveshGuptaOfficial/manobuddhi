"""F5 tests and regressions: real probe-based router and downstream integration."""

from types import SimpleNamespace

import numpy as np
import pytest

from engine.interface import (
    PathResult,
    RouteRequest,
    RouteResult,
    RouterSettings,
)
from engine.real_router import (
    A2_FAST_MODEL,
    A2_PROBE_FEATURE,
    A2_SLOW_MODEL,
    A2_THRESHOLD,
    DEFAULT_PROBE_PATH,
    RealRouter,
)


# ---------------------------------------------------------
# Mock probe
# ---------------------------------------------------------

class FakeProbe:
    feature = A2_PROBE_FEATURE

    def __init__(self, risk):
        self.risk = float(risk)
        self.coef = np.zeros(1536, dtype=np.float32)

    def predict(self, x):
        assert x.shape == (1, 1536)
        return np.array([self.risk], dtype=np.float32)


# ---------------------------------------------------------
# Mock model
# ---------------------------------------------------------

class FakeModel:
    def __init__(self, name, answer, risk=None):
        self.name = name
        self.answer = answer
        self.risk = risk
        self.generate_calls = 0
        self.probe_calls = 0

    def generate(self, prompts, **kwargs):
        self.generate_calls += 1

        generation = SimpleNamespace(
            model=self.name,
            text=self.answer,
            raw_text=self.answer,
            prompt_tokens=20,
            completion_tokens=10,
            finished=True,
            ttft_ms=50.0,
            latency_ms=200.0,
            gpu_seconds=0.0,
            gpu_util_pct=None,
            gpu_mem_mb=None,
            energy_j=None,
            conf_seq=0.9,
        )

        return [(generation, [1, 2, 3], [4, 5, 6])]

    def probe_features(self, prompt_ids, comp_ids, layers):
        self.probe_calls += 1

        assert layers == [20]

        return SimpleNamespace(
            features={
                A2_PROBE_FEATURE: np.zeros(1536, dtype=np.float16)
            },
            cost={
                "latency_ms": 15.0,
                "gpu_seconds": 0.0,
                "gpu_util_pct": None,
                "energy_j": None,
            },
        )


# ---------------------------------------------------------
# Test helpers
# ---------------------------------------------------------

def make_router(risk):
    probe = FakeProbe(risk)

    fast = FakeModel(
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Fast answer",
    )

    slow = FakeModel(
        "Qwen/Qwen2.5-7B-Instruct",
        "Slow answer",
    )

    router = RealRouter(
        probe=probe,
        fast_model=fast,
        slow_model=slow,
        threshold=A2_THRESHOLD,
    )

    return router, fast, slow


def make_request(force_route=None, probe_enabled=True):
    settings = RouterSettings(
        mode="small_large",
        probe_enabled=probe_enabled,
        disagreement_enabled=False,
    )

    return RouteRequest(
        prompt="What is 2 + 2?",
        settings=settings,
        force_route=force_route,
    )


# ---------------------------------------------------------
# Core RealRouter Tests
# ---------------------------------------------------------

def test_router_initialization_does_not_load_models():
    router = RealRouter(probe=FakeProbe(0.05))

    assert router.name == "real"
    assert router.fast_model is None
    assert router.slow_model is None
    assert router.threshold == A2_THRESHOLD


def test_probe_below_threshold_accepts_fast_answer():
    router, fast, slow = make_router(A2_THRESHOLD - 0.01)

    result = router.route(make_request())

    assert result.route == "fast"
    assert result.final_answer == "Fast answer"
    assert result.fast_answer == "Fast answer"
    assert result.slow_answer is None

    assert result.probe_score < A2_THRESHOLD
    assert result.threshold == A2_THRESHOLD

    assert fast.generate_calls == 1
    assert fast.probe_calls == 1
    assert slow.generate_calls == 0

    assert [p.path for p in result.paths] == ["fast", "probe"]


def test_probe_at_threshold_escalates():
    router, fast, slow = make_router(A2_THRESHOLD)

    result = router.route(make_request())

    assert result.route == "escalated"
    assert result.final_answer == "Slow answer"
    assert result.fast_answer == "Fast answer"
    assert result.slow_answer == "Slow answer"

    assert result.probe_score == A2_THRESHOLD

    assert fast.generate_calls == 1
    assert fast.probe_calls == 1
    assert slow.generate_calls == 1

    assert [p.path for p in result.paths] == [
        "fast",
        "probe",
        "slow",
    ]


def test_probe_above_threshold_escalates():
    router, fast, slow = make_router(0.90)

    result = router.route(make_request())

    assert result.route == "escalated"
    assert result.final_answer == "Slow answer"
    assert slow.generate_calls == 1


def test_force_fast_skips_probe_and_slow_model():
    router, fast, slow = make_router(0.99)

    result = router.route(make_request(force_route="fast"))

    assert result.route == "fast"
    assert result.final_answer == "Fast answer"

    assert fast.generate_calls == 1
    assert fast.probe_calls == 0
    assert slow.generate_calls == 0

    assert [p.path for p in result.paths] == ["fast"]


def test_force_escalated_skips_fast_model():
    router, fast, slow = make_router(0.01)

    result = router.route(make_request(force_route="escalated"))

    assert result.route == "escalated"
    assert result.final_answer == "Slow answer"

    assert fast.generate_calls == 0
    assert fast.probe_calls == 0
    assert slow.generate_calls == 1

    assert [p.path for p in result.paths] == ["slow"]


def test_probe_disabled_accepts_fast_answer():
    router, fast, slow = make_router(0.99)

    result = router.route(make_request(probe_enabled=False))

    assert result.route == "fast"
    assert result.final_answer == "Fast answer"

    assert fast.generate_calls == 1
    assert fast.probe_calls == 0
    assert slow.generate_calls == 0

    assert [p.path for p in result.paths] == ["fast"]


def test_probe_path_records_its_cost():
    router, _, _ = make_router(0.01)

    result = router.route(make_request())

    probe_paths = [p for p in result.paths if p.path == "probe"]

    assert len(probe_paths) == 1
    assert probe_paths[0].latency_ms == 15.0
    assert probe_paths[0].gpu_seconds == 0.0


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        RealRouter(
            probe=FakeProbe(0.5),
            threshold=1.5,
        )


def test_wrong_probe_feature_rejected():
    probe = FakeProbe(0.5)
    probe.feature = "last_L20"

    with pytest.raises(ValueError):
        RealRouter(probe=probe)


def test_disagreement_routing_is_rejected_until_implemented():
    router, _, _ = make_router(0.05)

    request = make_request()
    request.settings.disagreement_enabled = True

    with pytest.raises(
        NotImplementedError,
        match="probe-only routing",
    ):
        router.route(request)


# ---------------------------------------------------------
# F5 Regression: Latency & Properties Calculation
# ---------------------------------------------------------

def test_total_latency_ms_regressions():
    """Verify latency accumulation across path combinations including probe."""
    base_kwargs = dict(
        route="fast",
        final_answer="A",
        fast_answer="A",
        slow_answer=None,
        conflict_score=0.1,
        probe_score=0.1,
        disagreement=None,
        logprob_confidence=0.9,
        threshold=0.5,
        threshold_mode="calibrated_probe",
        reuse_method=None,
        backend="hf-transformers",
        simulated=False,
    )

    # 1. Empty paths
    assert RouteResult(**base_kwargs, paths=[]).total_latency_ms == 0.0

    # 2. Fast only
    r_fast = RouteResult(
        **base_kwargs,
        paths=[PathResult(path="fast", model="m", text="a", latency_ms=100.0)],
    )
    assert r_fast.total_latency_ms == 100.0

    # 3. Fast + Probe
    r_fp = RouteResult(
        **base_kwargs,
        paths=[
            PathResult(path="fast", model="m", text="a", latency_ms=100.0),
            PathResult(path="probe", model="m", text="", latency_ms=18.5),
        ],
    )
    assert r_fp.total_latency_ms == 118.5

    # 4. Fast + Probe + Slow
    r_fps = RouteResult(
        **{
            **base_kwargs,
            "route": "escalated",
            "slow_answer": "B",
        },
        paths=[
            PathResult(path="fast", model="m", text="a", latency_ms=100.0),
            PathResult(path="probe", model="m", text="", latency_ms=18.5),
            PathResult(path="slow", model="m2", text="b", latency_ms=350.0),
        ],
    )
    assert r_fps.total_latency_ms == 468.5

    # 5. Fast + Fast Sample + Probe + Slow
    r_samp = RouteResult(
        **{
            **base_kwargs,
            "route": "escalated",
            "slow_answer": "B",
        },
        paths=[
            PathResult(path="fast", model="m", text="a", latency_ms=100.0),
            PathResult(path="fast_sample", model="m", text="s1", latency_ms=125.0),
            PathResult(path="probe", model="m", text="", latency_ms=18.5),
            PathResult(path="slow", model="m2", text="b", latency_ms=350.0),
        ],
    )
    # max(fast, fast_sample) + probe + slow = 125.0 + 18.5 + 350.0 = 493.5
    assert r_samp.total_latency_ms == 493.5

    # 6. Slow only (forced escalation)
    r_slow = RouteResult(
        **{
            **base_kwargs,
            "route": "escalated",
            "slow_answer": "B",
        },
        paths=[PathResult(path="slow", model="m2", text="b", latency_ms=350.0)],
    )
    assert r_slow.total_latency_ms == 350.0


def test_route_result_aggregated_properties():
    """Verify total_tokens, completion_tokens, and total_gpu_seconds."""
    paths = [
        PathResult(
            path="fast",
            model="fast-m",
            text="a",
            prompt_tokens=25,
            completion_tokens=15,
            gpu_seconds=0.12,
        ),
        PathResult(
            path="probe",
            model="fast-m",
            text="",
            prompt_tokens=0,
            completion_tokens=0,
            gpu_seconds=0.01,
        ),
        PathResult(
            path="slow",
            model="slow-m",
            text="b",
            prompt_tokens=30,
            completion_tokens=40,
            gpu_seconds=0.85,
        ),
    ]

    res = RouteResult(
        route="escalated",
        final_answer="B",
        fast_answer="A",
        slow_answer="B",
        conflict_score=0.8,
        probe_score=0.8,
        disagreement=None,
        logprob_confidence=0.5,
        threshold=0.07,
        threshold_mode="calibrated_probe",
        reuse_method=None,
        backend="hf-transformers",
        simulated=False,
        paths=paths,
    )

    assert res.total_tokens == (25 + 15) + 0 + (30 + 40)
    assert res.completion_tokens == 15 + 0 + 40
    assert res.total_gpu_seconds == pytest.approx(0.12 + 0.01 + 0.85)


# ---------------------------------------------------------
# F5 Regression: Backend Router & Settings Dispatch
# ---------------------------------------------------------

def test_active_router_resolution_and_cache():
    """Verify active_router creates correct instances, caches, and rejects unknowns."""
    from backend.router import active_router
    from engine.placeholder import PlaceholderRouter

    p = active_router("placeholder")
    assert isinstance(p, PlaceholderRouter)
    assert p.name == "placeholder"

    r = active_router("real")
    assert isinstance(r, RealRouter)
    assert r.name == "real"

    # LRU cache: identical instances returned
    assert active_router("placeholder") is p
    assert active_router("real") is r

    with pytest.raises(ValueError, match="Unknown router backend"):
        active_router("nonexistent_backend")


def test_settings_router_backend_validation_and_mapping():
    """Verify settings accepts real and placeholder, rejects invalid, maps to backend."""
    import pydantic
    from backend.settings import SettingsModel, SettingsUpdate, router_settings

    # Valid updates
    u1 = SettingsUpdate(router_backend="placeholder")
    assert u1.router_backend == "placeholder"
    u2 = SettingsUpdate(router_backend="real")
    assert u2.router_backend == "real"

    # Invalid update
    with pytest.raises(pydantic.ValidationError):
        SettingsUpdate(router_backend="invalid_backend")

    # Mapping to RouterSettings.backend
    rs_placeholder = router_settings({"router_backend": "placeholder"})
    assert rs_placeholder.backend == "placeholder"

    rs_real = router_settings({"router_backend": "real"})
    assert rs_real.backend == "real"


# ---------------------------------------------------------
# F5 Regression: Real Probe Loading & Model Factory
# ---------------------------------------------------------

def test_real_disk_probe_loading_and_prediction():
    """Verify loading real calibrated A2 probe file and executing prediction."""
    if not DEFAULT_PROBE_PATH.exists():
        pytest.skip("A2 probe file not present on disk")

    router = RealRouter()
    assert router.probe is not None
    assert router.probe.feature == A2_PROBE_FEATURE
    assert len(router.probe.coef) == 1536
    assert router.threshold == A2_THRESHOLD

    # Predict on dummy 1536-dim feature vector
    feat = np.zeros((1, 1536), dtype=np.float32)
    score = float(router.probe.predict(feat)[0])
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0


def test_model_factory_lazy_loading_and_force_routes():
    """Verify models are instantiated strictly on demand via model_factory."""
    created = []

    def mock_factory(name, slow=False):
        created.append((name, slow))
        return FakeModel(name, f"{'Slow' if slow else 'Fast'} answer")

    # 1. No models loaded on __init__
    router = RealRouter(probe=FakeProbe(0.01), model_factory=mock_factory)
    assert len(created) == 0

    # 2. Low risk: only fast model created
    res_fast = router.route(make_request())
    assert res_fast.route == "fast"
    assert len(created) == 1
    assert created[0] == (A2_FAST_MODEL, False)

    # 3. High risk: slow model created lazily
    router.probe = FakeProbe(0.99)
    res_esc = router.route(make_request())
    assert res_esc.route == "escalated"
    assert len(created) == 2
    assert created[1] == (A2_SLOW_MODEL, True)

    # 4. Force fast: fast model only
    created.clear()
    router_ff = RealRouter(probe=FakeProbe(0.99), model_factory=mock_factory)
    router_ff.route(make_request(force_route="fast"))
    assert len(created) == 1
    assert created[0] == (A2_FAST_MODEL, False)

    # 5. Force escalated: slow model only
    created.clear()
    router_fe = RealRouter(probe=FakeProbe(0.01), model_factory=mock_factory)
    router_fe.route(make_request(force_route="escalated"))
    assert len(created) == 1
    assert created[0] == (A2_SLOW_MODEL, True)


def test_cpu_hardware_guard_without_cuda(monkeypatch):
    """Verify attempting real HFModel creation on CPU raises informative RuntimeError."""
    router = RealRouter(probe=FakeProbe(0.5))
    try:
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    except ModuleNotFoundError:
        pass

    with pytest.raises(RuntimeError, match="Real model inference requires a CUDA GPU"):
        router._create_model(A2_FAST_MODEL)


# ---------------------------------------------------------
# F5 Regression: Detailed Error Conditions
# ---------------------------------------------------------

def test_missing_probe_file_raises_filenotfound():
    """Verify routing without probe object and missing probe file raises FileNotFoundError."""
    router = RealRouter(
        probe_path="results/a2/probe/nonexistent.npz",
        fast_model=FakeModel(A2_FAST_MODEL, "fast"),
        slow_model=FakeModel(A2_SLOW_MODEL, "slow"),
    )
    with pytest.raises(FileNotFoundError, match="A2 probe not found"):
        router.route(make_request())


def test_feature_dimension_mismatch_raises_valueerror():
    """Verify probe feature dimension mismatch is caught with clear error."""
    class BadDimModel(FakeModel):
        def probe_features(self, prompt_ids, comp_ids, layers):
            return SimpleNamespace(
                features={A2_PROBE_FEATURE: np.zeros(512, dtype=np.float16)},
                cost={"latency_ms": 10.0, "gpu_seconds": 0.0},
            )

    fast = BadDimModel(A2_FAST_MODEL, "fast")
    slow = FakeModel(A2_SLOW_MODEL, "slow")
    router = RealRouter(probe=FakeProbe(0.5), fast_model=fast, slow_model=slow)

    with pytest.raises(ValueError, match="Probe dimension mismatch"):
        router.route(make_request())


def test_missing_feature_key_raises_keyerror():
    """Verify missing expected feature key in probe extraction raises KeyError."""
    class MissingKeyModel(FakeModel):
        def probe_features(self, prompt_ids, comp_ids, layers):
            return SimpleNamespace(
                features={"unexpected_feature": np.zeros(1536, dtype=np.float16)},
                cost={"latency_ms": 10.0, "gpu_seconds": 0.0},
            )

    fast = MissingKeyModel(A2_FAST_MODEL, "fast")
    slow = FakeModel(A2_SLOW_MODEL, "slow")
    router = RealRouter(probe=FakeProbe(0.5), fast_model=fast, slow_model=slow)

    with pytest.raises(KeyError, match="Missing probe feature"):
        router.route(make_request())


def test_non_finite_probe_score_raises_valueerror():
    """Verify NaN/Inf risk score is rejected."""
    class NanProbe(FakeProbe):
        def predict(self, x):
            return np.array([float("nan")], dtype=np.float32)

    fast = FakeModel(A2_FAST_MODEL, "fast")
    slow = FakeModel(A2_SLOW_MODEL, "slow")
    router = RealRouter(probe=NanProbe(0.5), fast_model=fast, slow_model=slow)

    with pytest.raises(ValueError, match="Probe returned a non-finite score"):
        router.route(make_request())


def test_mismatched_model_names_raise_valueerror():
    """Verify calibrated router rejects models differing from A2 configuration."""
    router, _, _ = make_router(0.01)

    req_fast_mismatch = make_request()
    req_fast_mismatch.settings.fast_model = "meta-llama/Llama-3-8B"
    with pytest.raises(ValueError, match="A2 probe is calibrated for"):
        router.route(req_fast_mismatch)

    req_slow_mismatch = make_request()
    req_slow_mismatch.settings.slow_model = "meta-llama/Llama-3-70B"
    with pytest.raises(ValueError, match="A2 configuration expects"):
        router.route(req_slow_mismatch)


def test_invalid_force_route_raises_valueerror():
    """Verify unsupported force_route raises ValueError."""
    router, _, _ = make_router(0.01)
    req = make_request(force_route="unsupported_route")
    with pytest.raises(ValueError, match="Unsupported force_route"):
        router.route(req)


# ---------------------------------------------------------
# F5 Regression: Downstream Persistence & Compatibility
# ---------------------------------------------------------

def test_save_result_compatibility_with_real_router():
    """Verify save_result correctly serializes RealRouter RouteResult to database models."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from backend.db_models import Base, PathTelemetry, Request
    from backend.persistence import save_result

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    # 1. Fast route result persistence
    router_fast, _, _ = make_router(0.01)
    req_fast = make_request()
    res_fast = router_fast.route(req_fast)

    with Session(engine) as s:
        db_fast = save_result(
            s,
            res_fast,
            query=req_fast.prompt,
            settings=req_fast.settings,
            source="playground",
        )
        assert db_fast.route == "fast"
        assert db_fast.probe_score == pytest.approx(0.01)
        assert db_fast.conflict_score == pytest.approx(0.01)
        assert db_fast.final_answer == "Fast answer"
        assert db_fast.total_latency_ms == pytest.approx(res_fast.total_latency_ms)
        assert len(db_fast.paths) == 2
        assert [p.path for p in db_fast.paths] == ["fast", "probe"]

    # 2. Escalated route result persistence
    router_esc, _, _ = make_router(0.99)
    req_esc = make_request()
    res_esc = router_esc.route(req_esc)

    with Session(engine) as s:
        db_esc = save_result(
            s,
            res_esc,
            query=req_esc.prompt,
            settings=req_esc.settings,
            source="eval",
        )
        assert db_esc.route == "escalated"
        assert db_esc.probe_score == pytest.approx(0.99)
        assert db_esc.final_answer == "Slow answer"
        assert db_esc.fast_answer == "Fast answer"
        assert db_esc.total_latency_ms == pytest.approx(res_esc.total_latency_ms)
        assert len(db_esc.paths) == 3
        assert [p.path for p in db_esc.paths] == ["fast", "probe", "slow"]


def test_placeholder_router_backward_compatibility():
    """Ensure PlaceholderRouter functions unchanged and preserves simulated behavior."""
    from engine.placeholder import PlaceholderRouter

    p = PlaceholderRouter()
    req = make_request()
    res = p.route(req)

    assert res.simulated is True
    assert res.backend == "placeholder"
    assert res.route in ("fast", "escalated")
    assert res.final_answer is not None
    assert len(res.paths) > 0
