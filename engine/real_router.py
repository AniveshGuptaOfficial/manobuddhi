
"""
F5: Real probe-based inference router.

Initial implementation:
    Fast: Qwen2.5-1.5B-Instruct
    Slow: Qwen2.5-7B-Instruct
    Probe: A2 LinearProbe (mean_L20)
    Policy: calibrated probe threshold

Models are loaded lazily, only when real inference is requested.
"""

from pathlib import Path

import numpy as np

from .interface import PathResult, RouteRequest, RouteResult
from .probe import LinearProbe


# ---------------------------------------------------------
# A2 calibration configuration
# ---------------------------------------------------------

A2_FAST_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
A2_SLOW_MODEL = "Qwen/Qwen2.5-7B-Instruct"

A2_PROBE_FEATURE = "mean_L20"
A2_PROBE_LAYER = 20

A2_THRESHOLD = 0.07466565817594528

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBE_PATH = PROJECT_ROOT / "results" / "a2" / "probe" / "probe.npz"


class RealRouter:
    """
    Real inference router using the calibrated A2 probe.

    Optional injected models and probe allow CPU-only unit tests.
    """

    name = "real"

    def __init__(
        self,
        *,
        probe_path=DEFAULT_PROBE_PATH,
        threshold=A2_THRESHOLD,
        probe=None,
        fast_model=None,
        slow_model=None,
        model_factory=None,
    ):
        self.threshold = float(threshold)

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Probe threshold must be between 0 and 1.")

        self.probe_path = Path(probe_path)

        # Loading a small NumPy probe does not load a language model.
        self.probe = probe

        if self.probe is None and self.probe_path.exists():
            self.probe = LinearProbe.load(self.probe_path)

        if self.probe is not None:
            if self.probe.feature != A2_PROBE_FEATURE:
                raise ValueError(
                    f"Expected {A2_PROBE_FEATURE}, "
                    f"got {self.probe.feature}"
                )

        # Injected models are useful for CPU tests.
        self.fast_model = fast_model
        self.slow_model = slow_model

        # Optional factory for model construction.
        self.model_factory = model_factory

    # -----------------------------------------------------
    # Lazy model loading
    # -----------------------------------------------------

    def _create_model(self, name, *, slow=False):
        """
        Construct a Hugging Face model only when inference needs it.
        """

        if self.model_factory is not None:
            return self.model_factory(name, slow=slow)

        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ModuleNotFoundError:
            has_cuda = False

        if not has_cuda:
            raise RuntimeError(
                "Real model inference requires a CUDA GPU. "
                "This computer currently has CPU-only PyTorch or no PyTorch. "
                "Use injected mock models for CPU tests."
            )

        from .serving import HFModel

        if slow:
            return HFModel(
                name,
                device_map="auto",
                dtype=torch.float16,
            )

        return HFModel(
            name,
            device="cuda:0",
            dtype=torch.float16,
        )

    def _get_fast_model(self, settings):
        if self.fast_model is None:
            self.fast_model = self._create_model(
                settings.fast_model,
                slow=False,
            )

        return self.fast_model

    def _get_slow_model(self, settings):
        if self.slow_model is None:
            self.slow_model = self._create_model(
                settings.slow_model,
                slow=True,
            )

        return self.slow_model

    # -----------------------------------------------------
    # Result conversion
    # -----------------------------------------------------

    @staticmethod
    def _number(value, default=0.0):
        if value is None:
            return default
        return float(value)

    @classmethod
    def _generation_path(cls, generation, path):
        return PathResult(
            path=path,
            model=generation.model,
            text=generation.text,
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            ttft_ms=cls._number(generation.ttft_ms),
            latency_ms=cls._number(generation.latency_ms),
            gpu_seconds=cls._number(generation.gpu_seconds),
            gpu_util_pct=generation.gpu_util_pct,
            gpu_mem_mb=generation.gpu_mem_mb,
            energy_j=generation.energy_j,
        )

    @classmethod
    def _probe_path(cls, model_name, cost):
        return PathResult(
            path="probe",
            model=model_name,
            text="",
            latency_ms=cls._number(cost.get("latency_ms")),
            gpu_seconds=cls._number(cost.get("gpu_seconds")),
            gpu_util_pct=cost.get("gpu_util_pct"),
            energy_j=cost.get("energy_j"),
        )

    # -----------------------------------------------------
    # Generation helpers
    # -----------------------------------------------------

    @staticmethod
    def _generate(model, prompt, max_new_tokens):
        results = model.generate(
            [prompt],
            max_new_tokens=max_new_tokens,
            timed=True,
        )

        if len(results) != 1:
            raise RuntimeError(
                "Expected exactly one generation result."
            )

        return results[0]

    def _fast_generation(self, req):
        model = self._get_fast_model(req.settings)

        generation, prompt_ids, completion_ids = self._generate(
            model,
            req.prompt,
            512,
        )

        return model, generation, prompt_ids, completion_ids

    def _slow_generation(self, req):
        model = self._get_slow_model(req.settings)

        generation, _, _ = self._generate(
            model,
            req.prompt,
            768,
        )

        return generation

    # -----------------------------------------------------
    # Probe scoring
    # -----------------------------------------------------

    def _score_fast_answer(
        self,
        model,
        prompt_ids,
        completion_ids,
    ):
        if self.probe is None:
            raise FileNotFoundError(
                f"A2 probe not found: {self.probe_path}"
            )

        extracted = model.probe_features(
            prompt_ids,
            completion_ids,
            layers=[A2_PROBE_LAYER],
        )

        if A2_PROBE_FEATURE not in extracted.features:
            raise KeyError(
                f"Missing probe feature: {A2_PROBE_FEATURE}"
            )

        feature = np.asarray(
            extracted.features[A2_PROBE_FEATURE],
            dtype=np.float32,
        ).reshape(1, -1)

        expected_dim = len(self.probe.coef)

        if feature.shape[1] != expected_dim:
            raise ValueError(
                f"Probe dimension mismatch: expected {expected_dim}, "
                f"received {feature.shape[1]}"
            )

        risk = float(self.probe.predict(feature)[0])

        if not np.isfinite(risk):
            raise ValueError("Probe returned a non-finite score.")

        return risk, extracted.cost

    # -----------------------------------------------------
    # Main routing method
    # -----------------------------------------------------

    def route(self, req: RouteRequest) -> RouteResult:
        settings = req.settings

        if settings.disagreement_enabled:
            raise NotImplementedError(
                "F5 currently supports probe-only routing. "
                "Set disagreement_enabled=False until disagreement "
                "routing is implemented."
            )

        if settings.fast_model != A2_FAST_MODEL:
            raise ValueError(
                "The A2 probe is calibrated for "
                f"{A2_FAST_MODEL}, not {settings.fast_model}."
            )

        if settings.slow_model != A2_SLOW_MODEL:
            raise ValueError(
                "The A2 configuration expects "
                f"{A2_SLOW_MODEL}, not {settings.slow_model}."
            )

        if req.force_route not in (None, "fast", "escalated"):
            raise ValueError(
                f"Unsupported force_route: {req.force_route}"
            )

        # ---------------------------------------------
        # Always-slow baseline: do not run the fast model.
        # ---------------------------------------------

        if req.force_route == "escalated":
            slow = self._slow_generation(req)

            slow_path = self._generation_path(slow, "slow")

            return RouteResult(
                route="escalated",
                final_answer=slow.text,
                fast_answer="",
                slow_answer=slow.text,
                conflict_score=0.0,
                probe_score=None,
                disagreement=None,
                logprob_confidence=None,
                threshold=self.threshold,
                threshold_mode="calibrated_probe",
                reuse_method=None,
                backend="hf-transformers",
                simulated=False,
                paths=[slow_path],
            )

        # ---------------------------------------------
        # Fast generation.
        # ---------------------------------------------

        model, fast, prompt_ids, completion_ids = (
            self._fast_generation(req)
        )

        fast_path = self._generation_path(fast, "fast")

        # ---------------------------------------------
        # Always-fast baseline: no probe or escalation.
        # ---------------------------------------------

        if req.force_route == "fast":
            return RouteResult(
                route="fast",
                final_answer=fast.text,
                fast_answer=fast.text,
                slow_answer=None,
                conflict_score=0.0,
                probe_score=None,
                disagreement=None,
                logprob_confidence=fast.conf_seq,
                threshold=self.threshold,
                threshold_mode="calibrated_probe",
                reuse_method=None,
                backend="hf-transformers",
                simulated=False,
                paths=[fast_path],
            )

        # ---------------------------------------------
        # Probe disabled: fast-only fallback.
        # ---------------------------------------------

        if not settings.probe_enabled:
            return RouteResult(
                route="fast",
                final_answer=fast.text,
                fast_answer=fast.text,
                slow_answer=None,
                conflict_score=0.0,
                probe_score=None,
                disagreement=None,
                logprob_confidence=fast.conf_seq,
                threshold=self.threshold,
                threshold_mode="probe_disabled",
                reuse_method=None,
                backend="hf-transformers",
                simulated=False,
                paths=[fast_path],
            )

        # ---------------------------------------------
        # Calculate probability of fast-answer error.
        # ---------------------------------------------

        risk, probe_cost = self._score_fast_answer(
            model,
            prompt_ids,
            completion_ids,
        )

        probe_path = self._probe_path(
            settings.fast_model,
            probe_cost,
        )

        paths = [fast_path, probe_path]

        # ---------------------------------------------
        # Escalation decision.
        # ---------------------------------------------

        if risk >= self.threshold:
            slow = self._slow_generation(req)
            slow_path = self._generation_path(slow, "slow")

            paths.append(slow_path)

            return RouteResult(
                route="escalated",
                final_answer=slow.text,
                fast_answer=fast.text,
                slow_answer=slow.text,
                conflict_score=risk,
                probe_score=risk,
                disagreement=None,
                logprob_confidence=fast.conf_seq,
                threshold=self.threshold,
                threshold_mode="calibrated_probe",
                reuse_method=None,
                backend="hf-transformers",
                simulated=False,
                paths=paths,
            )

        # ---------------------------------------------
        # Accept fast answer.
        # ---------------------------------------------

        return RouteResult(
            route="fast",
            final_answer=fast.text,
            fast_answer=fast.text,
            slow_answer=None,
            conflict_score=risk,
            probe_score=risk,
            disagreement=None,
            logprob_confidence=fast.conf_seq,
            threshold=self.threshold,
            threshold_mode="calibrated_probe",
            reuse_method=None,
            backend="hf-transformers",
            simulated=False,
            paths=paths,
        )
