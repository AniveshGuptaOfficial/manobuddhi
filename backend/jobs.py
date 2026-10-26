"""Benchmark jobs. Run by the RQ worker (Docker) or inline in the API process."""
import traceback
from datetime import datetime, timezone

from engine.data import load_split
from engine.interface import RouteRequest, RouterSettings
from engine.placeholder import get_router
from engine.prompts import build_prompt

from . import metrics
from .db import SessionLocal
from .db_models import EvalRun
from .persistence import save_result

FORCE = {"manobuddhi": None, "always_fast": "fast", "always_slow": "escalated"}
COMMIT_EVERY = 25


def run_eval(run_id: int) -> dict:
    with SessionLocal() as s:
        run = s.get(EvalRun, run_id)
        if run is None:
            raise ValueError(f"eval run {run_id} not found")
        run.status, run.progress = "running", 0.0
        s.commit()
        try:
            items = load_split(run.benchmark, run.split)[: run.n_items]
            settings = RouterSettings.from_dict(run.router_config)
            router = get_router(settings.backend)
            force = FORCE[run.baseline or "manobuddhi"]
            for i, item in enumerate(items, 1):
                prompt = build_prompt(item)
                res = router.route(RouteRequest(prompt, settings, item=item, force_route=force))
                save_result(s, res, query=prompt, settings=settings, source="eval",
                            eval_run_id=run.id, item=item, commit=False)
                if i % COMMIT_EVERY == 0 or i == len(items):
                    run.progress = i / len(items)
                    s.commit()
            summary = metrics.compute(s, eval_run_id=run.id)
            summary["router_backend"] = settings.backend
            summary["simulated"] = settings.backend == "placeholder"
            run.summary, run.status = summary, "done"
        except Exception as e:  # noqa: BLE001
            s.rollback()
            run = s.get(EvalRun, run_id)
            run.status = "failed"
            run.summary = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
        run.finished_at = datetime.now(timezone.utc)
        s.commit()
        return run.summary
