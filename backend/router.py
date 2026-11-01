"""Active router instance (one per backend name)."""
from functools import lru_cache

from engine.placeholder import PlaceholderRouter


@lru_cache(maxsize=4)
def active_router(backend: str):
    if backend == "placeholder":
        return PlaceholderRouter()

    if backend == "real":
        from engine.real_router import RealRouter
        return RealRouter()

    raise ValueError(f"Unknown router backend: {backend}")