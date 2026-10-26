"""Active router instance (one per backend name)."""
from functools import lru_cache

from engine.placeholder import get_router


@lru_cache(maxsize=4)
def active_router(backend: str):
    return get_router(backend)
