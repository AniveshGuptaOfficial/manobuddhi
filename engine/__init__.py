"""ManoBuddhi router engine (built in F4-F6, needs a GPU).

Planned modules:
  serving.py     - fast/slow model clients (vLLM) + hidden-state extraction (Transformers)
  probe.py       - linear probe on intermediate hidden states (O1)
  conflict.py    - conflict score = probe divergence x K-sample disagreement (O1)
  controller.py  - learn-then-test certified set + load-adaptive threshold (O3)
  reuse.py       - prompt KV prefix reuse (Config A) / speculative draft (Config B) (O4)
  telemetry.py   - NVML GPU-seconds, TTFT, p95 latency (O2)

The backend talks to the engine only through one router interface, so the
placeholder in backend/router.py can be swapped out without touching the API.
"""
