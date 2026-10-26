"""ManoBuddhi router engine.

Built so far (no GPU needed to import unless noted):
  interface.py   - the one Router contract between API and engine
  placeholder.py - simulated router (until F5)
  threshold.py, prompts.py, grading.py, stats.py, data/ (F1-F2)
  telemetry.py   - NVML GPU-seconds, energy, utilisation (F4)
  serving.py     - real models via Transformers, TTFT, log-probs, hidden states (F4, needs torch)
  collect.py     - paired fast/slow runs on a benchmark split, resumable (F4, needs torch)
  probe.py       - linear probe on hidden states: P(fast answer wrong) (F4, O1)
  baselines.py   - always-slow, log-prob cascade, probe-only from paired runs (F4)
  results.py, pipeline.py - results folder and the CPU steps after collection (F4)

Still to come:
  conflict.py    - conflict score = probe divergence x K-sample disagreement (F5, O1)
  controller.py  - learn-then-test certified set + load-adaptive threshold (F6, O3)
  reuse.py       - prompt KV prefix reuse (Config A) / speculative draft (Config B) (F5, O4)

The backend talks to the engine only through one router interface, so the
placeholder can be swapped out without touching the API.
"""
