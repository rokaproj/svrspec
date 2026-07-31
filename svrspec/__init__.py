"""svrspec — CPU-only LLM server spec sizing simulator.

Feed it a model and it tells you what server to buy: which CPU candidates meet
the alarm-to-Teams latency target, how much RAM to populate, and how much
headroom is left. Stdlib only, so it runs on the air-gapped CPU+RAM box it is
sizing for.
"""

__version__ = "0.2.1"

__all__ = ["__version__"]
