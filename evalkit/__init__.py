"""evalkit: a small, dependency-light toolkit for evaluating LLM/RAG systems.

Design rules:
  * stdlib + requests only, no numpy/pandas, so every formula stays readable;
  * everything deterministic is tested offline; only judge calls need a key;
  * every reported proportion carries a confidence interval.
"""

__version__ = "0.1.0"
