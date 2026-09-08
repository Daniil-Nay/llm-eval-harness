"""evalkit: a small, dependency-light toolkit for evaluating LLM/RAG systems.

Design rules:
  * stdlib + requests only, no numpy/pandas - every formula is readable;
  * everything deterministic is tested offline; only judge calls need a key;
  * numbers ship with confidence intervals or they do not ship at all.
"""

__version__ = "0.1.0"
