"""XAI-Gov: agent-governed digital twin for supply chains.

The package is organized in the seven layers described in the project
protocol; every layer exposes a contract verified by tests.

    core           determinism, configuration, paths, logging
    io             persistence, hash-chained decision log, run writers
    simulation     the digital twin (discrete events + agents)
    policies       decision policies proposing actions
    causal         structural causal model and causal recourse
    xai            post hoc explainers, subordinate to `causal`
    conformal      adaptive conformal inference and change detection
    governance     the constrained POMDP governance agent
    verification   PCTL specifications, shield synthesis, runtime monitors
    oversight      escalation as learned delegation
    calibration    simulation-based inference against public data
    kpis           the six indicator layers
    orchestration  experiment composition and campaign running
"""

from __future__ import annotations

__version__ = "0.2.0"
__all__ = ["__version__"]
