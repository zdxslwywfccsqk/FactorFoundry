"""Agentic research runtime for FactorFoundry.

The package intentionally keeps numerical evaluation in deterministic tools and
uses language-model agents only for proposal, implementation selection, critique,
and research narration. This separation is what makes a run reproducible.
"""

from .orchestrator import ResearchOrchestrator
from .schemas import ArtifactRecord, HypothesisCard, RunManifest
from .session import ResearchSession

__all__ = [
    "ArtifactRecord",
    "HypothesisCard",
    "ResearchOrchestrator",
    "ResearchSession",
    "RunManifest",
]
