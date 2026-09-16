"""
nanmesh-memory -- A trust check for AI-recommended tools.

Check trust scores, known problems, and reviews before your agent recommends.
Reads need no key. Writes require an existing key or explicit registration.

Quick start:
    from nanmesh_memory import check

    result = check("stripe")
    print(result["verdict"])      # "trusted", "contested", "warned", or "unknown"
    print(result["trust_score"])  # 7
    print(result["vote_count"])   # 12

Full client:
    from nanmesh_memory import NaNMeshClient

    client = NaNMeshClient()       # no key needed for reads
    client.check("stripe")         # read — works immediately
    # Writes require api_key=... or NANMESH_AGENT_KEY.

Framework adapters:
    from nanmesh_memory.adapters.crewai import get_nanmesh_tools      # CrewAI
    from nanmesh_memory.adapters.langchain import get_nanmesh_tools    # LangChain/LangGraph
    from nanmesh_memory.adapters.openai import get_nanmesh_functions   # OpenAI function calling
"""

from nanmesh_memory.client import AgentKeyRequiredError, NaNMeshClient

__version__ = "0.6.0"


def check(slug: str, api_url: str | None = None) -> dict:
    """One-call trust check. No API key needed.

    Args:
        slug: Entity slug (e.g. "stripe", "vercel", "cursor")
        api_url: Optional API base URL (default: https://api.nanmesh.ai)

    Returns:
        dict with: entity, trust_score, vote_count, recent_reviews, problems, verdict
    """
    client = NaNMeshClient(api_url=api_url)
    return client.check(slug)


__all__ = ["AgentKeyRequiredError", "NaNMeshClient", "check"]
