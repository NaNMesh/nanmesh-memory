"""
CrewAI adapter — drop-in NaN Mesh tools for any CrewAI agent.

Usage:
    from nanmesh_memory.adapters.crewai import get_nanmesh_tools

    tools = get_nanmesh_tools(api_key="nmk_live_...")

    agent = Agent(
        role="Trust Evaluator",
        goal="Run tool-risk preflight before recommending tools",
        tools=tools,
    )
"""

from __future__ import annotations

from typing import Any

from nanmesh_memory.client import NaNMeshClient

try:
    from crewai.tools import BaseTool
except ImportError:
    raise ImportError(
        "crewai is required for this adapter. Install with: pip install nanmesh-memory[crewai]"
    )


def _make_tool(tool_name: str, tool_description: str, func):
    """Create a CrewAI BaseTool from a function."""

    _name = tool_name
    _desc = tool_description
    _func = func

    class DynamicTool(BaseTool):
        name: str = _name
        description: str = _desc

        def _run(self, **kwargs) -> str:
            try:
                result = _func(**kwargs)
                if isinstance(result, (dict, list)):
                    import json
                    return json.dumps(result, indent=2, default=str)
                return str(result)
            except Exception as e:
                return f"Error: {e}"

    return DynamicTool()


def get_nanmesh_tools(
    api_key: str | None = None,
    api_url: str | None = None,
    agent_id: str | None = None,
) -> list[BaseTool]:
    """Get all NaN Mesh tools ready for CrewAI agents.

    Returns a list of BaseTool instances that can be passed to any CrewAI Agent.
    """
    client = NaNMeshClient(api_key=api_key, api_url=api_url, agent_id=agent_id)

    tools = [
        _make_tool(
            "nanmesh_search",
            "Search NaN Mesh for entities before recommending a tool. Args: query (str), limit (int, default 10)",
            lambda query, limit=10: client.search(query, int(limit)),
        ),
        _make_tool(
            "nanmesh_get_entity",
            "Read an evidence-aware entity payload by slug. Args: slug (str)",
            lambda slug: client.get_entity(slug, format="agent"),
        ),
        _make_tool(
            "nanmesh_get_entity_problems",
            "Check known problem threads for an entity before deciding. Args: slug (str), limit (int, default 20)",
            lambda slug, limit=20: client.get_entity_problems(slug, limit=int(limit)),
        ),
        _make_tool(
            "nanmesh_list_entities",
            "List entities with optional category filter. Args: category (str, optional), limit (int, default 20)",
            lambda category="", limit=20: client.list_entities(category, int(limit)),
        ),
        _make_tool(
            "nanmesh_categories",
            "Get all categories in the trust network with entity counts. No args.",
            lambda: client.categories(),
        ),
        _make_tool(
            "nanmesh_recommend",
            "Get trust-ranked recommendations for a use case. Args: intent (str), limit (int, default 5)",
            lambda intent, limit=5: client.recommend(intent, int(limit)),
        ),
        _make_tool(
            "nanmesh_vote",
            "Cast an optional, explicitly authorized +1 or -1 trust vote after real evaluation. Requires API key. Args: entity_slug (str), positive (bool), context (str, max 200 chars), review (str, max 500 chars, optional)",
            lambda entity_slug, positive, context="", review="": client.vote(
                entity_slug, positive if isinstance(positive, bool) else str(positive).lower() == "true", context, review
            ),
        ),
        _make_tool(
            "nanmesh_report_outcome",
            "Optionally publish an observed execution outcome only with explicit user authorization, including task/environment, errors, and limitations; omit private context. Args: entity_slug (str), worked (bool), context (str, optional)",
            lambda entity_slug, worked, context="": client.report_outcome(
                entity_slug, worked if isinstance(worked, bool) else str(worked).lower() == "true", context
            ),
        ),
        _make_tool(
            "nanmesh_trust_rank",
            "Get trust score, rank, and vote breakdown for an entity. Args: entity_slug (str)",
            lambda entity_slug: client.trust_rank(entity_slug),
        ),
        _make_tool(
            "nanmesh_trust_trends",
            "Get entities gaining or losing trust momentum. Args: limit (int, default 20), entity_type (str, optional)",
            lambda limit=20, entity_type="": client.trust_trends(int(limit), entity_type),
        ),
        _make_tool(
            "nanmesh_check_website",
            "Check if a website is live and get basic info (status, title). Args: url (str)",
            lambda url: client.check_website(url),
        ),
        _make_tool(
            "nanmesh_post",
            "Optionally publish an explicitly authorized post from actual tested outcomes. Research-only questions require approval of the exact draft; missing coverage calls for a local trial, not automatic publication. 1/hour limit. Args: title (str), content (str), post_type (str, default 'article')",
            lambda title, content, post_type="article": client.post(title, content, post_type),
        ),
        _make_tool(
            "nanmesh_stats",
            "Get NaN Mesh platform statistics (total entities, agents, votes). No args.",
            lambda: client.stats(),
        ),
    ]

    return tools
