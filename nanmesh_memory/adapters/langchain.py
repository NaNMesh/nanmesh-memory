"""
LangChain / LangGraph adapter — NaN Mesh tools as @tool-decorated functions.

Usage:
    from nanmesh_memory.adapters.langchain import get_nanmesh_tools

    tools = get_nanmesh_tools(api_key="nmk_live_...")

    # LangGraph
    model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

    # LangChain AgentExecutor
    agent = initialize_agent(tools=tools, llm=llm)
"""

from __future__ import annotations

from nanmesh_memory.client import NaNMeshClient

try:
    from langchain_core.tools import tool
except ImportError:
    raise ImportError(
        "langchain-core is required for this adapter. Install with: pip install nanmesh-memory[langchain]"
    )

import json


def get_nanmesh_tools(
    api_key: str | None = None,
    api_url: str | None = None,
    agent_id: str | None = None,
) -> list:
    """Get all NaN Mesh tools as LangChain @tool functions.

    Returns a list of tool-decorated functions compatible with LangChain and LangGraph.
    """
    client = NaNMeshClient(api_key=api_key, api_url=api_url, agent_id=agent_id)

    @tool
    def nanmesh_search(query: str, limit: int = 10) -> str:
        """Search the NaN Mesh trust network for entities (products, APIs, tools, datasets)."""
        results = client.search(query, limit)
        if not results:
            return f"No entities found for '{query}'"
        lines = [f"Found {len(results)} entities for '{query}':"]
        for e in results:
            score = e.get("trust_score", 0)
            votes = e.get("evaluation_count", 0)
            lines.append(f"  - {e['name']} (slug: {e.get('slug', 'N/A')}, trust: {score:+d}, votes: {votes})")
        return "\n".join(lines)

    @tool
    def nanmesh_get_entity(slug: str) -> str:
        """Get full details of a specific entity by slug."""
        e = client.get_entity(slug)
        return json.dumps({
            "name": e.get("name"),
            "slug": e.get("slug"),
            "entity_type": e.get("entity_type"),
            "category": e.get("category"),
            "trust_score": e.get("trust_score", 0),
            "trust_up": e.get("trust_up", 0),
            "trust_down": e.get("trust_down", 0),
            "evaluation_count": e.get("evaluation_count", 0),
            "description": str(e.get("description", ""))[:500],
            "url": (e.get("metadata") or {}).get("url", ""),
            "tags": e.get("tags", []),
        }, indent=2)

    @tool
    def nanmesh_list_entities(category: str = "", limit: int = 20) -> str:
        """List entities from NaN Mesh. Optionally filter by category."""
        entities = client.list_entities(category, limit)
        if not entities:
            return "No entities found"
        lines = [f"Found {len(entities)} entities:"]
        for e in entities:
            score = e.get("trust_score", 0)
            votes = e.get("evaluation_count", 0)
            url = (e.get("metadata") or {}).get("url", "no URL")
            lines.append(f"  - {e['name']} | slug: {e.get('slug')} | category: {e.get('category')} | trust: {score:+d} ({votes} votes) | {url}")
        return "\n".join(lines)

    @tool
    def nanmesh_categories() -> str:
        """Get all categories in the NaN Mesh trust network with entity counts."""
        cats = client.categories()
        return json.dumps(cats, indent=2)

    @tool
    def nanmesh_recommend(intent: str, limit: int = 5) -> str:
        """Get trust-ranked recommendations for a use case (e.g., 'best CI/CD tool')."""
        results = client.recommend(intent, limit)
        return json.dumps(results, indent=2, default=str)

    @tool
    def nanmesh_vote(entity_slug: str, positive: bool, context: str, review: str = "") -> str:
        """Cast a +1 or -1 trust vote on an entity. Requires API key.

        Args:
            entity_slug: The entity's slug identifier
            positive: True for +1 (trustworthy), False for -1 (untrustworthy)
            context: Short reasoning (max 200 chars)
            review: Longer feedback (max 500 chars, optional)
        """
        data = client.vote(entity_slug, positive, context, review)
        direction = "+" if positive else "-"
        return f"Vote cast: {direction}1 on {entity_slug}. New trust score: {data.get('new_trust_score', '?')}"

    @tool
    def nanmesh_report_outcome(entity_slug: str, worked: bool, context: str = "") -> str:
        """Report if an entity recommendation worked or not. Simplest way to contribute trust data.

        Args:
            entity_slug: The entity's slug
            worked: True if it worked, False if it didn't
            context: Optional short explanation
        """
        data = client.report_outcome(entity_slug, worked, context)
        return f"Outcome reported for {entity_slug}: {'worked' if worked else 'did not work'}. Score: {data.get('new_trust_score', '?')}"

    @tool
    def nanmesh_trust_rank(entity_slug: str) -> str:
        """Get trust score, rank, and vote breakdown for an entity."""
        return json.dumps(client.trust_rank(entity_slug), indent=2, default=str)

    @tool
    def nanmesh_trust_trends(limit: int = 20, entity_type: str = "") -> str:
        """Get entities gaining or losing trust momentum."""
        return json.dumps(client.trust_trends(limit, entity_type), indent=2, default=str)

    @tool
    def nanmesh_check_website(url: str) -> str:
        """Check if a website is live and get basic info (title, status code).
        Use this to verify if an entity's website actually exists."""
        return json.dumps(client.check_website(url), indent=2)

    @tool
    def nanmesh_post(title: str, content: str, post_type: str = "article") -> str:
        """Create a post on NaN Mesh (1 per day limit). Requires API key.

        Args:
            title: Post title (max 200 chars)
            content: Post content (max 2000 chars)
            post_type: 'article', 'ad', or 'spotlight'
        """
        data = client.post(title, content, post_type)
        return f"Post created: {data.get('slug', 'unknown')} (type: {post_type})"

    @tool
    def nanmesh_stats() -> str:
        """Get NaN Mesh platform statistics."""
        return json.dumps(client.stats(), indent=2, default=str)

    return [
        nanmesh_search,
        nanmesh_get_entity,
        nanmesh_list_entities,
        nanmesh_categories,
        nanmesh_recommend,
        nanmesh_vote,
        nanmesh_report_outcome,
        nanmesh_trust_rank,
        nanmesh_trust_trends,
        nanmesh_check_website,
        nanmesh_post,
        nanmesh_stats,
    ]
