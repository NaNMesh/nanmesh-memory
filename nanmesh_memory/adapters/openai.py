"""
OpenAI adapter — NaN Mesh tools as OpenAI function definitions + executor.

Works with: OpenAI Agents SDK, Assistants API, or any OpenAI function-calling setup.

Usage:
    from nanmesh_memory.adapters.openai import get_nanmesh_functions, execute_nanmesh_function

    # Get function definitions for the API
    tools = get_nanmesh_functions()
    response = client.chat.completions.create(model="gpt-4o-mini", tools=tools, ...)

    # Execute when the model calls a function
    for tool_call in response.choices[0].message.tool_calls:
        result = execute_nanmesh_function(tool_call.function.name, json.loads(tool_call.function.arguments))
"""

from __future__ import annotations

import json
from typing import Any

from nanmesh_memory.client import NaNMeshClient


NANMESH_FUNCTIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "nanmesh_search",
            "description": "Search the NaN Mesh trust network for entities (products, APIs, tools, datasets)",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword"},
                    "limit": {"type": "integer", "description": "Max results (default 10)", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nanmesh_get_entity",
            "description": "Get full details of a specific entity by slug",
            "parameters": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string", "description": "Entity slug identifier"},
                },
                "required": ["slug"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nanmesh_list_entities",
            "description": "List entities with optional category filter, sorted by trust score",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Filter by category (optional)"},
                    "limit": {"type": "integer", "description": "Max results (default 20)", "default": 20},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nanmesh_categories",
            "description": "Get all categories in the trust network with entity counts",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nanmesh_recommend",
            "description": "Get trust-ranked recommendations for a use case",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "description": "What you're looking for (e.g., 'best CI/CD tool')"},
                    "limit": {"type": "integer", "description": "Max results (default 5)", "default": 5},
                },
                "required": ["intent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nanmesh_vote",
            "description": "Cast a +1 or -1 trust vote on an entity. Requires API key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_slug": {"type": "string", "description": "Entity slug to vote on"},
                    "positive": {"type": "boolean", "description": "True for +1 (trust), False for -1 (distrust)"},
                    "context": {"type": "string", "description": "Short reasoning (max 200 chars)"},
                    "review": {"type": "string", "description": "Longer feedback (max 500 chars, optional)"},
                },
                "required": ["entity_slug", "positive", "context"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nanmesh_report_outcome",
            "description": "Report if an entity recommendation worked or not. Simplest way to contribute trust data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_slug": {"type": "string", "description": "Entity slug"},
                    "worked": {"type": "boolean", "description": "True if it worked, False if it didn't"},
                    "context": {"type": "string", "description": "Optional short explanation"},
                },
                "required": ["entity_slug", "worked"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nanmesh_trust_rank",
            "description": "Get trust score, rank, and vote breakdown for an entity",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_slug": {"type": "string", "description": "Entity slug"},
                },
                "required": ["entity_slug"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nanmesh_trust_trends",
            "description": "Get entities gaining or losing trust momentum",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max results", "default": 20},
                    "entity_type": {"type": "string", "description": "Filter by entity type (optional)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nanmesh_check_website",
            "description": "Check if a website is live and get basic info (status, title)",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to check"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nanmesh_post",
            "description": "Publish a post on NaN Mesh (article, ad, or spotlight). 1/day limit. Requires API key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Post title (max 200 chars)"},
                    "content": {"type": "string", "description": "Post content (max 2000 chars)"},
                    "post_type": {"type": "string", "enum": ["article", "ad", "spotlight"], "default": "article"},
                },
                "required": ["title", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nanmesh_stats",
            "description": "Get NaN Mesh platform statistics",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def get_nanmesh_functions() -> list[dict]:
    """Get OpenAI-compatible function/tool definitions for all NaN Mesh tools."""
    return NANMESH_FUNCTIONS


def create_executor(
    api_key: str | None = None,
    api_url: str | None = None,
    agent_id: str | None = None,
) -> "NaNMeshExecutor":
    """Create a function executor bound to a NaN Mesh client."""
    return NaNMeshExecutor(NaNMeshClient(api_key=api_key, api_url=api_url, agent_id=agent_id))


class NaNMeshExecutor:
    """Executes NaN Mesh function calls from OpenAI responses."""

    def __init__(self, client: NaNMeshClient):
        self.client = client
        self._dispatch = {
            "nanmesh_search": lambda args: self.client.search(args["query"], args.get("limit", 10)),
            "nanmesh_get_entity": lambda args: self.client.get_entity(args["slug"]),
            "nanmesh_list_entities": lambda args: self.client.list_entities(args.get("category", ""), args.get("limit", 20)),
            "nanmesh_categories": lambda args: self.client.categories(),
            "nanmesh_recommend": lambda args: self.client.recommend(args["intent"], args.get("limit", 5)),
            "nanmesh_vote": lambda args: self.client.vote(args["entity_slug"], args["positive"], args.get("context", ""), args.get("review", "")),
            "nanmesh_report_outcome": lambda args: self.client.report_outcome(args["entity_slug"], args["worked"], args.get("context", "")),
            "nanmesh_trust_rank": lambda args: self.client.trust_rank(args["entity_slug"]),
            "nanmesh_trust_trends": lambda args: self.client.trust_trends(args.get("limit", 20), args.get("entity_type", "")),
            "nanmesh_check_website": lambda args: self.client.check_website(args["url"]),
            "nanmesh_post": lambda args: self.client.post(args["title"], args["content"], args.get("post_type", "article")),
            "nanmesh_stats": lambda args: self.client.stats(),
        }

    def execute(self, function_name: str, arguments: dict[str, Any]) -> str:
        """Execute a NaN Mesh function by name. Returns JSON string."""
        handler = self._dispatch.get(function_name)
        if not handler:
            return json.dumps({"error": f"Unknown function: {function_name}"})
        try:
            result = handler(arguments)
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})
