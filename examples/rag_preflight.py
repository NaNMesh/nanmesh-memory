"""Read-only candidate checks; this does not execute or benchmark a vector store."""
import argparse
import json
from datetime import datetime, timezone
from nanmesh_memory import NaNMeshClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slugs", nargs="*", default=["pgvector", "qdrant", "pinecone", "weaviate", "chroma"])
    parser.add_argument("--task", default="vector_memory")
    args = parser.parse_args()
    client = NaNMeshClient()
    for slug in args.slugs:
        result = client.check(slug, task_type=args.task)
        print(json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(),
            "slug": slug, "task": args.task, "kind": "read_only_preflight",
            "result": result, "next_step": "Compare official docs and run a small local trial matching your workload; do not treat missing evidence as failure.",
            "limitations": "No candidate was installed or executed; this is not an execution report."}, default=str))


if __name__ == "__main__":
    main()
