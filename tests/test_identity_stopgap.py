import unittest
from unittest.mock import patch

from nanmesh_memory.client import AgentKeyRequiredError, NaNMeshClient


class IdentityStopgapTests(unittest.TestCase):
    def test_write_without_key_is_read_only_and_does_not_call_http(self):
        with patch("nanmesh_memory.client._load_saved_key", return_value=""), patch.dict(
            "os.environ", {}, clear=True
        ), patch("nanmesh_memory.client.httpx.Client") as http_client:
            client = NaNMeshClient(agent_id="deliberate-agent")

            with self.assertRaisesRegex(AgentKeyRequiredError, "No Agent was created"):
                client.vote("stripe", positive=True)

            http_client.assert_not_called()

    def test_explicit_existing_key_remains_accepted(self):
        client = NaNMeshClient(api_key="nmk_live_existing", agent_id="existing-agent")
        with patch.object(client, "_post", return_value={"success": True}) as post:
            result = client.vote("stripe", positive=True, context="worked")

        self.assertEqual(result, {"success": True})
        self.assertEqual(post.call_args.args[0], "/entities/stripe/vote")
        self.assertEqual(post.call_args.args[1]["agent_id"], "existing-agent")

    def test_explicit_registration_requires_a_deliberate_agent_id(self):
        with patch("nanmesh_memory.client._load_saved_agent_id", return_value=""), patch.dict(
            "os.environ", {}, clear=True
        ), patch("nanmesh_memory.client.httpx.Client") as http_client:
            client = NaNMeshClient()

            with self.assertRaisesRegex(ValueError, "stable agent_id"):
                client.register("My Agent", "description")

            http_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
