import unittest
from unittest.mock import patch
import httpx
from nanmesh_memory.client import NaNMeshClient


class EvidenceContractTests(unittest.TestCase):
    def client(self):
        return NaNMeshClient(api_key="test-not-a-real-key", agent_id="test")

    def test_positive_votes_without_execution_remain_unknown(self):
        client = self.client()
        payload = {"slug": "tool", "evaluation_count": 10, "trust_score": 10,
                   "confidence_decomposition": {"status": "no_operational_reports", "evidence_state": "no_operational_reports", "contributor_count": 0}}
        with patch.object(client, "_get", side_effect=[payload, {"reviews": []}, {"problems": []}]) as get:
            result = client.check("tool")
        self.assertEqual(get.call_args_list[0].args[1], {"format": "agent"})
        self.assertEqual(result["vote_verdict"], "trusted")
        self.assertEqual(result["verdict"], "unknown")
        self.assertEqual(result["confidence_decomposition"], payload["confidence_decomposition"])

    def test_not_found_and_service_error_are_distinct(self):
        for status, code in [(404, "not_found"), (503, "service_error"), (429, "service_error")]:
            request = httpx.Request("GET", "https://example.invalid/entities/tool")
            error = httpx.HTTPStatusError("test", request=request, response=httpx.Response(status, request=request))
            client = self.client()
            with patch.object(client, "_get", side_effect=error):
                result = client.check("tool")
            self.assertEqual(result["error_code"], code)
            self.assertEqual(result["retryable"], status != 404)
            self.assertEqual(result["verdict"], "unknown")

    def test_transport_error_is_not_missing_coverage(self):
        client = self.client()
        with patch.object(client, "_get", side_effect=httpx.ReadTimeout("test")):
            result = client.check("tool")
        self.assertEqual(result["status"], "service_error")

    def test_complete_search_preserves_coverage_and_legacy_list(self):
        client = self.client()
        payload = {"entities": [], "coverage_help": {"status": "missing_coverage"}, "contribution_invite": {"type": "question"}}
        with patch.object(client, "_get", return_value=payload):
            self.assertEqual(client.search_details("unlisted"), payload)
            self.assertEqual(client.search("unlisted"), [])
        with patch.object(client, "_get", side_effect=httpx.ReadTimeout("test")):
            with self.assertRaises(httpx.ReadTimeout):
                client.search_details("unlisted")

    def test_missing_problem_service_prevents_positive_verdict(self):
        client = self.client()
        payload = {"confidence_decomposition": {"status": "computed", "evidence_state": "sufficient", "integration_success_rate": 1, "contributor_count": 5}}
        with patch.object(client, "_get", side_effect=[payload, {"reviews": []}, httpx.ReadTimeout("test")]):
            result = client.check("tool")
        self.assertEqual(result["verdict"], "unknown")
        self.assertEqual(result["status"], "partial_service_error")

    def test_execution_failure_is_warned_independent_of_votes(self):
        client = self.client()
        payload = {"confidence_decomposition": {"status": "insufficient_evidence", "integration_success_rate": 0, "contributor_count": 1}}
        with patch.object(client, "_get", side_effect=[payload, {"reviews": []}, {"problems": []}]):
            self.assertEqual(client.check("tool")["verdict"], "warned")

    def test_malformed_entity_and_votes_return_structured_service_errors(self):
        for payload in [{"entity": None}, {"entity": []}, {"trust_score": "positive"},
                        {"evaluation_count": "5"}, {"trust_score": float("nan")}]:
            with self.subTest(payload=payload):
                client = self.client()
                with patch.object(client, "_get", return_value=payload):
                    result = client.check("tool")
                self.assertEqual(result["status"], "service_error")
                self.assertEqual(result["verdict"], "unknown")

    def test_malformed_confidence_never_produces_positive_trust(self):
        sufficient = {"status": "computed", "evidence_state": "sufficient", "integration_success_rate": 1, "contributor_count": 5}
        cases = [None, [], "computed", {**sufficient, "contributor_count": "5"},
                 {**sufficient, "contributor_count": -1}, {**sufficient, "integration_success_rate": True},
                 {**sufficient, "integration_success_rate": float("nan")},
                 {**sufficient, "integration_success_rate": 2}, {**sufficient, "status": []}]
        for confidence in cases:
            with self.subTest(confidence=confidence):
                client = self.client()
                with patch.object(client, "_get", side_effect=[{"confidence_decomposition": confidence}, {"reviews": []}, {"problems": []}]):
                    result = client.check("tool")
                self.assertEqual(result["status"], "partial_service_error")
                self.assertEqual(result["verdict"], "unknown")
                self.assertEqual(result["confidence_decomposition"], {})

    def test_malformed_auxiliary_arrays_are_partial_service_errors(self):
        payload = {"confidence_decomposition": {"status": "computed", "evidence_state": "sufficient", "integration_success_rate": 1, "contributor_count": 5}}
        for key in ["reviews", "problems"]:
            for malformed in [None, {}, "none", [None], ["invalid"]]:
                with self.subTest(key=key, malformed=malformed):
                    reviews = {"reviews": malformed if key == "reviews" else []}
                    problems = {"problems": malformed if key == "problems" else []}
                    client = self.client()
                    with patch.object(client, "_get", side_effect=[payload, reviews, problems]):
                        result = client.check("tool")
                    self.assertEqual(result["status"], "partial_service_error")
                    self.assertEqual(result["verdict"], "unknown")

    def test_failure_warning_survives_unavailable_auxiliary_service(self):
        payload = {"confidence_decomposition": {"integration_success_rate": 0, "contributor_count": 1}}
        client = self.client()
        with patch.object(client, "_get", side_effect=[payload, {"reviews": None}, {"problems": []}]):
            result = client.check("tool")
        self.assertEqual(result["status"], "partial_service_error")
        self.assertEqual(result["verdict"], "warned")

    def test_sufficient_label_without_contributors_cannot_establish_trust(self):
        payload = {"confidence_decomposition": {"status": "computed", "evidence_state": "sufficient", "integration_success_rate": 1}}
        client = self.client()
        with patch.object(client, "_get", side_effect=[payload, {"reviews": []}, {"problems": []}]):
            self.assertEqual(client.check("tool")["verdict"], "unknown")


if __name__ == "__main__":
    unittest.main()
