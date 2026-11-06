import http.client
import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from brainroute.server import BrainRouteHandler


class ServerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), BrainRouteHandler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5)
        cls.server.server_close()

    def get(self, path, headers=None):
        request = urllib.request.Request(f"{self.base_url}{path}", headers=headers or {})
        with urllib.request.urlopen(request) as response:
            return response, json.loads(response.read().decode("utf-8"))

    def post(self, path, payload, headers=None):
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            return response, json.loads(response.read().decode("utf-8"))


class ReadEndpointTests(ServerTestCase):
    def test_config_endpoint(self):
        _, data = self.get("/api/config")
        self.assertIn("balanced", data["profiles"])
        self.assertGreaterEqual(len(data["models"]), 3)

    def test_healthz_endpoint(self):
        _, data = self.get("/healthz")
        self.assertTrue(data["ok"])

    def test_dashboard_endpoint(self):
        _, data = self.get("/api/dashboard")
        self.assertIn("models", data)

    def test_reliability_endpoint(self):
        _, data = self.get("/api/reliability")
        self.assertIn("models", data)

    def test_v1_models_endpoint_advertises_auto_routing(self):
        _, data = self.get("/v1/models")
        self.assertEqual(data["object"], "list")
        self.assertIn("brainroute-auto", [item["id"] for item in data["data"]])

    def test_evals_endpoint(self):
        # Reference routing expectations use seed scores; earlier HTTP tests
        # deliberately record latency telemetry that differs across runners.
        with patch("brainroute.scorer.snapshot", return_value={}):
            _, data = self.get("/api/evals")
        self.assertTrue(data["results"])
        self.assertEqual(data["passed"], data["total"], [r for r in data["results"] if not r["passed"]])

    def test_responses_carry_a_request_id(self):
        response, _ = self.get("/healthz")
        self.assertTrue(response.headers.get("X-Request-Id"))

    def test_unknown_api_post_returns_json_not_found(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post("/api/does-not-exist", {})
        self.assertEqual(caught.exception.code, 404)


class RoutingEndpointTests(ServerTestCase):
    def test_ask_dry_run_endpoint(self):
        _, data = self.post("/api/ask", {
            "prompt": "Summarize this confidential patient note",
            "profile": "private",
            "execute": False,
        })
        self.assertFalse(data["executed"])
        self.assertEqual(data["decision"]["recommended_model"], "local-qwen2-5")

    def test_route_endpoint_returns_an_explanation(self):
        _, data = self.post("/api/route", {"prompt": "Refactor this python module"})
        self.assertIn("explain", data)
        self.assertIn("weights", data)
        self.assertTrue(data["ranked"])

    def test_messages_are_accepted_in_place_of_a_prompt(self):
        _, data = self.post("/api/route", {
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "Summarize this confidential patient note"},
            ],
            "profile": "private",
        })
        self.assertEqual(data["privacy_level"], "high")

    def test_malformed_json_body_does_not_crash_the_handler(self):
        request = urllib.request.Request(
            f"{self.base_url}/api/route",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 200)

    def test_oversized_bodies_are_rejected(self):
        # The declared length is checked before the body is read, so the test
        # announces a huge body without sending one.
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_address[1], timeout=5
        )
        try:
            connection.putrequest("POST", "/api/route")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(8 * 1024 * 1024))
            connection.endheaders()
            self.assertEqual(connection.getresponse().status, 413)
        finally:
            connection.close()


class PostureEndpointTests(ServerTestCase):
    def test_postures_are_listed_with_the_active_one(self):
        _, data = self.get("/api/postures")
        self.assertEqual(data["active"], "local_first")
        self.assertIn("air_gapped", [item["id"] for item in data["postures"]])

    def test_config_advertises_postures_and_pinned_flags(self):
        _, data = self.get("/api/config")
        self.assertTrue(data["postures"])
        self.assertIn("pinned_policy", data)

    def test_an_unknown_posture_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post("/api/postures", {"posture": "nonsense"})
        self.assertEqual(caught.exception.code, 400)

    def test_routing_reports_destination_and_posture(self):
        _, data = self.post("/api/route", {"prompt": "Rename these three variables."})
        self.assertIn("local", data)
        self.assertEqual(data["policy"]["posture"], "local_first")
        self.assertIn("warnings", data)

    def test_a_hard_task_reports_the_privacy_escalation(self):
        _, data = self.post("/api/route", {
            "prompt": "Design a complex production architecture for a routing layer",
            "profile": "balanced",
        })
        self.assertFalse(data["local"])
        self.assertIn("privacy_escalation", [item["code"] for item in data["warnings"]])


class GatewayTests(ServerTestCase):
    def test_v1_chat_completions_returns_openai_shape(self):
        with patch("brainroute.execution.generate", return_value="hello there"):
            _, data = self.post("/v1/chat/completions", {
                "model": "brainroute-auto",
                "messages": [{"role": "user", "content": "say hi"}],
            })
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["choices"][0]["message"]["content"], "hello there")
        self.assertIn("decision", data["brainroute"])

    def test_v1_chat_completions_streams_openai_chunks(self):
        with patch("brainroute.execution.stream_generate", return_value=iter(["he", "llo"])):
            request = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps({
                    "model": "brainroute-auto",
                    "stream": True,
                    "messages": [{"role": "user", "content": "say hi"}],
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                body = response.read().decode("utf-8")

        self.assertIn("chat.completion.chunk", body)
        self.assertIn("data: [DONE]", body)
        chunks = [
            json.loads(line[5:].strip())
            for line in body.splitlines()
            if line.startswith("data:") and "[DONE]" not in line
        ]
        text = "".join(
            chunk["choices"][0]["delta"].get("content", "")
            for chunk in chunks if "choices" in chunk
        )
        self.assertEqual(text, "hello")

    def test_provider_failure_surfaces_as_502(self):
        from brainroute.providers import ProviderError

        with patch("brainroute.execution.generate", side_effect=ProviderError("down")):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.post("/v1/chat/completions", {
                    "model": "brainroute-auto",
                    "messages": [{"role": "user", "content": "hi"}],
                })
        self.assertEqual(caught.exception.code, 502)


class AuthTests(ServerTestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"BRAINROUTE_API_KEY": "secret-key"})
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_gateway_requires_the_api_key(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/v1/models")
        self.assertEqual(caught.exception.code, 401)

    def test_gateway_accepts_the_api_key(self):
        _, data = self.get("/v1/models", headers={"Authorization": "Bearer secret-key"})
        self.assertEqual(data["object"], "list")

    def test_the_privacy_posture_cannot_be_changed_without_the_key(self):
        # Downgrading the posture is the single most security-relevant write
        # the API exposes.
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post("/api/postures", {"posture": "open"})
        self.assertEqual(caught.exception.code, 401)

    def test_settings_cannot_be_rewritten_without_the_key(self):
        # Unauthenticated writes to policy would let a local page disable the
        # privacy rules or enable an expensive model.
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post("/api/settings", {"policy": {"allow_cloud_for_private": True}})
        self.assertEqual(caught.exception.code, 401)

    def test_read_only_endpoints_stay_open(self):
        _, data = self.get("/healthz")
        self.assertTrue(data["ok"])

    def test_a_wrong_key_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/v1/models", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(caught.exception.code, 401)


if __name__ == "__main__":
    unittest.main()
