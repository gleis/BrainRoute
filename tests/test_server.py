import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from brainroute.server import BrainRouteHandler


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), BrainRouteHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=2)

    def test_config_endpoint(self):
        with urllib.request.urlopen(f"{self.base_url}/api/config") as response:
            data = json.loads(response.read().decode("utf-8"))
        self.assertIn("balanced", data["profiles"])
        self.assertGreaterEqual(len(data["models"]), 3)

    def test_dashboard_endpoint(self):
        with urllib.request.urlopen(f"{self.base_url}/api/dashboard") as response:
            data = json.loads(response.read().decode("utf-8"))
        self.assertIn("models", data)

    def test_v1_models_endpoint(self):
        with urllib.request.urlopen(f"{self.base_url}/v1/models") as response:
            data = json.loads(response.read().decode("utf-8"))
        self.assertEqual(data["object"], "list")

    def test_evals_endpoint(self):
        with urllib.request.urlopen(f"{self.base_url}/api/evals") as response:
            data = json.loads(response.read().decode("utf-8"))
        self.assertTrue(data["results"])

    def test_ask_dry_run_endpoint(self):
        payload = json.dumps({
            "prompt": "Summarize this confidential patient note",
            "profile": "private",
            "execute": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/ask",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode("utf-8"))
        self.assertFalse(data["executed"])
        self.assertEqual(data["decision"]["recommended_model"], "local-qwen2-5")


if __name__ == "__main__":
    unittest.main()
