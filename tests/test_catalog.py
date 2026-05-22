import unittest

from brainroute.catalog import overlay_models


class CatalogTests(unittest.TestCase):
    def test_discovered_models_do_not_replace_configured_models(self):
        configured = [{"id": "local", "enabled": True, "name": "Configured"}]
        discovered = [{"id": "local", "enabled": False}, {"id": "new", "enabled": False}]
        models = {model["id"]: model for model in overlay_models(configured, discovered)}
        self.assertTrue(models["local"]["enabled"])
        self.assertFalse(models["new"]["enabled"])


if __name__ == "__main__":
    unittest.main()
