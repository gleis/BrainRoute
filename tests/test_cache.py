import unittest

from modelroute.cache import TTLCache


class TTLCacheTests(unittest.TestCase):
    def test_values_round_trip(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("k", 1)
        self.assertEqual(cache.get("k"), 1)

    def test_expired_entries_are_dropped(self):
        cache = TTLCache(ttl_seconds=-1)
        cache.set("k", 1)
        self.assertIsNone(cache.get("k"))

    def test_factory_runs_once_per_key(self):
        cache = TTLCache(ttl_seconds=60)
        calls = {"count": 0}

        def factory():
            calls["count"] += 1
            return "value"

        self.assertEqual(cache.get_or_set("k", factory), "value")
        self.assertEqual(cache.get_or_set("k", factory), "value")
        self.assertEqual(calls["count"], 1)

    def test_cache_is_bounded(self):
        cache = TTLCache(ttl_seconds=60, max_entries=8)
        for index in range(50):
            cache.set(index, index)
        self.assertLessEqual(len(cache._entries), 8)

    def test_clear_empties_the_cache(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("k", 1)
        cache.clear()
        self.assertIsNone(cache.get("k"))


if __name__ == "__main__":
    unittest.main()
