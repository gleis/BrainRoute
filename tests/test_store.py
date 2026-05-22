import unittest

from brainroute.store import append_message, create_session, get_session, record_run


class StoreTests(unittest.TestCase):
    def test_persists_session_messages(self):
        session_id = create_session()
        append_message(session_id, "user", "hello")
        session = get_session(session_id)
        self.assertEqual(session["messages"][-1]["content"], "hello")

    def test_records_run(self):
        record_run({"model_id": "test", "provider": "mock", "ok": True})


if __name__ == "__main__":
    unittest.main()
