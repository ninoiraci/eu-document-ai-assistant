import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

class AppTests(unittest.TestCase):
    def test_demo_query_and_clear(self):
        app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run()
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label == 'Index documents').click().run()
        self.assertFalse(app.exception)
        next(t for t in app.text_input if t.label == 'Question').set_value('Aurora records retained years')
        next(b for b in app.button if b.label == 'Search and answer').click().run()
        self.assertFalse(app.exception)
        self.assertIn('seven years', app.session_state['result']['answer'])
        self.assertEqual(app.session_state['result']['mode'], 'retrieval_only')
        next(b for b in app.button if b.label == 'Clear session').click().run()
        self.assertNotIn('index', app.session_state)
        self.assertNotIn('result', app.session_state)
