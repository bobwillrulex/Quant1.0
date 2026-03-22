import unittest

try:
    from main import create_app
except ModuleNotFoundError:
    create_app = None


class StopLossUITests(unittest.TestCase):
    def setUp(self):
        if create_app is None:
            self.skipTest("Flask is not installed in this environment.")
        try:
            self.app = create_app().test_client()
        except ModuleNotFoundError:
            self.skipTest("Flask is not installed in this environment.")

    def test_stop_loss_dropdown_and_fixed_input_present(self):
        response = self.app.get("/")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Stop Loss Strategy", html)
        self.assertIn('option value="none"', html)
        self.assertIn('option value="atr"', html)
        self.assertIn('option value="model_invalidation"', html)
        self.assertIn('option value="time_decay"', html)
        self.assertIn('option value="fixed_percentage"', html)
        self.assertIn('id="fixedStopLossWrap"', html)
        self.assertIn("toggleFixedStopField", html)


if __name__ == "__main__":
    unittest.main()
