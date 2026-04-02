import unittest
from unittest.mock import MagicMock, patch

from main import wait_for_ngrok_url


class NgrokSupportTests(unittest.TestCase):
    @patch("main.time.sleep")
    @patch("main.urlopen")
    def test_wait_for_ngrok_url_prefers_https(self, urlopen_mock, _sleep_mock):
        response = MagicMock()
        response.read.return_value = (
            b'{"tunnels":[{"public_url":"http://abc.ngrok-free.app"},{"public_url":"https://abc.ngrok-free.app"}]}'
        )
        context_manager = MagicMock()
        context_manager.__enter__.return_value = response
        context_manager.__exit__.return_value = False
        urlopen_mock.return_value = context_manager

        url = wait_for_ngrok_url(timeout_seconds=0.5)

        self.assertEqual(url, "https://abc.ngrok-free.app")

