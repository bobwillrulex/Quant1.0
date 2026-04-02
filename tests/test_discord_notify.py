import unittest
from unittest.mock import MagicMock, patch

from quant.discord_notify import send_discord_webhook


class DiscordNotifyTests(unittest.TestCase):
    @patch("quant.discord_notify.urllib.request.urlopen")
    def test_send_discord_webhook_posts_json_payload(self, urlopen_mock):
        response = MagicMock()
        response.status = 204
        urlopen_mock.return_value.__enter__.return_value = response

        webhook_url = "https://discord.com/api/webhooks/123/token"
        send_discord_webhook(webhook_url, "Bot is online 🚀")

        self.assertEqual(urlopen_mock.call_count, 1)
        request_obj = urlopen_mock.call_args[0][0]
        timeout = urlopen_mock.call_args[1]["timeout"]

        self.assertEqual(timeout, 10.0)
        self.assertEqual(request_obj.get_method(), "POST")
        self.assertEqual(request_obj.full_url, webhook_url)
        self.assertEqual(
            request_obj.headers.get("Content-type"),
            "application/json",
        )
        self.assertEqual(
            request_obj.headers.get("User-agent"),
            "Quant1.0-DiscordWebhook/1.0",
        )
        self.assertEqual(
            request_obj.data.decode("utf-8"),
            '{"content": "Bot is online \\ud83d\\ude80"}',
        )

    def test_send_discord_webhook_requires_non_empty_url(self):
        with self.assertRaisesRegex(ValueError, "Discord webhook URL is required."):
            send_discord_webhook("   ", "hello")


if __name__ == "__main__":
    unittest.main()
