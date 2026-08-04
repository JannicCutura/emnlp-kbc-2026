from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import requests

from lm_kbc import client as client_module
from lm_kbc.client import LMStudioClient, load_env_file, resolve_api_key
from lm_kbc.config import LMStudioConfig


def _response(status: int, body: object, headers: dict[str, str] | None = None):
    response = mock.Mock(spec=requests.Response)
    response.status_code = status
    response.headers = headers or {}
    response.text = str(body)
    response.json.return_value = body
    response.raise_for_status.side_effect = (
        requests.HTTPError(f"HTTP {status}") if status >= 400 else None
    )
    return response


def _ok(content: str = '{"answers": ["Haiti"]}'):
    return _response(
        200, {"choices": [{"message": {"content": content}}], "usage": {}}
    )


class ApiKeyResolutionTest(unittest.TestCase):
    def test_yaml_key_is_used_when_no_env_name_is_given(self):
        config = LMStudioConfig(model="m", model_parameters_billion=27.4)
        self.assertEqual(resolve_api_key(config), "lm-studio")

    def test_env_variable_wins_over_yaml_key(self):
        config = LMStudioConfig(
            model="m", model_parameters_billion=27.4, api_key_env="TEST_KEY_NAME"
        )
        with mock.patch.dict(os.environ, {"TEST_KEY_NAME": "sk-or-v1-secret"}):
            self.assertEqual(resolve_api_key(config), "sk-or-v1-secret")

    def test_missing_environment_variable_fails_fast(self):
        config = LMStudioConfig(
            model="m", model_parameters_billion=27.4, api_key_env="ABSENT_KEY_NAME"
        )
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ABSENT_KEY_NAME", None)
            with self.assertRaisesRegex(ValueError, "ABSENT_KEY_NAME"):
                resolve_api_key(config)

    def test_env_file_does_not_override_the_real_environment(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "# comment\nFROM_FILE=file-value\nPREEXISTING=file-value\n",
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"PREEXISTING": "real-value"}):
                client_module._env_file_loaded = False
                try:
                    load_env_file(path)
                    self.assertEqual(os.environ["FROM_FILE"], "file-value")
                    self.assertEqual(os.environ["PREEXISTING"], "real-value")
                finally:
                    os.environ.pop("FROM_FILE", None)
                    client_module._env_file_loaded = False


class ChatRetryTest(unittest.TestCase):
    def setUp(self):
        self.config = LMStudioConfig(
            base_url="https://openrouter.ai/api/v1",
            model="google/gemma-3-27b-it",
            model_parameters_billion=27.4,
            retry_backoff_seconds=0.001,
        )

    def _client(self):
        with mock.patch("lm_kbc.client.requests.Session"):
            return LMStudioClient(self.config)

    def _chat(self, client):
        return client.chat(
            [{"role": "user", "content": "hi"}],
            temperature=0.6,
            top_p=0.95,
            max_tokens=64,
            seed=42,
        )

    def test_rate_limit_is_retried_then_succeeds(self):
        client = self._client()
        client.session.post.side_effect = [
            _response(429, "slow down", {"Retry-After": "0"}),
            _ok(),
        ]
        with mock.patch("lm_kbc.client.time.sleep"):
            content, _ = self._chat(client)
        self.assertIn("Haiti", content)
        self.assertEqual(client.session.post.call_count, 2)

    def test_connection_error_is_retried(self):
        client = self._client()
        client.session.post.side_effect = [
            requests.ConnectionError("reset by peer"),
            _ok(),
        ]
        with mock.patch("lm_kbc.client.time.sleep"):
            content, _ = self._chat(client)
        self.assertIn("Haiti", content)

    def test_openrouter_error_payload_with_status_200_is_retried(self):
        client = self._client()
        client.session.post.side_effect = [
            _response(200, {"error": {"code": 502, "message": "upstream down"}}),
            _ok(),
        ]
        with mock.patch("lm_kbc.client.time.sleep"):
            content, _ = self._chat(client)
        self.assertIn("Haiti", content)

    def test_permanent_gateway_rejection_is_not_retried(self):
        client = self._client()
        client.session.post.return_value = _response(
            200, {"error": {"code": 400, "message": "bad model id"}}
        )
        with self.assertRaisesRegex(RuntimeError, "bad model id"):
            self._chat(client)
        self.assertEqual(client.session.post.call_count, 1)

    def test_retries_are_bounded_and_then_raise(self):
        config = LMStudioConfig(
            model="m",
            model_parameters_billion=27.4,
            max_retries=2,
            retry_backoff_seconds=0.001,
        )
        with mock.patch("lm_kbc.client.requests.Session"):
            client = LMStudioClient(config)
        client.session.post.return_value = _response(503, "unavailable")
        with mock.patch("lm_kbc.client.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                self._chat(client)
        self.assertEqual(client.session.post.call_count, 3)

    def test_provider_routing_is_sent_when_configured(self):
        config = LMStudioConfig(
            model="m",
            model_parameters_billion=27.4,
            provider_routing={"allow_fallbacks": False, "order": ["deepinfra"]},
        )
        with mock.patch("lm_kbc.client.requests.Session"):
            client = LMStudioClient(config)
        client.session.post.return_value = _ok()
        self._chat(client)
        payload = client.session.post.call_args.kwargs["json"]
        self.assertEqual(payload["provider"]["order"], ["deepinfra"])

    def test_reasoning_content_is_used_when_content_is_empty(self):
        client = self._client()
        client.session.post.return_value = _response(
            200,
            {
                "choices": [
                    {"message": {"content": "", "reasoning_content": '["Haiti"]'}}
                ]
            },
        )
        content, _ = self._chat(client)
        self.assertEqual(content, '["Haiti"]')

class UnsupportedParameterTest(unittest.TestCase):
    def test_provider_rejecting_seed_is_retried_without_it(self):
        config = LMStudioConfig(
            model="google/gemma-4-31b-it",
            model_parameters_billion=30.7,
            retry_backoff_seconds=0.001,
        )
        with mock.patch("lm_kbc.client.requests.Session"):
            client = LMStudioClient(config)
        client.session.post.side_effect = [
            _response(
                200,
                {"error": {"message": "Upstream error from OpenInference: JAX "
                                      "does not support per-request seed."}},
            ),
            _ok(),
        ]
        with mock.patch("lm_kbc.client.time.sleep"):
            content, _ = client.chat(
                [{"role": "user", "content": "hi"}],
                temperature=0.6, top_p=0.95, max_tokens=64, seed=42,
            )
        self.assertIn("Haiti", content)
        first, second = client.session.post.call_args_list
        self.assertEqual(first.kwargs["json"]["seed"], 42)
        self.assertNotIn("seed", second.kwargs["json"])

    def test_ordinary_gateway_errors_do_not_strip_parameters(self):
        config = LMStudioConfig(
            model="m", model_parameters_billion=27.4, retry_backoff_seconds=0.001
        )
        with mock.patch("lm_kbc.client.requests.Session"):
            client = LMStudioClient(config)
        client.session.post.side_effect = [_response(503, "overloaded"), _ok()]
        with mock.patch("lm_kbc.client.time.sleep"):
            client.chat(
                [{"role": "user", "content": "hi"}],
                temperature=0.6, top_p=0.95, max_tokens=64, seed=42,
            )
        for call in client.session.post.call_args_list:
            self.assertEqual(call.kwargs["json"]["seed"], 42)

if __name__ == "__main__":
    unittest.main()
