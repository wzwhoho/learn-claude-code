"""Offline checks for first-chapter request failures and tool execution.

Run with: .venv/Scripts/python.exe -m unittest discover -s tests -p test_s01_api_recovery.py
"""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import anthropic
try:
    import httpx2 as httpx
except ImportError:
    import httpx


LESSON = Path(__file__).resolve().parents[1] / "s01_agent_loop" / "code.py"


class FirstChapterRecoveryTests(unittest.TestCase):
    def run_repl(self, replies, inputs):
        calls = []
        replies = iter(replies)

        def request(**kwargs):
            calls.append(copy.deepcopy(kwargs))
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        client = Mock()
        client.messages.create.side_effect = request
        output = io.StringIO()
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "MODEL_ID": "test-model", "API_PROTOCOL": "anthropic"}),
            patch("dotenv.load_dotenv"),
            patch("anthropic.Anthropic", return_value=client),
            patch("builtins.input", side_effect=inputs),
            patch("subprocess.run", return_value=SimpleNamespace(stdout="tool ok", stderr="")) as shell,
            contextlib.redirect_stdout(output),
        ):
            runpy.run_path(str(LESSON), run_name="__main__")
        return output.getvalue(), calls, shell

    def test_connection_error_allows_another_prompt(self):
        error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.com"))
        reply = SimpleNamespace(content=[SimpleNamespace(type="text", text="success")])
        output, calls, _ = self.run_repl([error, reply], ["first", "second", "q"])
        self.assertIn("Cannot connect", output)
        self.assertIn("success", output)
        self.assertEqual(calls[1]["messages"], [{"role": "user", "content": "second"}])

    def test_http_errors_show_status_without_response_body(self):
        for status in (400, 401, 403, 404, 429, 500):
            with self.subTest(status=status):
                response = httpx.Response(status, request=httpx.Request("POST", "https://example.com"))
                error = anthropic.APIStatusError("secret-response", response=response, body=None)
                output, _, _ = self.run_repl([error], ["hello", "q"])
                self.assertIn(f"HTTP {status}", output)
                self.assertNotIn("secret-response", output)

    def test_timeout_allows_exit(self):
        error = anthropic.APITimeoutError(request=httpx.Request("POST", "https://example.com"))
        output, _, _ = self.run_repl([error], ["hello", "q"])
        self.assertIn("timed out", output)

    def test_server_error_identifies_model_failure_and_redacts_secrets(self):
        response = httpx.Response(404, request=httpx.Request(
            "POST", "https://user:password@example.com/v1/messages?api_key=test-key"
        ))
        error = anthropic.APIStatusError("failure", response=response, body={
            "error": {"message": "model_not_found; credential=test-key; sk-hidden123"}
        })
        output, _, _ = self.run_repl([error], ["hello", "q"])
        self.assertIn("Server: model_not_found", output)
        self.assertIn("Request: POST https://example.com/v1/messages", output)
        for secret in ("test-key", "sk-hidden123", "password", "api_key="):
            self.assertNotIn(secret, output)

    def test_plain_route_error_is_displayed(self):
        response = httpx.Response(404, request=httpx.Request("POST", "https://example.com/v1/messages"))
        error = anthropic.APIStatusError("failure", response=response, body="404 page not found")
        output, _, _ = self.run_repl([error], ["hello", "q"])
        self.assertIn("Server: 404 page not found", output)

    def test_tool_result_returns_to_model(self):
        tool = SimpleNamespace(type="tool_use", id="tool-1", name="bash", input={"command": "echo hello"})
        final = SimpleNamespace(type="text", text="finished")
        output, calls, shell = self.run_repl(
            [SimpleNamespace(content=[tool]), SimpleNamespace(content=[final])],
            ["run a command", "q"],
        )
        shell.assert_called_once()
        self.assertEqual(calls[1]["messages"][-1]["content"], [
            {"type": "tool_result", "tool_use_id": "tool-1", "content": "tool ok"}
        ])
        self.assertIn("finished", output)

    def test_programming_errors_are_not_hidden(self):
        with self.assertRaisesRegex(ValueError, "bad code"):
            self.run_repl([ValueError("bad code")], ["hello"])


class ChatCompletionsTests(unittest.TestCase):
    def run_repl(self, handler, base="https://example.com", inputs=None):
        output = io.StringIO()
        real_client = httpx.Client
        def create_client(**kwargs):
            return real_client(transport=httpx.MockTransport(handler), **kwargs)
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "MODEL_ID": "test-model",
                                    "API_PROTOCOL": "openai", "ANTHROPIC_BASE_URL": base}),
            patch("dotenv.load_dotenv"),
            patch("anthropic.Anthropic"),
            patch.object(httpx, "Client", side_effect=create_client),
            patch("builtins.input", side_effect=inputs or ["hello", "q"]),
            patch("subprocess.run", return_value=SimpleNamespace(stdout="tool ok", stderr="")) as shell,
            contextlib.redirect_stdout(output),
        ):
            runpy.run_path(str(LESSON), run_name="__main__")
        return output.getvalue(), shell

    def test_text_response_and_endpoint_for_root_or_v1_base(self):
        for base in ("https://example.com", "https://example.com/v1/"):
            with self.subTest(base=base):
                def handler(request):
                    self.assertEqual(str(request.url), "https://example.com/v1/chat/completions")
                    self.assertEqual(request.headers["authorization"], "Bearer test-key")
                    body = json.loads(request.content)
                    self.assertEqual(body["model"], "test-model")
                    self.assertEqual(body["tools"][0]["function"]["name"], "bash")
                    return httpx.Response(200, json={"choices": [{"message": {"content": "hello back"}}]})
                output, shell = self.run_repl(handler, base)
                self.assertIn("hello back", output)
                shell.assert_not_called()

    def test_tool_results_and_reasoning_return_to_model(self):
        calls = []
        def handler(request):
            calls.append(json.loads(request.content))
            if len(calls) == 1:
                return httpx.Response(200, json={"choices": [{"message": {
                    "content": None, "reasoning_content": "need a tool",
                    "tool_calls": [{"id": "call-1", "type": "function", "function": {
                        "name": "bash", "arguments": '{"command":"echo hello"}'
                    }}],
                }}]})
            self.assertEqual(calls[1]["messages"][-2]["reasoning_content"], "need a tool")
            self.assertEqual(calls[1]["messages"][-1], {
                "role": "tool", "tool_call_id": "call-1", "content": "tool ok"
            })
            return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})
        output, shell = self.run_repl(handler)
        self.assertEqual(len(calls), 2)
        shell.assert_called_once()
        self.assertIn("done", output)

    def test_model_error_is_displayed_without_exiting(self):
        def handler(request):
            return httpx.Response(404, json={"error": {"message": "model_not_found"}})
        output, shell = self.run_repl(handler)
        self.assertIn("Server: model_not_found", output)
        self.assertIn("/v1/chat/completions", output)
        shell.assert_not_called()

    def test_truncated_tool_call_does_not_execute(self):
        def handler(request):
            return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": {
                "content": None, "tool_calls": [{"id": "call-1", "function": {
                    "name": "bash", "arguments": '{"command":"echo hello"}'
                }}],
            }}]})
        output, shell = self.run_repl(handler)
        self.assertIn("Incomplete model response", output)
        shell.assert_not_called()


if __name__ == "__main__":
    unittest.main()
