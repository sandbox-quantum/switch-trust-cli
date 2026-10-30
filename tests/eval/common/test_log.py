import unittest

from switch_trust.eval.common.log import redact


class TestRedact(unittest.TestCase):
    def test_bearer_token(self) -> None:
        assert redact("Authorization: Bearer abc.def-123") == "[REDACTED]"

    def test_anthropic_style_key(self) -> None:
        assert redact("key=sk-ant-api03-abcdefgh") == "key=[REDACTED]"

    def test_github_token(self) -> None:
        assert redact("token ghp_" + "a" * 36) == "token [REDACTED]"

    def test_bare_authorization_value(self) -> None:
        assert redact("authorization: sometoken123") == "[REDACTED]"

    def test_api_key_param(self) -> None:
        assert redact("api_key=abcdef123456") == "[REDACTED]"

    def test_no_secret_is_unchanged(self) -> None:
        assert redact("hello world, no secrets here") == "hello world, no secrets here"

    def test_api_key_value_is_a_private_key_block(self) -> None:
        # Regression: the key-name pattern's \S+ used to stop at the first
        # space in "api_key: -----BEGIN...", consuming only "-----BEGIN" and
        # leaving the rest of the key material unredacted.
        text = (
            "api_key: -----BEGIN RSA PRIVATE KEY-----\n"
            "MIIB...\n"
            "-----END RSA PRIVATE KEY-----"
        )
        result = redact(text)
        assert "MIIB" not in result
        assert "BEGIN" not in result

    def test_prefixed_key_name_with_bearer_value(self) -> None:
        # Regression: "x-api-key: Bearer <token>" used to redact only up to
        # the first space of the key-name match, leaving the token exposed.
        assert redact("x-api-key: Bearer abcdef123456") == "x-[REDACTED]"

    def test_authorization_basic_scheme(self) -> None:
        # Regression: only "bearer" was special-cased, so a "Basic" (or any
        # other) scheme word leaked the credential after it.
        assert redact("Authorization: Basic dXNlcjpwYXNz") == "[REDACTED]"


if __name__ == "__main__":
    unittest.main()
