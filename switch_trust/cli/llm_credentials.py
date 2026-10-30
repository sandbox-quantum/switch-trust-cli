"""What each LLM provider will authenticate with, read from the environment.

Three places have to answer "does this model have a credential?" before doing
any work, and they had each answered it from scratch:

- `inventory/common/llm_provider.py::api_key_present` — the scanner's
  preflight, which decides whether the LLM stage runs at all.
- `cli/eval_cli.py::_check_target_credentials` — the eval CLI's preflight on
  the model *under test*.
- `eval/core/models/generator_model.py::GeneratorConfig.from_env` — whether
  the judge/generator client can be built.

They drifted, which is the whole reason this module exists: the eval CLI's
copy knew only about `GOOGLE_API_KEY`/`GEMINI_API_KEY` and so refused — at
exit 20, with no override — a gemini target that `llm_provider`'s copy, one
package over, correctly accepted on `GOOGLE_GENAI_USE_VERTEXAI`.

**The provider is not the whole question; the client matters too**, and for
Google every live caller builds the same one:

| caller | client | keyless credential |
|---|---|---|
| scanner preflight | ADK `Gemini`, i.e. google-genai `Client()` | the Vertex flag only |
| eval target, judge | google-genai `Client()` | the Vertex flag only |

`GOOGLE_APPLICATION_CREDENTIALS` on its own leaves google-genai in
Developer-API mode, where `Client()` still raises for a missing key, so it is
not a credential for any of them.

This table used to put the scanner in a wider ADC row, on the premise that it
"reaches Google through ADK and litellm". Only its *non-Google* providers go
through litellm: `llm_provider.make_model` answers `provider == "google"` with
ADK's `Gemini`, which is google-genai's `Client()` underneath. Counting ADC
there made the preflight report a credential the client could not spend, and
every runner holding `GOOGLE_APPLICATION_CREDENTIALS` for unrelated GCP access
failed both LLM stages instead of skipping them — ~16.5k CI failures a month,
reported to customers as a rejected key.

So the split that remains is between the *keyless* question and the whole one,
not between two clients: `vertexai_mode_enabled` answers the first and
`genai_client_credential_present` the second (a key under either name, or
Vertex mode). Callers that already check a key separately want the narrow one.

Everything here answers only whether a credential is **present**. Whether the
provider accepts it takes a call to find out, and belongs to whatever handles
the failure afterwards.

Lives in `common/` rather than in any of the three callers because `cli/`
already depends on `inventory/` (`lib_cli` -> `lib_agent_scanner`), so putting
it in either would make that edge circular — the same reasoning that put
`exit_codes.py` here. See its docstring.
"""

import os

# Canonical provider ids. These match `llm_provider`'s resolved provider names
# (after its `gemini` -> `google` aliasing), which is what lets that module
# key straight into `API_KEY_ENV_VARS` with no translation of its own.
PROVIDER_GOOGLE = "google"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"

# The env var(s) each provider's SDK reads an API key from when none is passed
# explicitly. Only the providers we can actually check for are listed, so a
# caller can treat "no entry" as "assume configured" and never refuse an
# exotic provider on a guess.
#
# Google has two names because google-genai accepts either.
API_KEY_ENV_VARS: dict[str, tuple[str, ...]] = {
    PROVIDER_GOOGLE: ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    PROVIDER_OPENAI: ("OPENAI_API_KEY",),
    PROVIDER_ANTHROPIC: ("ANTHROPIC_API_KEY",),
}

# What google-genai itself reads out of GOOGLE_GENAI_USE_VERTEXAI to decide a
# client is Vertex-mode. Mirrored from the SDK rather than re-invented,
# because the point of every check below is to predict what it will do —
# notably that `false` and `0` are off, and that an unrecognised value is off
# too rather than being treated as an opt-in.
_VERTEXAI_ENV_TRUE = frozenset({"true", "1"})


def _is_set(name: str) -> bool:
    """Whether *name* holds something other than blank.

    An exported-but-empty variable is the common shape of this mistake
    (`GEMINI_API_KEY=` in a CI job's env block), and every SDK here treats it
    as no credential at all, so it must not read as one.
    """
    return bool(os.environ.get(name, "").strip())


def api_key_env_present(provider: str) -> bool:
    """Whether an API key is exported for *provider*.

    False for a provider with no entry in `API_KEY_ENV_VARS`, which is only
    the right answer when the caller has already decided that an unmapped
    provider is not its business — `api_key_present` in `llm_provider` checks
    for the entry first, precisely so it can assume-configured instead.
    """
    return any(_is_set(var) for var in API_KEY_ENV_VARS.get(provider, ()))


def vertexai_mode_enabled() -> bool:
    """Whether GOOGLE_GENAI_USE_VERTEXAI puts google-genai in Vertex mode.

    In that mode `Client()` fills the project in from ADC and needs no API
    key; the "No API key was provided" raise lives in the Developer-API
    branch it never reaches.
    """
    value = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower()
    return value in _VERTEXAI_ENV_TRUE


def google_adc_present() -> bool:
    """Whether Google credentials exist via a google-auth ADC path.

    Wider than `vertexai_mode_enabled` by `GOOGLE_APPLICATION_CREDENTIALS`,
    which google-auth reads directly.

    **No caller, deliberately.** It describes a litellm route to Google, and
    nothing here builds one: the scanner's Google path is ADK's `Gemini`, i.e.
    google-genai's `Client()`, which this predicate is wrong for. Do not reach
    for it as the Google check — see the module docstring. It stays because the
    route is a plausible future (`make_model` growing real Vertex support), and
    the answer is then already written down.
    """
    return vertexai_mode_enabled() or _is_set("GOOGLE_APPLICATION_CREDENTIALS")


def genai_client_credential_present() -> bool:
    """Whether google-genai's `Client()` can be built with no explicit key.

    The union of the two ways it authenticates itself from the environment:
    a Developer-API key under either name, or Vertex mode. Anything short of
    this and the constructor raises.
    """
    return api_key_env_present(PROVIDER_GOOGLE) or vertexai_mode_enabled()


def anthropic_auth_token_present() -> bool:
    """Whether ANTHROPIC_AUTH_TOKEN is exported.

    The Anthropic SDK resolves it into an `Authorization: Bearer` header as
    an alternative to the API key — how a target behind a gateway is
    configured — and raises for neither credential at construction time.
    """
    return _is_set("ANTHROPIC_AUTH_TOKEN")
