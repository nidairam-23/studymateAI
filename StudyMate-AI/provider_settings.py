"""Load supported deployment secrets into the environment for existing clients."""

import os
from collections.abc import Mapping, MutableMapping
from typing import Any

from streamlit.errors import StreamlitSecretNotFoundError

SETTING_NAMES = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_MODEL",
    "STUDYMATE_CLAUDE_MODEL",
    "STUDYMATE_OPENAI_MODEL",
)


def load_streamlit_secrets(
    secrets: Mapping[str, Any],
    environ: MutableMapping[str, str] | None = None,
) -> None:
    target = os.environ if environ is None else environ
    for name in SETTING_NAMES:
        value = secrets.get(name)
        if value is not None and str(value).strip():
            target[name] = str(value).strip()


def streamlit_secrets_or_empty(secrets: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return dict(secrets)
    except StreamlitSecretNotFoundError:
        return {}