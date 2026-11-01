from types import SimpleNamespace

from provider_settings import load_streamlit_secrets, streamlit_secrets_or_empty
from streamlit.errors import StreamlitSecretNotFoundError
from tests import chatassisstant
from tests.chatassisstant import OFFLINE, available_providers


def test_offline_provider_is_listed_first(monkeypatch):
    monkeypatch.setattr(chatassisstant.ollama_provider, "available", lambda: False)
    monkeypatch.setattr(chatassisstant.ollama_provider, "available", lambda: True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    assert available_providers() == [OFFLINE]


def test_configured_provider_keys_are_enabled(monkeypatch):
    monkeypatch.setattr(chatassisstant.ollama_provider, "available", lambda: False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    assert available_providers() == ["Claude", "ChatGPT"]


def test_streamlit_secrets_configure_provider_environment():
    environment = {"OPENAI_API_KEY": "old-key"}

    load_streamlit_secrets({"ANTHROPIC_API_KEY": " cloud-key ", "OPENAI_API_KEY": ""}, environment)

    assert environment == {"ANTHROPIC_API_KEY": "cloud-key", "OPENAI_API_KEY": "old-key"}


def test_missing_local_streamlit_secrets_are_treated_as_empty():
    class MissingSecrets:
        def __iter__(self):
            raise StreamlitSecretNotFoundError("No local secrets configured")

    assert streamlit_secrets_or_empty(MissingSecrets()) == {}


def test_audio_transcription_joins_segments_without_loading_a_model(monkeypatch):
    class FakeModel:
        def transcribe(self, audio_stream):
            assert audio_stream.read() == b"fake audio"
            return [SimpleNamespace(text=" Plan my "), SimpleNamespace(text="study week. ")], None

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(chatassisstant, "_local_whisper", lambda: FakeModel())

    assert chatassisstant.transcribe_audio(b"fake audio") == "Plan my study week."
