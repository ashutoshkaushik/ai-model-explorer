"""Thin wrapper around the Anthropic client. The API key comes from st.secrets or the environment."""

import os

import anthropic
import streamlit as st

MODEL = "claude-sonnet-5"


def _api_key() -> str | None:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return os.getenv("ANTHROPIC_API_KEY")


def get_client() -> anthropic.Anthropic | None:
    key = _api_key()
    return anthropic.Anthropic(api_key=key) if key else None
