
import os

from crewai import LLM


def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it in Streamlit Cloud Secrets."
        )

    return LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.1,
        max_tokens=2500,
        timeout=120,
    )
