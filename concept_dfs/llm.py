import os
import re
from pydantic import BaseModel, Field
from typing import Generator, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from pydantic import SecretStr
from concept_dfs.provider import PROVIDERS, ensure_api_key, resolve_model

SYSTEM_PROMPT = (
    "You are an expert researcher. Given a concept, provide a highly informative explanation "
    "and exactly 3 to 5 related child-concepts or sub-topics for further exploration."
)

EXPLAIN_PROMPT = (
    "You are an expert researcher. Given a concept, provide a clear, concise, "
    "and highly informative explanation. Focus on what it is, why it matters, "
    "and how it works.\n\n"
    "At the very end of your response, on a new line, list 3 to 5 related "
    "sub-concepts for further exploration in this exact format:\n"
    "KEYWORDS: concept1, concept2, concept3"
)

KEYWORDS_PROMPT = (
    "You are an expert researcher. Given a concept and its explanation, "
    "provide exactly 3 to 5 related child-concepts or sub-topics that would "
    "be valuable for further exploration."
)


class ConceptResponse(BaseModel):
    explanation: str = Field(
        description="A clear, concise, and practical explanation of the concept."
    )
    keywords: List[str] = Field(
        description="List of exactly 3 to 5 related sub-concepts or advanced topics to explore next."
    )


class KeywordsResponse(BaseModel):
    keywords: List[str] = Field(
        description="List of exactly 3 to 5 related sub-concepts or advanced topics to explore next."
    )


# Pattern to match KEYWORDS: ... at the end of streamed text
_KEYWORDS_PATTERN = re.compile(
    r"\n\s*(?:\*\*)?KEYWORDS(?:\*\*)?:\s*(.+)$", re.IGNORECASE | re.MULTILINE
)


def parse_keywords_from_text(text: str) -> Tuple[str, List[str]]:
    """Extract inline keywords from streamed LLM output.

    Looks for a trailing line like ``KEYWORDS: concept1, concept2, concept3``
    and splits them out.

    Returns:
        (explanation, keywords) – explanation with the KEYWORDS line stripped,
        and the parsed list of keyword strings.  If no KEYWORDS line is found
        the full text is returned as the explanation with an empty list.
    """
    match = _KEYWORDS_PATTERN.search(text)
    if not match:
        return text.strip(), []

    raw = match.group(1).strip()
    explanation = text[: match.start()].strip()
    keywords = [k.strip().strip("*").strip() for k in raw.split(",") if k.strip()]
    return explanation, keywords


def get_model() -> ChatOpenAI:
    """Parse 'provider:model' string and return a ChatOpenAI with the correct base_url."""
    model_str = resolve_model()
    if ":" in model_str:
        provider, model = model_str.split(":", 1)
    else:
        provider = model_str
        info = PROVIDERS.get(provider)
        model = info.default_model if info else model_str

    # Ensure API key is available (prompt if missing)
    api_key = ensure_api_key(provider)

    info = PROVIDERS.get(provider)
    base_url = info.base_url if info else None

    return ChatOpenAI(
        model=model,
        api_key=SecretStr(api_key),
        base_url=base_url,
    )


def _build_messages(
    query: str, history: Optional[List[Tuple[str, str]]] = None
) -> List[BaseMessage]:
    """Build the message list for a concept query."""
    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    if history:
        for h_query, h_explanation in history:
            messages.append(HumanMessage(content=f"Concept: {h_query}"))
            messages.append(AIMessage(content=h_explanation))

    messages.append(HumanMessage(content=f"Concept: {query}"))
    return messages


def fetch_concept(
    query: str, history: Optional[List[Tuple[str, str]]] = None
) -> ConceptResponse:
    """Fetch an LLM explanation for a concept, optionally with root-to-parent history.

    Args:
        query: The concept to explore.
        history: Optional list of (concept, explanation) tuples representing the
                 path from root to the parent of the current concept.
    """
    llm = get_model().with_structured_output(ConceptResponse)
    messages = _build_messages(query, history)
    result = llm.invoke(messages)
    if isinstance(result, ConceptResponse):
        return result
    # Structured output may return a dict with some providers
    if isinstance(result, dict):
        return ConceptResponse(**result)
    return ConceptResponse(explanation=str(result), keywords=[])


def stream_explanation(
    query: str, history: Optional[List[Tuple[str, str]]] = None
) -> Generator[str, None, None]:
    """Stream a raw-text explanation token by token.

    Yields each text token as it arrives from the LLM, enabling true
    real-time display in the UI.

    Args:
        query: The concept to explore.
        history: Optional list of (concept, explanation) tuples representing the
                 path from root to the parent of the current concept.
    """
    llm = get_model()
    messages: List[BaseMessage] = [SystemMessage(content=EXPLAIN_PROMPT)]

    if history:
        for h_query, h_explanation in history:
            messages.append(HumanMessage(content=f"Concept: {h_query}"))
            messages.append(AIMessage(content=h_explanation))

    messages.append(HumanMessage(content=f"Concept: {query}"))

    stream = llm.stream(messages)
    if stream is None:
        return

    for chunk in stream:
        token: str = chunk.content if hasattr(chunk, "content") else str(chunk)  # type: ignore[assignment]
        if token:
            yield token


def fetch_keywords(
    concept: str,
    explanation: str,
) -> List[str]:
    """Fetch related keywords/sub-concepts given a concept and its explanation.

    Uses structured output to reliably extract a list of keywords.
    This is a fallback for when inline keyword parsing fails.

    Args:
        concept: The concept that was explained.
        explanation: The explanation that was generated for the concept.
    """
    llm = get_model().with_structured_output(KeywordsResponse)
    messages: List[BaseMessage] = [
        SystemMessage(content=KEYWORDS_PROMPT),
        HumanMessage(
            content=(
                f"Concept: {concept}\n\n"
                f"Explanation: {explanation}\n\n"
                "Provide 3 to 5 related sub-concepts for further exploration."
            )
        ),
    ]
    response = llm.invoke(messages)
    if isinstance(response, KeywordsResponse):
        return response.keywords or []
    elif isinstance(response, dict):
        return response.get("keywords") or []
    return []
