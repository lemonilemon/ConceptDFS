from unittest.mock import MagicMock, patch
from pydantic import SecretStr
from concept_dfs.llm import (
    fetch_concept,
    fetch_keywords,
    parse_keywords_from_text,
    stream_explanation,
    ConceptResponse,
    KeywordsResponse,
)


@patch("concept_dfs.llm.resolve_model", return_value="gemini:gemini-2.5-pro")
@patch("concept_dfs.llm.ensure_api_key", return_value="fake-key")
@patch("concept_dfs.llm.ChatOpenAI")
def test_fetch_concept(mock_chat, mock_ensure, mock_resolve):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = ConceptResponse(
        explanation="This is a test explanation.", keywords=["Keyword 1", "Keyword 2"]
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat.return_value = mock_llm

    result = fetch_concept("test_concept")

    assert isinstance(result, ConceptResponse)
    assert result.explanation == "This is a test explanation."
    assert len(result.keywords) == 2
    assert "Keyword 1" in result.keywords
    assert "Keyword 2" in result.keywords

    # Verify ChatOpenAI was called with api_key (SecretStr) and base_url
    mock_chat.assert_called_once()
    call_kwargs = mock_chat.call_args[1]
    assert isinstance(call_kwargs["api_key"], SecretStr)
    assert call_kwargs["api_key"].get_secret_value() == "fake-key"
    assert "base_url" in call_kwargs
    mock_llm.with_structured_output.assert_called_once_with(ConceptResponse)


@patch("concept_dfs.llm.resolve_model", return_value="gemini:gemini-2.5-pro")
@patch("concept_dfs.llm.ensure_api_key", return_value="fake-key")
@patch("concept_dfs.llm.ChatOpenAI")
def test_fetch_concept_with_history(mock_chat, mock_ensure, mock_resolve):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = ConceptResponse(
        explanation="Deep explanation with context.",
        keywords=["Sub A", "Sub B", "Sub C"],
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat.return_value = mock_llm

    history = [
        ("Root Concept", "Root explanation."),
        ("Child Concept", "Child explanation."),
    ]
    result = fetch_concept("Grandchild Concept", history=history)

    assert isinstance(result, ConceptResponse)
    assert result.explanation == "Deep explanation with context."

    # system + 2 history pairs + 1 query = 6 messages
    call_args = mock_structured.invoke.call_args[0][0]
    assert len(call_args) == 6


@patch("concept_dfs.llm.resolve_model", return_value="gemini:gemini-2.5-pro")
@patch("concept_dfs.llm.ensure_api_key", return_value="fake-key")
@patch("concept_dfs.llm.ChatOpenAI")
def test_fetch_concept_without_history(mock_chat, mock_ensure, mock_resolve):
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = ConceptResponse(
        explanation="Simple explanation.", keywords=["A", "B", "C"]
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat.return_value = mock_llm

    result = fetch_concept("Simple Concept")

    # system + human only
    call_args = mock_structured.invoke.call_args[0][0]
    assert len(call_args) == 2


# ── stream_explanation() tests ──


@patch("concept_dfs.llm.resolve_model", return_value="gemini:gemini-2.5-pro")
@patch("concept_dfs.llm.ensure_api_key", return_value="fake-key")
@patch("concept_dfs.llm.ChatOpenAI")
def test_stream_explanation_yields_tokens(mock_chat, mock_ensure, mock_resolve):
    """stream_explanation yields each token string from the LLM stream."""
    mock_chunk_1 = MagicMock()
    mock_chunk_1.content = "Binary "
    mock_chunk_2 = MagicMock()
    mock_chunk_2.content = "search is "
    mock_chunk_3 = MagicMock()
    mock_chunk_3.content = "an algorithm."

    mock_llm = MagicMock()
    mock_llm.stream.return_value = [mock_chunk_1, mock_chunk_2, mock_chunk_3]
    mock_chat.return_value = mock_llm

    tokens = list(stream_explanation("Binary Search"))

    assert tokens == ["Binary ", "search is ", "an algorithm."]
    # Should NOT use with_structured_output — raw streaming
    mock_llm.with_structured_output.assert_not_called()
    mock_llm.stream.assert_called_once()


@patch("concept_dfs.llm.resolve_model", return_value="gemini:gemini-2.5-pro")
@patch("concept_dfs.llm.ensure_api_key", return_value="fake-key")
@patch("concept_dfs.llm.ChatOpenAI")
def test_stream_explanation_with_history(mock_chat, mock_ensure, mock_resolve):
    """stream_explanation includes history messages in the prompt."""
    mock_chunk = MagicMock()
    mock_chunk.content = "Detailed explanation."

    mock_llm = MagicMock()
    mock_llm.stream.return_value = [mock_chunk]
    mock_chat.return_value = mock_llm

    history = [("Algorithms", "Algorithms are step-by-step procedures.")]
    tokens = list(stream_explanation("Sorting", history=history))

    assert tokens == ["Detailed explanation."]
    # system + 1 history pair (H+A) + 1 query = 4 messages
    call_args = mock_llm.stream.call_args[0][0]
    assert len(call_args) == 4


@patch("concept_dfs.llm.resolve_model", return_value="gemini:gemini-2.5-pro")
@patch("concept_dfs.llm.ensure_api_key", return_value="fake-key")
@patch("concept_dfs.llm.ChatOpenAI")
def test_stream_explanation_skips_empty_tokens(mock_chat, mock_ensure, mock_resolve):
    """stream_explanation skips chunks with empty content."""
    mock_chunk_1 = MagicMock()
    mock_chunk_1.content = ""
    mock_chunk_2 = MagicMock()
    mock_chunk_2.content = "Hello"
    mock_chunk_3 = MagicMock()
    mock_chunk_3.content = ""

    mock_llm = MagicMock()
    mock_llm.stream.return_value = [mock_chunk_1, mock_chunk_2, mock_chunk_3]
    mock_chat.return_value = mock_llm

    tokens = list(stream_explanation("Test"))
    assert tokens == ["Hello"]


# ── fetch_keywords() tests ──


@patch("concept_dfs.llm.resolve_model", return_value="gemini:gemini-2.5-pro")
@patch("concept_dfs.llm.ensure_api_key", return_value="fake-key")
@patch("concept_dfs.llm.ChatOpenAI")
def test_fetch_keywords_returns_list(mock_chat, mock_ensure, mock_resolve):
    """fetch_keywords returns a list of keyword strings."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = KeywordsResponse(
        keywords=["DFS", "BFS", "Dijkstra"]
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat.return_value = mock_llm

    result = fetch_keywords("Graph Traversal", "Graph traversal visits nodes.")

    assert result == ["DFS", "BFS", "Dijkstra"]
    mock_llm.with_structured_output.assert_called_once_with(KeywordsResponse)


@patch("concept_dfs.llm.resolve_model", return_value="gemini:gemini-2.5-pro")
@patch("concept_dfs.llm.ensure_api_key", return_value="fake-key")
@patch("concept_dfs.llm.ChatOpenAI")
def test_fetch_keywords_handles_dict_response(mock_chat, mock_ensure, mock_resolve):
    """fetch_keywords handles dict responses (some LangChain versions)."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = {"keywords": ["A", "B", "C"]}
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat.return_value = mock_llm

    result = fetch_keywords("Test", "Test explanation.")
    assert result == ["A", "B", "C"]


@patch("concept_dfs.llm.resolve_model", return_value="gemini:gemini-2.5-pro")
@patch("concept_dfs.llm.ensure_api_key", return_value="fake-key")
@patch("concept_dfs.llm.ChatOpenAI")
def test_fetch_keywords_prompt_includes_concept_and_explanation(
    mock_chat, mock_ensure, mock_resolve
):
    """fetch_keywords passes both concept and explanation in the prompt."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = KeywordsResponse(keywords=["X"])
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat.return_value = mock_llm

    fetch_keywords("Neural Networks", "Neural networks are computational models.")

    call_args = mock_structured.invoke.call_args[0][0]
    # system + human = 2 messages
    assert len(call_args) == 2
    human_msg = call_args[1].content
    assert "Neural Networks" in human_msg
    assert "Neural networks are computational models." in human_msg


# ── parse_keywords_from_text() tests ──


def test_parse_keywords_basic():
    """Basic KEYWORDS: line is parsed correctly."""
    text = "Binary search is a divide-and-conquer algorithm.\n\nKEYWORDS: divide and conquer, logarithmic time, sorted array"
    explanation, keywords = parse_keywords_from_text(text)
    assert explanation == "Binary search is a divide-and-conquer algorithm."
    assert keywords == ["divide and conquer", "logarithmic time", "sorted array"]


def test_parse_keywords_bold_variant():
    """**KEYWORDS**: bold markdown variant is parsed correctly."""
    text = "Explanation here.\n\n**KEYWORDS**: alpha, beta, gamma"
    explanation, keywords = parse_keywords_from_text(text)
    assert explanation == "Explanation here."
    assert keywords == ["alpha", "beta", "gamma"]


def test_parse_keywords_no_keywords_line():
    """When no KEYWORDS line is present, returns full text and empty list."""
    text = "Just an explanation with no keywords."
    explanation, keywords = parse_keywords_from_text(text)
    assert explanation == "Just an explanation with no keywords."
    assert keywords == []


def test_parse_keywords_case_insensitive():
    """KEYWORDS matching is case-insensitive."""
    text = "Some explanation.\n\nkeywords: one, two, three"
    explanation, keywords = parse_keywords_from_text(text)
    assert explanation == "Some explanation."
    assert keywords == ["one", "two", "three"]


def test_parse_keywords_extra_whitespace():
    """Handles extra whitespace around commas and keywords."""
    text = "Explanation.\n\nKEYWORDS:   foo ,  bar  ,  baz  "
    explanation, keywords = parse_keywords_from_text(text)
    assert explanation == "Explanation."
    assert keywords == ["foo", "bar", "baz"]


def test_parse_keywords_five_keywords():
    """Handles the upper bound of 5 keywords."""
    text = "Explanation text.\n\nKEYWORDS: a, b, c, d, e"
    explanation, keywords = parse_keywords_from_text(text)
    assert explanation == "Explanation text."
    assert len(keywords) == 5
    assert keywords == ["a", "b", "c", "d", "e"]


def test_parse_keywords_with_leading_whitespace_on_line():
    """Handles leading whitespace before KEYWORDS line."""
    text = "Explanation.\n\n  KEYWORDS: x, y, z"
    explanation, keywords = parse_keywords_from_text(text)
    assert explanation == "Explanation."
    assert keywords == ["x", "y", "z"]


def test_parse_keywords_strips_bold_asterisks_from_values():
    """Strips bold asterisks from individual keyword values."""
    text = (
        "Explanation.\n\n**KEYWORDS**: **concept one**, **concept two**, concept three"
    )
    explanation, keywords = parse_keywords_from_text(text)
    assert explanation == "Explanation."
    assert keywords == ["concept one", "concept two", "concept three"]


# ── stream_explanation uses EXPLAIN_PROMPT ──


@patch("concept_dfs.llm.resolve_model", return_value="gemini:gemini-2.5-pro")
@patch("concept_dfs.llm.ensure_api_key", return_value="fake-key")
@patch("concept_dfs.llm.ChatOpenAI")
def test_stream_explanation_uses_explain_prompt(mock_chat, mock_ensure, mock_resolve):
    """stream_explanation uses EXPLAIN_PROMPT (which asks for inline KEYWORDS)."""
    from concept_dfs.llm import EXPLAIN_PROMPT

    mock_chunk = MagicMock()
    mock_chunk.content = "Token"

    mock_llm = MagicMock()
    mock_llm.stream.return_value = [mock_chunk]
    mock_chat.return_value = mock_llm

    list(stream_explanation("Test"))

    call_args = mock_llm.stream.call_args[0][0]
    system_msg = call_args[0]
    assert system_msg.content == EXPLAIN_PROMPT


@patch("concept_dfs.llm.resolve_model", return_value="gemini:gemini-2.5-pro")
@patch("concept_dfs.llm.ensure_api_key", return_value="fake-key")
@patch("concept_dfs.llm.ChatOpenAI")
def test_stream_explanation_handles_none_stream(mock_chat, mock_ensure, mock_resolve):
    """stream_explanation handles None return from llm.stream()."""
    mock_llm = MagicMock()
    mock_llm.stream.return_value = None
    mock_chat.return_value = mock_llm

    tokens = list(stream_explanation("Test"))
    assert tokens == []


# ── fetch_keywords None guard ──


@patch("concept_dfs.llm.resolve_model", return_value="gemini:gemini-2.5-pro")
@patch("concept_dfs.llm.ensure_api_key", return_value="fake-key")
@patch("concept_dfs.llm.ChatOpenAI")
def test_fetch_keywords_handles_none_keywords(mock_chat, mock_ensure, mock_resolve):
    """fetch_keywords returns empty list when response.keywords is None."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    # Simulate a provider returning a KeywordsResponse-like object with None keywords
    mock_response = MagicMock(spec=KeywordsResponse)
    mock_response.keywords = None
    mock_structured.invoke.return_value = mock_response
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat.return_value = mock_llm

    result = fetch_keywords("Test", "Test explanation.")
    assert result == []


# ── fetch_concept fallback cases ──


@patch("concept_dfs.llm.resolve_model", return_value="gemini:gemini-2.5-pro")
@patch("concept_dfs.llm.ensure_api_key", return_value="fake-key")
@patch("concept_dfs.llm.ChatOpenAI")
def test_fetch_concept_handles_dict_response(mock_chat, mock_ensure, mock_resolve):
    """fetch_concept handles dict response from structured output."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = {
        "explanation": "Dict explanation.",
        "keywords": ["A", "B"],
    }
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat.return_value = mock_llm

    result = fetch_concept("Test")
    assert isinstance(result, ConceptResponse)
    assert result.explanation == "Dict explanation."
    assert result.keywords == ["A", "B"]


@patch("concept_dfs.llm.resolve_model", return_value="gemini:gemini-2.5-pro")
@patch("concept_dfs.llm.ensure_api_key", return_value="fake-key")
@patch("concept_dfs.llm.ChatOpenAI")
def test_fetch_concept_handles_unexpected_response(
    mock_chat, mock_ensure, mock_resolve
):
    """fetch_concept handles unexpected response type gracefully."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = "Just a string response"
    mock_llm.with_structured_output.return_value = mock_structured
    mock_chat.return_value = mock_llm

    result = fetch_concept("Test")
    assert isinstance(result, ConceptResponse)
    assert result.explanation == "Just a string response"
    assert result.keywords == []
