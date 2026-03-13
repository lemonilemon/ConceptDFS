from unittest.mock import MagicMock, patch
from src.llm import fetch_concept, ConceptResponse

@patch("src.llm.completion")
def test_fetch_concept(mock_completion):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"explanation": "This is a test explanation.", "keywords": ["Keyword 1", "Keyword 2"]}'
    mock_completion.return_value = mock_response
    
    result = fetch_concept("test_concept")
    
    assert isinstance(result, ConceptResponse)
    assert result.explanation == "This is a test explanation."
    assert len(result.keywords) == 2
    assert "Keyword 1" in result.keywords
    assert "Keyword 2" in result.keywords

@patch("src.llm.completion")
def test_fetch_concept_with_invalid_json(mock_completion):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = 'Invalid JSON response'
    mock_completion.return_value = mock_response
    
    try:
        fetch_concept("test_concept")
    except Exception:
        pass  # Expected to fail when parsing invalid JSON
