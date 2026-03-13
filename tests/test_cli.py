import os
import pytest
from unittest.mock import MagicMock, patch
from src.cli import export_report
from src.db import init_db, insert_node

@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_concepts.db"
    os.environ["CONCEPT_DFS_DB"] = str(db_file)
    init_db()
    yield
    if db_file.exists():
        os.remove(db_file)

def test_export_report_empty(test_db):
    with patch("src.cli.console.print") as mock_print:
        export_report()
        mock_print.assert_called_with("[yellow]No concepts found in database to export.[/yellow]")

def test_export_report_with_data(test_db):
    insert_node(None, "Concept A", "Explanation A")
    insert_node("Concept A", "Concept B", "Explanation B")
    
    # Run the export function
    export_report()
    
    assert os.path.exists("report.md")
    
    with open("report.md", "r") as f:
        content = f.read()
        assert "# ConceptDFS Exploration Report" in content
        assert "## Knowledge Graph" in content
        assert "## Concepts Explained" in content
        assert "### Concept A" in content
        assert "### Concept B" in content
        assert "Explanation A" in content
        assert "Explanation B" in content
        assert "graph TD;" in content
    
    # Cleanup
    os.remove("report.md")

@patch("src.cli.Prompt.ask")
@patch("src.cli.fetch_concept")
def test_main_loop(mock_fetch_concept, mock_ask, tmp_path):
    # Set up the environment for the test
    db_file = tmp_path / "test_main_concepts.db"
    os.environ["CONCEPT_DFS_DB"] = str(db_file)
    from src.db import init_db
    init_db()
    
    from src.llm import ConceptResponse
    
    # 1. 'Graph Theory' is not in cache, so fetch_concept is called.
    # 2. 'Keywords' are presented, Prompt.ask('Your choice') is called.
    # 3. 'Graph Types' is pushed onto the stack.
    # 4. 'Graph Types' is popped, fetch_concept is called.
    # 5. 'Graph Types' has no keywords, Prompt.ask('Your choice') is not called.
    # 6. 'Would you like to export...?' Prompt.ask is called.
    
    mock_fetch_concept.side_effect = [
        ConceptResponse(explanation="Graph Theory Explanation", keywords=["Graph Types", "Algorithms"]),
        ConceptResponse(explanation="Graph Types Explanation", keywords=[])
    ]
    
    mock_ask.side_effect = [
        "1", # Choice for Graph Theory keywords: selects "Graph Types"
        "n"  # Final report question: "no"
    ]
    
    import sys
    from src.cli import main
    
    # Mock sys.argv
    with patch.object(sys, 'argv', ['concept-dfs', 'Graph Theory']):
        main()
        
    # Check that fetch_concept was called for the initial query and the selected keyword
    assert mock_fetch_concept.call_count == 2
    mock_fetch_concept.assert_any_call("Graph Theory")
    mock_fetch_concept.assert_any_call("Graph Types")
