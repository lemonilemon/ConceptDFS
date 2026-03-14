import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from concept_dfs.db import init_db, insert_node, create_session
from concept_dfs.app import build_report


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_concepts.db"
    os.environ["CONCEPT_DFS_DB"] = str(db_file)
    init_db()
    yield
    if db_file.exists():
        os.remove(db_file)
    os.environ.pop("CONCEPT_DFS_DB", None)


# ── build_report() tests (pure function) ──


def test_build_report_empty(test_db):
    result = build_report()
    assert result is None


def test_build_report_with_data(test_db):
    insert_node(None, "Concept A", "Explanation A")
    insert_node("Concept A", "Concept B", "Explanation B")

    content = build_report()
    assert content is not None
    assert "# ConceptDFS Exploration Report" in content
    assert "## Knowledge Graph" in content
    assert "## Concepts Explained" in content
    assert "### Concept A" in content
    assert "### Concept B" in content
    assert "Explanation A" in content
    assert "Explanation B" in content
    assert "graph TD;" in content


def test_build_report_with_session_id(test_db):
    """build_report(session_id=N) only includes nodes/edges from that session."""
    s1 = create_session("Session 1")
    s2 = create_session("Session 2")

    insert_node(None, "A", "A explanation", session_id=s1)
    insert_node("A", "B", "B explanation", session_id=s1)

    insert_node(None, "C", "C explanation", session_id=s2)
    insert_node("C", "D", "D explanation", session_id=s2)

    content_s1 = build_report(session_id=s1)
    assert content_s1 is not None
    assert "### A" in content_s1
    assert "### B" in content_s1
    assert "### C" not in content_s1
    assert "### D" not in content_s1
    assert "Session 1" in content_s1  # session metadata

    content_s2 = build_report(session_id=s2)
    assert content_s2 is not None
    assert "### C" in content_s2
    assert "### D" in content_s2
    assert "### A" not in content_s2

    # No session_id — all nodes
    content_all = build_report()
    assert content_all is not None
    assert "### A" in content_all
    assert "### B" in content_all
    assert "### C" in content_all
    assert "### D" in content_all


def test_build_report_empty_session(test_db):
    """build_report for a session with no nodes returns None."""
    sid = create_session("Empty")
    assert build_report(session_id=sid) is None


# ── CLI export_report() wrapper tests ──


def test_export_report_empty(test_db):
    from concept_dfs.cli import export_report

    with patch("concept_dfs.cli.console.print") as mock_print:
        export_report()
        mock_print.assert_called_with(
            "[yellow]No concepts found in database to export.[/yellow]"
        )


def test_export_report_with_data(test_db, tmp_path):
    insert_node(None, "Concept A", "Explanation A")
    insert_node("Concept A", "Concept B", "Explanation B")

    from concept_dfs.cli import export_report

    # Run in tmp_path so report.md is created there
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        export_report()
        report_path = tmp_path / "report.md"
        assert report_path.exists()

        content = report_path.read_text()
        assert "# ConceptDFS Exploration Report" in content
        assert "### Concept A" in content
        assert "### Concept B" in content
    finally:
        os.chdir(original_dir)


def test_export_report_with_session(test_db, tmp_path):
    """export_report(session_id=N) exports only that session."""
    sid = create_session("Export Session")
    insert_node(None, "X", "X exp", session_id=sid)
    insert_node("X", "Y", "Y exp", session_id=sid)

    # Also add data outside the session
    insert_node(None, "Z", "Z exp")

    from concept_dfs.cli import export_report

    original_dir = os.getcwd()
    os.chdir(tmp_path)
    try:
        export_report(session_id=sid)
        report_path = tmp_path / "report.md"
        assert report_path.exists()

        content = report_path.read_text()
        assert "### X" in content
        assert "### Y" in content
        assert "### Z" not in content
    finally:
        os.chdir(original_dir)


# ── show_sessions() tests ──


def test_show_sessions_empty(test_db):
    from concept_dfs.cli import show_sessions

    with patch("concept_dfs.cli.console.print") as mock_print:
        show_sessions()
        mock_print.assert_called_with("[dim]No sessions found.[/dim]")


def test_show_sessions_lists_sessions(test_db):
    sid = create_session("My Explore")
    insert_node(None, "Root", "Root exp", session_id=sid)
    insert_node("Root", "Child", "Child exp", session_id=sid)

    from concept_dfs.cli import show_sessions

    with patch("concept_dfs.cli.console.print") as mock_print:
        show_sessions()
        # Should have been called multiple times (header + each session + trailing newline)
        assert mock_print.call_count >= 2


# ── main() dispatch tests ──


@patch("concept_dfs.cli.select_model")
def test_main_model_subcommand(mock_select, test_db):
    from concept_dfs.cli import main

    with patch.object(sys, "argv", ["concept-dfs", "model"]):
        main()

    mock_select.assert_called_once()


@patch("concept_dfs.cli.force_auth")
def test_main_auth_subcommand(mock_auth, test_db):
    from concept_dfs.cli import main

    with patch.object(sys, "argv", ["concept-dfs", "auth"]):
        main()

    mock_auth.assert_called_once()


@patch("concept_dfs.cli.export_report")
def test_main_export_subcommand(mock_export, test_db):
    from concept_dfs.cli import main

    with patch.object(sys, "argv", ["concept-dfs", "export"]):
        main()

    mock_export.assert_called_once_with(session_id=None)


@patch("concept_dfs.cli.export_report")
def test_main_export_with_session_flag(mock_export, test_db):
    """concept-dfs export --session 3 passes session_id=3."""
    from concept_dfs.cli import main

    with patch.object(sys, "argv", ["concept-dfs", "export", "--session", "3"]):
        main()

    mock_export.assert_called_once_with(session_id=3)


@patch("concept_dfs.cli.export_report")
def test_main_export_invalid_session_id(mock_export, test_db):
    """concept-dfs export --session abc shows an error and does not call export."""
    from concept_dfs.cli import main

    with patch("concept_dfs.cli.console.print") as mock_print:
        with patch.object(sys, "argv", ["concept-dfs", "export", "--session", "abc"]):
            main()

    mock_export.assert_not_called()
    mock_print.assert_called_with("[red]Invalid session ID.[/red]")


@patch("concept_dfs.cli.show_sessions")
def test_main_sessions_subcommand(mock_show, test_db):
    from concept_dfs.cli import main

    with patch.object(sys, "argv", ["concept-dfs", "sessions"]):
        main()

    mock_show.assert_called_once()


@patch("concept_dfs.cli.ConceptDFSApp")
def test_main_no_args_launches_tui(mock_app_cls, test_db):
    """When no args, main() launches the Textual TUI."""
    from concept_dfs.cli import main

    mock_app = MagicMock()
    mock_app_cls.return_value = mock_app

    with patch.object(sys, "argv", ["concept-dfs"]):
        main()

    mock_app_cls.assert_called_once_with()
    mock_app.run.assert_called_once()


@patch("concept_dfs.cli.ConceptDFSApp")
def test_main_with_concept_arg_launches_tui(mock_app_cls, test_db):
    """When a concept is given as arg, main() launches TUI with that concept."""
    from concept_dfs.cli import main

    mock_app = MagicMock()
    mock_app_cls.return_value = mock_app

    with patch.object(sys, "argv", ["concept-dfs", "Graph Theory"]):
        main()

    mock_app_cls.assert_called_once_with(initial_concept="Graph Theory")
    mock_app.run.assert_called_once()


@patch("concept_dfs.cli.ConceptDFSApp")
def test_main_multi_word_concept(mock_app_cls, test_db):
    """Multi-word concepts are joined from remaining argv."""
    from concept_dfs.cli import main

    mock_app = MagicMock()
    mock_app_cls.return_value = mock_app

    with patch.object(sys, "argv", ["concept-dfs", "Machine", "Learning"]):
        main()

    mock_app_cls.assert_called_once_with(initial_concept="Machine Learning")
    mock_app.run.assert_called_once()
