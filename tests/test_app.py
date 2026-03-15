"""Tests for Textual TUI app: KeywordSelector, GraphScreen, and tree styling."""

import os
import pytest
from unittest.mock import patch

from rich.text import Text
from textual.widgets import Tree

from concept_dfs.app import (
    ConceptDFSApp,
    GraphScreen,
    KeywordSelector,
    ModelSelectScreen,
    AuthScreen,
    SessionListScreen,
)
from concept_dfs.db import create_session, init_db, insert_node


def _markup(label: str | Text) -> str:
    """Return the Rich markup string for a tree-node label."""
    if isinstance(label, Text):
        return label.markup
    return label


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_concepts.db"
    os.environ["CONCEPT_DFS_DB"] = str(db_file)
    init_db()
    yield
    if db_file.exists():
        os.remove(db_file)
    os.environ.pop("CONCEPT_DFS_DB", None)


# ── SelectionList keyword selection tests ──


@pytest.mark.asyncio
async def test_selection_list_select_keywords(test_db):
    """Toggling keywords in SelectionList and confirming pushes them to stack."""
    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        app._current_concept = "Graphs"
        app._current_keywords = ["DFS", "BFS", "Dijkstra"]
        app._state = "waiting_selection"

        with patch.object(app, "_process_next"):
            app._show_keyword_selector(["DFS", "BFS", "Dijkstra"])
            await pilot.pause()

            selector = app.query_one(".keyword-selector", KeywordSelector)
            # Toggle by value (not index)
            selector.toggle("DFS")
            selector.toggle("Dijkstra")
            await pilot.pause()

            selector.focus()
            await pilot.pause()
            app.action_confirm_selection()

        # Selected DFS and Dijkstra, pushed in reversed order
        assert len(app._stack) == 2
        assert app._stack[0] == ("Graphs", "Dijkstra")
        assert app._stack[1] == ("Graphs", "DFS")


@pytest.mark.asyncio
async def test_selection_list_empty_selection_skips(test_db):
    """Confirming with nothing selected calls _process_next (skip)."""
    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        app._current_concept = "Graphs"
        app._current_keywords = ["DFS", "BFS"]
        app._state = "waiting_selection"

        with patch.object(app, "_process_next") as mock_next:
            app._show_keyword_selector(["DFS", "BFS"])
            await pilot.pause()

            selector = app.query_one(".keyword-selector", KeywordSelector)
            selector.focus()
            await pilot.pause()
            app.action_confirm_selection()

        assert len(app._stack) == 0
        mock_next.assert_called_once()


@pytest.mark.asyncio
async def test_selection_list_custom_concept(test_db):
    """Selecting '+ Custom concept...' switches to waiting_custom state."""
    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        app._current_concept = "Graphs"
        app._current_keywords = ["DFS", "BFS"]
        app._state = "waiting_selection"

        app._show_keyword_selector(["DFS", "BFS"])
        await pilot.pause()

        selector = app.query_one(".keyword-selector", KeywordSelector)
        # Toggle the custom sentinel by its value
        selector.toggle(app._custom_concept_sentinel)
        await pilot.pause()

        selector.focus()
        await pilot.pause()
        app.action_confirm_selection()
        # Allow the async remove() to complete
        await pilot.pause()

        assert app._state == "waiting_custom"
        # The selector should have been removed from the DOM
        assert len(app.query(".keyword-selector")) == 0


@pytest.mark.asyncio
async def test_selection_list_custom_with_keywords(test_db):
    """Selecting keywords AND custom sentinel pushes keywords then enters custom mode."""
    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        app._current_concept = "Graphs"
        app._current_keywords = ["DFS", "BFS"]
        app._state = "waiting_selection"

        app._show_keyword_selector(["DFS", "BFS"])
        await pilot.pause()

        selector = app.query_one(".keyword-selector", KeywordSelector)
        # Toggle DFS and custom sentinel by value
        selector.toggle("DFS")
        selector.toggle(app._custom_concept_sentinel)
        await pilot.pause()

        selector.focus()
        await pilot.pause()
        app.action_confirm_selection()

        # DFS should be pushed to stack
        assert len(app._stack) == 1
        assert app._stack[0] == ("Graphs", "DFS")
        # And state should be waiting_custom
        assert app._state == "waiting_custom"


# ── GraphScreen tests ──


@pytest.mark.asyncio
async def test_graph_screen_single_session_with_data(test_db):
    """GraphScreen shows tree for a single session with nodes and edges."""
    sid = create_session("Test Session")
    insert_node(None, "Root", "Root explanation", session_id=sid)
    insert_node("Root", "Child1", "Child1 explanation", session_id=sid)
    insert_node("Root", "Child2", "Child2 explanation", session_id=sid)

    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        screen = GraphScreen(session_id=sid)
        app.push_screen(screen)
        await pilot.pause()

        tree = screen.query_one("#graph-tree", Tree)
        assert tree is not None
        # The tree should have content (root node with children)
        # Root should be hidden (show_root = False), but
        # there should be at least one branch node under root
        assert len(tree.root.children) > 0


@pytest.mark.asyncio
async def test_graph_screen_single_session_empty(test_db):
    """GraphScreen shows session name for a session with no edges yet."""
    sid = create_session("Empty Session")

    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        screen = GraphScreen(session_id=sid)
        app.push_screen(screen)
        await pilot.pause()

        tree = screen.query_one("#graph-tree", Tree)
        # With no edges, should show the session name as a leaf
        assert len(tree.root.children) == 1
        assert "Empty Session" in str(tree.root.children[0].label)


@pytest.mark.asyncio
async def test_graph_screen_all_sessions(test_db):
    """GraphScreen with session_id=None shows all sessions."""
    sid1 = create_session("Session A")
    insert_node(None, "Alpha", "Alpha explanation", session_id=sid1)
    insert_node("Alpha", "Beta", "Beta explanation", session_id=sid1)

    sid2 = create_session("Session B")
    insert_node(None, "Gamma", "Gamma explanation", session_id=sid2)

    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        screen = GraphScreen(session_id=None)
        app.push_screen(screen)
        await pilot.pause()

        tree = screen.query_one("#graph-tree", Tree)
        # Should have one top-level branch per session
        assert len(tree.root.children) == 2


@pytest.mark.asyncio
async def test_graph_screen_all_sessions_empty(test_db):
    """GraphScreen with no sessions shows 'No data yet'."""
    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        screen = GraphScreen(session_id=None)
        app.push_screen(screen)
        await pilot.pause()

        tree = screen.query_one("#graph-tree", Tree)
        assert len(tree.root.children) == 1
        assert "No data yet" in str(tree.root.children[0].label)


@pytest.mark.asyncio
async def test_graph_screen_dismiss_on_escape(test_db):
    """GraphScreen is dismissed when Escape is pressed."""
    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        app.push_screen(GraphScreen(session_id=None))
        await pilot.pause()

        # Should be on the GraphScreen
        assert isinstance(app.screen, GraphScreen)

        await pilot.press("escape")
        await pilot.pause()

        # Should be back on the main screen
        assert not isinstance(app.screen, GraphScreen)


@pytest.mark.asyncio
async def test_graph_command_no_session(test_db):
    """/graph without active session logs a warning."""
    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        app._session_id = None
        app._state = "waiting_initial"

        app._handle_command("/graph")
        await pilot.pause()

        # Should NOT push a GraphScreen since there's no active session
        assert not isinstance(app.screen, GraphScreen)


# ── Graph sidebar tests ──


@pytest.mark.asyncio
async def test_sidebar_hidden_by_default(test_db):
    """The graph sidebar is hidden when the app starts."""
    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        sidebar = app.query_one("#graph-sidebar")
        assert not sidebar.has_class("visible")


@pytest.mark.asyncio
async def test_sidebar_toggle_via_ctrl_g(test_db):
    """Ctrl+G toggles the graph sidebar visibility."""
    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        sidebar = app.query_one("#graph-sidebar")
        assert not sidebar.has_class("visible")

        # Toggle on
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert sidebar.has_class("visible")

        # Toggle off
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert not sidebar.has_class("visible")


@pytest.mark.asyncio
async def test_sidebar_shows_session_data(test_db):
    """When sidebar is toggled on with an active session, it shows nodes."""
    sid = create_session("Sidebar Test")
    insert_node(None, "Root", "Root explanation", session_id=sid)
    insert_node("Root", "Child", "Child explanation", session_id=sid)

    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        app._session_id = sid

        await pilot.press("ctrl+g")
        await pilot.pause()

        tree = app.query_one("#sidebar-tree", Tree)
        # Should have at least one root branch
        assert len(tree.root.children) > 0


@pytest.mark.asyncio
async def test_sidebar_no_session_shows_message(test_db):
    """When sidebar is toggled on without an active session, shows message."""
    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        app._session_id = None

        await pilot.press("ctrl+g")
        await pilot.pause()

        tree = app.query_one("#sidebar-tree", Tree)
        assert len(tree.root.children) == 1
        assert "No active session" in str(tree.root.children[0].label)


# ── Tree styling tests: pending, current, partial ──


@pytest.mark.asyncio
async def test_sidebar_shows_pending_concepts(test_db):
    """Pending concepts on the stack appear as dim nodes in the sidebar tree."""
    sid = create_session("Pending Test")
    insert_node(None, "Root", "Root explanation", session_id=sid)
    insert_node("Root", "ExistingChild", "ExistingChild explanation", session_id=sid)

    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        app._session_id = sid
        app._current_concept = None
        app._current_parent = None
        app._stack = [("Root", "PendingChild")]

        await pilot.press("ctrl+g")
        await pilot.pause()

        tree = app.query_one("#sidebar-tree", Tree)
        # Root should be present with children
        root_node = tree.root.children[0]
        assert "Root" in str(root_node.label)
        pending = [c for c in root_node.children if "PendingChild" in str(c.label)]
        assert len(pending) == 1
        assert _markup(pending[0].label).startswith("[dim]")


@pytest.mark.asyncio
async def test_sidebar_current_concept_indicator(test_db):
    """The currently-being-fetched concept shows a ▶ indicator in the sidebar."""
    sid = create_session("Current Test")
    insert_node(None, "Root", "Root explanation", session_id=sid)
    insert_node("Root", "ExistingChild", "ExistingChild explanation", session_id=sid)

    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        app._session_id = sid
        app._current_concept = "CurrentFetch"
        app._current_parent = "Root"
        app._stack = []

        await pilot.press("ctrl+g")
        await pilot.pause()

        tree = app.query_one("#sidebar-tree", Tree)
        root_node = tree.root.children[0]
        assert "Root" in str(root_node.label)
        # Root should have a child with the ▶ indicator
        current = [c for c in root_node.children if "CurrentFetch" in str(c.label)]
        assert len(current) == 1
        assert "\u25b6" in str(current[0].label)
        assert _markup(current[0].label).startswith("[bold yellow]")


@pytest.mark.asyncio
async def test_sidebar_partial_visited_styling(test_db):
    """A visited node with a pending descendant shows as partially visited (cyan, not bold)."""
    sid = create_session("Partial Test")
    insert_node(None, "Root", "Root explanation", session_id=sid)
    insert_node("Root", "VisitedChild", "VisitedChild explanation", session_id=sid)

    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        app._session_id = sid
        app._current_concept = None
        app._current_parent = None
        app._stack = [("Root", "PendingChild")]

        await pilot.press("ctrl+g")
        await pilot.pause()

        tree = app.query_one("#sidebar-tree", Tree)
        root_node = tree.root.children[0]
        root_markup = _markup(root_node.label)
        # Root is partially visited: [cyan] but NOT [bold cyan]
        assert root_markup.startswith("[cyan]")
        assert "Root" in root_markup
        # VisitedChild should be fully visited (bold cyan, leaf)
        visited = [c for c in root_node.children if "VisitedChild" in str(c.label)]
        assert len(visited) == 1
        assert _markup(visited[0].label).startswith("[bold cyan]")
        # PendingChild should be dim
        pending = [c for c in root_node.children if "PendingChild" in str(c.label)]
        assert len(pending) == 1
        assert _markup(pending[0].label).startswith("[dim]")


@pytest.mark.asyncio
async def test_graph_screen_with_pending_stack(test_db):
    """GraphScreen renders pending stack concepts as dim nodes."""
    sid = create_session("Graph Pending")
    insert_node(None, "Root", "Root explanation", session_id=sid)
    insert_node("Root", "ExistingChild", "ExistingChild explanation", session_id=sid)

    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        screen = GraphScreen(
            session_id=sid,
            pending_stack=[("Root", "PendingConcept")],
        )
        app.push_screen(screen)
        await pilot.pause()

        tree = screen.query_one("#graph-tree", Tree)
        root_node = tree.root.children[0]
        assert "Root" in str(root_node.label)
        pending = [c for c in root_node.children if "PendingConcept" in str(c.label)]
        assert len(pending) == 1
        assert _markup(pending[0].label).startswith("[dim]")


@pytest.mark.asyncio
async def test_graph_screen_current_concept(test_db):
    """GraphScreen renders the current concept with a ▶ indicator."""
    sid = create_session("Graph Current")
    insert_node(None, "Root", "Root explanation", session_id=sid)
    insert_node("Root", "ExistingChild", "ExistingChild explanation", session_id=sid)

    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        screen = GraphScreen(
            session_id=sid,
            current_concept="ActiveNode",
            current_parent="Root",
        )
        app.push_screen(screen)
        await pilot.pause()

        tree = screen.query_one("#graph-tree", Tree)
        root_node = tree.root.children[0]
        assert "Root" in str(root_node.label)
        current = [c for c in root_node.children if "ActiveNode" in str(c.label)]
        assert len(current) == 1
        assert "\u25b6" in str(current[0].label)
        assert _markup(current[0].label).startswith("[bold yellow]")


# ── Modal event isolation tests (regression) ──


@pytest.mark.asyncio
async def test_model_select_input_does_not_trigger_exploration(test_db, tmp_path):
    """Input.Submitted from ModelSelectScreen must NOT reach the main app handler."""
    # Patch DATA_DIR and AUTH_FILE so save_model writes to a temp directory
    tmp_auth = tmp_path / "auth.json"
    with (
        patch("concept_dfs.provider.CONFIG_DIR", tmp_path),
        patch("concept_dfs.provider.AUTH_FILE", tmp_auth),
    ):
        app = ConceptDFSApp()
        async with app.run_test() as pilot:
            app._state = "initial"

            # Push ModelSelectScreen like /model command does
            app.push_screen(ModelSelectScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectScreen)

            # Select a provider (first one)
            option_list = screen.query_one("#provider-list")
            option_list.action_select()
            await pilot.pause()

            # Now type a model name and submit
            model_input = screen.query_one("#model-input")
            model_input.value = "test-model-123"
            await pilot.press("enter")
            await pilot.pause()

            # The model name must NOT have started exploration
            assert app._state == "initial"
            assert app._session_id is None
            assert len(app._stack) == 0


@pytest.mark.asyncio
async def test_auth_screen_input_does_not_trigger_exploration(test_db, tmp_path):
    """Input.Submitted from AuthScreen must NOT reach the main app handler."""
    # Patch DATA_DIR and AUTH_FILE so save_model writes to a temp directory
    tmp_auth = tmp_path / "auth.json"
    with (
        patch("concept_dfs.provider.CONFIG_DIR", tmp_path),
        patch("concept_dfs.provider.AUTH_FILE", tmp_auth),
    ):
        from concept_dfs.provider import save_model

        # Need a model set so AuthScreen shows the input field
        save_model("openai:gpt-4.1")

        app = ConceptDFSApp()
        async with app.run_test() as pilot:
            app._state = "initial"

            app.push_screen(AuthScreen())
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, AuthScreen)

            auth_input = screen.query_one("#auth-input")
            auth_input.value = "sk-fake-key-12345"
            await pilot.press("enter")
            await pilot.pause()

            # The API key must NOT have started exploration
            assert app._state == "initial"
            assert app._session_id is None
            assert len(app._stack) == 0


# ── Modal screen flow tests ──


@pytest.mark.asyncio
async def test_model_select_screen_dismiss_on_cancel(test_db):
    """Pressing Escape on ModelSelectScreen dismisses it with None."""
    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        results = []
        app.push_screen(ModelSelectScreen(), callback=lambda r: results.append(r))
        await pilot.pause()

        assert isinstance(app.screen, ModelSelectScreen)

        await pilot.press("escape")
        await pilot.pause()

        assert not isinstance(app.screen, ModelSelectScreen)
        assert results == [None]


@pytest.mark.asyncio
async def test_model_select_screen_returns_model_string(test_db, tmp_path):
    """ModelSelectScreen returns 'provider:model' on successful selection."""
    # Patch DATA_DIR and AUTH_FILE so save_model writes to a temp directory
    tmp_auth = tmp_path / "auth.json"
    with (
        patch("concept_dfs.provider.CONFIG_DIR", tmp_path),
        patch("concept_dfs.provider.AUTH_FILE", tmp_auth),
    ):
        app = ConceptDFSApp()
        async with app.run_test() as pilot:
            results = []
            app.push_screen(ModelSelectScreen(), callback=lambda r: results.append(r))
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, ModelSelectScreen)

            # Select first provider
            option_list = screen.query_one("#provider-list")
            option_list.action_select()
            await pilot.pause()

            # Accept the default model
            await pilot.press("enter")
            await pilot.pause()

            assert not isinstance(app.screen, ModelSelectScreen)
            assert len(results) == 1
            assert results[0] is not None
            # Should be "provider_id:model_name" format
            assert ":" in results[0]


@pytest.mark.asyncio
async def test_auth_screen_dismiss_on_cancel(test_db, tmp_path):
    """Pressing Escape on AuthScreen dismisses it with None."""
    tmp_auth = tmp_path / "auth.json"
    with (
        patch("concept_dfs.provider.CONFIG_DIR", tmp_path),
        patch("concept_dfs.provider.AUTH_FILE", tmp_auth),
    ):
        from concept_dfs.provider import save_model

        save_model("openai:gpt-4.1")

        app = ConceptDFSApp()
        async with app.run_test() as pilot:
            results = []
            app.push_screen(AuthScreen(), callback=lambda r: results.append(r))
            await pilot.pause()

            assert isinstance(app.screen, AuthScreen)

            await pilot.press("escape")
            await pilot.pause()

            assert not isinstance(app.screen, AuthScreen)
            assert results == [None]


@pytest.mark.asyncio
async def test_session_list_screen_dismiss_on_cancel(test_db):
    """Pressing Escape on SessionListScreen dismisses it with None."""
    app = ConceptDFSApp()
    async with app.run_test() as pilot:
        results = []
        app.push_screen(SessionListScreen(), callback=lambda r: results.append(r))
        await pilot.pause()

        assert isinstance(app.screen, SessionListScreen)

        await pilot.press("escape")
        await pilot.pause()

        assert not isinstance(app.screen, SessionListScreen)
        assert results == [None]
