"""Textual TUI application for ConceptDFS."""

import os
from typing import Optional, List, Tuple

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    OptionList,
    Rule,
    SelectionList,
    Static,
    Tree,
)
from textual.widgets.option_list import Option
from textual.widgets.selection_list import Selection

from concept_dfs.db import (
    create_session,
    get_all_edges,
    get_all_nodes,
    get_node,
    get_path_to_root,
    get_session,
    get_session_edges,
    get_session_nodes,
    get_session_root,
    init_db,
    insert_node,
    list_sessions,
)
from concept_dfs.llm import (
    ConceptResponse,
    fetch_concept,
    fetch_keywords,
    parse_keywords_from_text,
    stream_explanation,
)
from concept_dfs.provider import (
    PROVIDERS,
    get_api_key,
    get_saved_model,
    resolve_model,
    save_api_key,
    save_model,
)


class KeywordSelector(SelectionList[str]):
    """SelectionList subclass that handles Enter internally."""

    class Confirmed(Message):
        """Posted when the user presses Enter to confirm selection."""

    BINDINGS = [Binding("enter", "confirm_kw", "Confirm", show=False)]

    def action_confirm_kw(self) -> None:
        """Confirm the current selection."""
        self.post_message(self.Confirmed())


def build_report(session_id: Optional[int] = None) -> Optional[str]:
    """Build the markdown report content. Returns None if no data.

    If session_id is provided, only includes nodes and edges from that session.
    Otherwise includes all data.
    """
    nodes = get_all_nodes(session_id=session_id)
    edges = get_all_edges(session_id=session_id)

    if not nodes:
        return None

    session_info = ""
    if session_id is not None:
        session = get_session(session_id)
        if session:
            session_info = (
                f"\n**Session:** {session['name']}  \n"
                f"**Date:** {session['created_at']}\n"
            )

    lines = ["# ConceptDFS Exploration Report\n"]
    if session_info:
        lines.append(session_info)

    lines.append("## Knowledge Graph\n")
    lines.append("```mermaid")
    lines.append("graph TD;")

    id_to_concept = {n["id"]: n["concept"] for n in nodes}

    for edge in edges:
        parent = id_to_concept.get(edge["parent_id"])
        child = id_to_concept.get(edge["child_id"])
        if parent and child:
            p_safe = parent.replace('"', "")
            c_safe = child.replace('"', "")
            lines.append(
                f'  {edge["parent_id"]}["{p_safe}"] --> {edge["child_id"]}["{c_safe}"]'
            )

    lines.append("```\n")

    lines.append("## Concepts Explained\n")
    for n in nodes:
        lines.append(f"### {n['concept']}\n")
        lines.append(n["explanation"])
        lines.append("\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tree builder (shared by sidebar and GraphScreen)
# ---------------------------------------------------------------------------


def _build_combined_concept_tree(
    tree_root,
    db_nodes: List[dict],
    db_edges: List[dict],
    current_concept: Optional[str] = None,
    current_parent: Optional[str] = None,
    pending_stack: Optional[List[Tuple[Optional[str], str]]] = None,
    children_of: Optional[str] = None,
) -> None:
    """Build a concept tree combining DB data with pending/current state.

    Node styles:
        Fully visited (in DB, all descendants visited): [bold cyan]
        Partially visited (in DB, has unvisited descendants): [cyan]
        Current (being fetched): [bold yellow]▶
        Pending (on stack, not yet fetched): [dim]

    Args:
        tree_root: Tree node to add children to.
        db_nodes: Nodes from the database.
        db_edges: Edges from the database.
        current_concept: Concept currently being fetched (or None).
        current_parent: Parent of the current concept (or None).
        pending_stack: List of (parent, child) tuples on the DFS stack.
        children_of: If set, only add children of this concept (not itself).
    """
    visited_concepts: set[str] = {n["concept"] for n in db_nodes}
    id_to_concept = {n["id"]: n["concept"] for n in db_nodes}

    # Build name-based children map (parent_name -> [child_names])
    children_map: dict[Optional[str], list[str]] = {}

    for e in db_edges:
        parent_name = id_to_concept.get(e["parent_id"])
        child_name = id_to_concept.get(e["child_id"])
        if parent_name is not None and child_name is not None:
            children_map.setdefault(parent_name, [])
            if child_name not in children_map[parent_name]:
                children_map[parent_name].append(child_name)

    # Add current concept (if not already visited)
    if current_concept and current_concept not in visited_concepts:
        children_map.setdefault(current_parent, [])
        if current_concept not in children_map[current_parent]:
            children_map[current_parent].append(current_concept)

    # Add pending stack items (if not already visited or current)
    if pending_stack:
        for parent, child in pending_stack:
            if child not in visited_concepts and child != current_concept:
                children_map.setdefault(parent, [])
                if child not in children_map[parent]:
                    children_map[parent].append(child)

    # Collect all known concepts (including parents referenced by stack/current)
    all_concepts: set[str] = set(visited_concepts)
    if current_concept:
        all_concepts.add(current_concept)
    if current_parent:
        all_concepts.add(current_parent)
    if pending_stack:
        for parent, child in pending_stack:
            all_concepts.add(child)
            if parent is not None:
                all_concepts.add(parent)

    # Determine starting concepts
    if children_of is not None:
        start_concepts = list(children_map.get(children_of, []))
    else:
        # Root concepts: parent=None or never a child of any named parent
        all_children_set: set[str] = set()
        for key, ch_list in children_map.items():
            if key is not None:
                all_children_set.update(ch_list)

        explicit_roots = list(children_map.get(None, []))
        implicit_roots = [
            c
            for c in all_concepts
            if c not in all_children_set and c not in explicit_roots
        ]
        start_concepts = explicit_roots + implicit_roots

    if not start_concepts:
        if children_of is None:
            tree_root.add_leaf("[dim]No data yet.[/dim]")
        return

    # Memoized full-visit check
    _memo: dict[str, bool] = {}

    def is_fully_visited(concept: str) -> bool:
        if concept in _memo:
            return _memo[concept]
        _memo[concept] = False  # guard against cycles
        if concept not in visited_concepts or concept == current_concept:
            return False
        children = children_map.get(concept, [])
        result = all(is_fully_visited(c) for c in children)
        _memo[concept] = result
        return result

    def concept_label(concept: str) -> str:
        if concept == current_concept and concept not in visited_concepts:
            return f"[bold yellow]\u25b6 {concept}[/bold yellow]"
        if concept not in visited_concepts:
            return f"[dim]{concept}[/dim]"
        if is_fully_visited(concept):
            return f"[bold cyan]{concept}[/bold cyan]"
        return f"[cyan]{concept}[/cyan]"

    def add_subtree(parent_node, concept: str, seen: set) -> None:
        if concept in seen:
            return
        seen.add(concept)
        children = children_map.get(concept, [])
        label = concept_label(concept)
        if children:
            branch = parent_node.add(label)
            for child in children:
                add_subtree(branch, child, seen)
        else:
            parent_node.add_leaf(label)

    seen: set[str] = set()
    for concept in start_concepts:
        add_subtree(tree_root, concept, seen)


# ---------------------------------------------------------------------------
# Modal screens
# ---------------------------------------------------------------------------


class ModelSelectScreen(ModalScreen[Optional[str]]):
    """Modal screen for selecting LLM provider and model."""

    DEFAULT_CSS = """
    ModelSelectScreen {
        align: center middle;
        background: $background 60%;
    }

    #model-dialog {
        width: 64;
        height: auto;
        max-height: 80%;
        border: round $primary-lighten-2;
        border-title-color: $text;
        border-title-style: bold;
        padding: 1 2;
        background: $surface;
    }

    #provider-list {
        height: auto;
        max-height: 12;
        margin: 1 0 0 0;
    }

    #model-input {
        margin: 1 0 0 0;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self._selected_provider: Optional[str] = None

    def compose(self) -> ComposeResult:
        with Vertical(id="model-dialog") as dialog:
            dialog.border_title = "Select Provider"
            yield OptionList(
                *[Option(info.name) for info in PROVIDERS.values()],
                id="provider-list",
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        provider_ids = list(PROVIDERS.keys())
        self._selected_provider = provider_ids[event.option_index]
        info = PROVIDERS[self._selected_provider]

        dialog = self.query_one("#model-dialog")
        self.query_one("#provider-list").remove()

        dialog.border_title = f"Model for {info.name}"
        dialog.mount(Label(f"[dim]Default: {info.default_model}[/dim]"))
        dialog.mount(Input(value=info.default_model, id="model-input"))
        self.set_focus(self.query_one("#model-input"))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._selected_provider and event.input.id == "model-input":
            event.stop()
            model = (
                event.value.strip() or PROVIDERS[self._selected_provider].default_model
            )
            model_str = f"{self._selected_provider}:{model}"
            save_model(model_str)
            self.dismiss(model_str)

    def action_cancel(self) -> None:
        self.dismiss(None)


class AuthScreen(ModalScreen[Optional[str]]):
    """Modal screen for entering an API key."""

    DEFAULT_CSS = """
    AuthScreen {
        align: center middle;
        background: $background 60%;
    }

    #auth-dialog {
        width: 64;
        height: auto;
        border: round $primary-lighten-2;
        border-title-color: $text;
        border-title-style: bold;
        padding: 1 2;
        background: $surface;
    }

    #auth-input {
        margin: 1 0 0 0;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self._provider_id: Optional[str] = None

    def compose(self) -> ComposeResult:
        model_str = os.environ.get("CONCEPT_DFS_MODEL") or get_saved_model()
        if model_str and ":" in model_str:
            self._provider_id = model_str.split(":")[0]

        with Vertical(id="auth-dialog") as dialog:
            if self._provider_id and self._provider_id in PROVIDERS:
                info = PROVIDERS[self._provider_id]
                dialog.border_title = f"API Key \u2014 {info.name}"
                yield Label(f"[dim]Environment variable: {info.env_key}[/dim]")
                yield Input(
                    placeholder="Enter your API key...",
                    password=True,
                    id="auth-input",
                )
            else:
                dialog.border_title = "API Key"
                yield Label("[yellow]No provider selected. Use /model first.[/yellow]")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._provider_id and event.input.id == "auth-input":
            event.stop()
            key = event.value.strip()
            if key:
                save_api_key(self._provider_id, key)
                info = PROVIDERS[self._provider_id]
                os.environ[info.env_key] = key
                self.dismiss(key)
            else:
                self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SessionListScreen(ModalScreen[Optional[int]]):
    """Modal screen for listing and selecting a session to resume."""

    DEFAULT_CSS = """
    SessionListScreen {
        align: center middle;
        background: $background 60%;
    }

    #session-dialog {
        width: 72;
        height: auto;
        max-height: 80%;
        border: round $primary-lighten-2;
        border-title-color: $text;
        border-title-style: bold;
        padding: 1 2;
        background: $surface;
    }

    #session-list {
        height: auto;
        max-height: 20;
        margin: 1 0 0 0;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self._sessions: List = []

    def compose(self) -> ComposeResult:
        self._sessions = list_sessions()
        with Vertical(id="session-dialog") as dialog:
            dialog.border_title = "Sessions"
            if not self._sessions:
                yield Label("[dim]No sessions found.[/dim]")
            else:
                options = []
                for s in self._sessions:
                    root_node = get_session_root(s["id"])
                    root_label = root_node["concept"] if root_node else s["name"]
                    options.append(
                        Option(
                            f"#{s['id']}  {root_label}  [dim]({s['created_at']})[/dim]"
                        )
                    )
                yield OptionList(*options, id="session-list")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if self._sessions:
            session = self._sessions[event.option_index]
            self.dismiss(session["id"])

    def action_cancel(self) -> None:
        self.dismiss(None)


class GraphScreen(ModalScreen[None]):
    """Modal screen showing the exploration graph as an interactive tree."""

    DEFAULT_CSS = """
    GraphScreen {
        align: center middle;
        background: $background 60%;
    }

    #graph-dialog {
        width: 80;
        height: 80%;
        border: round $primary-lighten-2;
        border-title-color: $text;
        border-title-style: bold;
        padding: 1 2;
        background: $surface;
    }

    #graph-tree {
        height: 1fr;
    }
    """

    BINDINGS = [("escape", "cancel", "Close")]

    def __init__(
        self,
        session_id: Optional[int] = None,
        current_concept: Optional[str] = None,
        current_parent: Optional[str] = None,
        pending_stack: Optional[List[Tuple[Optional[str], str]]] = None,
    ) -> None:
        super().__init__()
        self._session_id = session_id
        self._current_concept = current_concept
        self._current_parent = current_parent
        self._pending_stack = pending_stack

    def compose(self) -> ComposeResult:
        with Vertical(id="graph-dialog") as dialog:
            if self._session_id is not None:
                session = get_session(self._session_id)
                name = session["name"] if session else f"#{self._session_id}"
                dialog.border_title = f"Graph \u2014 {name}"
            else:
                dialog.border_title = "Graph \u2014 All Sessions"

            tree: Tree[str] = Tree("ConceptDFS", id="graph-tree")
            tree.show_root = False

            if self._session_id is not None:
                nodes = get_session_nodes(self._session_id)
                edges = get_session_edges(self._session_id)

                if not nodes and not self._current_concept and not self._pending_stack:
                    session = get_session(self._session_id)
                    if session:
                        tree.root.add_leaf(f"[bold cyan]{session['name']}[/bold cyan]")
                    else:
                        tree.root.add_leaf("[dim]No data yet.[/dim]")
                else:
                    _build_combined_concept_tree(
                        tree.root,
                        nodes,
                        edges,
                        current_concept=self._current_concept,
                        current_parent=self._current_parent,
                        pending_stack=self._pending_stack,
                    )
            else:
                # Show all sessions as top-level branches
                sessions = list_sessions()
                if sessions:
                    for s in sessions:
                        nodes = get_session_nodes(s["id"])
                        edges = get_session_edges(s["id"])
                        root_node = get_session_root(s["id"])
                        label = root_node["concept"] if root_node else s["name"]
                        branch = tree.root.add(f"[bold cyan]{label}[/bold cyan]")
                        _build_combined_concept_tree(
                            branch, nodes, edges, children_of=label
                        )
                else:
                    tree.root.add_leaf("[dim]No data yet.[/dim]")

            tree.root.expand_all()
            yield tree

    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class ConceptDFSApp(App):
    """Textual app for exploring concepts via DFS."""

    TITLE = "ConceptDFS"

    CSS = """
    Screen {
        layout: vertical;
    }

    #main-area {
        height: 1fr;
    }

    #log {
        height: 1fr;
        padding: 1 2;
        margin: 0 1;
        border: round $primary-background-lighten-1;
        scrollbar-gutter: stable;
        scrollbar-size-vertical: 1;
    }

    .keyword-selector {
        height: auto;
        max-height: 12;
        margin: 0 0 1 0;
        border: round $accent;
        border-title-color: $text;
        border-title-style: bold;
    }

    #graph-sidebar {
        width: 36;
        height: 1fr;
        margin: 0 1 0 0;
        padding: 1;
        border: round $secondary-background-lighten-1;
        display: none;
    }

    #graph-sidebar.visible {
        display: block;
    }

    #sidebar-tree {
        height: 1fr;
    }

    #prompt {
        dock: bottom;
        margin: 1 1;
        border: round $accent;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+e", "export", "Export", priority=True),
        Binding("ctrl+g", "graph", "Graph Sidebar", priority=True),
    ]

    def __init__(self, initial_concept: Optional[str] = None) -> None:
        super().__init__()
        self._initial_concept = initial_concept
        self._stack: List[Tuple[Optional[str], str]] = []
        self._current_keywords: List[str] = []
        self._current_parent: Optional[str] = None
        self._current_concept: Optional[str] = None
        self._session_id: Optional[int] = None
        self._state = "initial"
        self._stream_widget_id: int = 0
        self._selector_id: int = 0
        self._custom_concept_sentinel = "__custom__"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-area"):
            with VerticalScroll(id="log"):
                yield Static(
                    "[bold cyan]Welcome to ConceptDFS[/bold cyan]\n\n"
                    "Explore branching knowledge via DFS over an "
                    "LLM-generated concept graph.\n\n"
                    "[dim]Type /help to see available commands.[/dim]",
                    id="welcome",
                )
            with Vertical(id="graph-sidebar"):
                sidebar_tree: Tree[str] = Tree("Graph", id="sidebar-tree")
                sidebar_tree.show_root = False
                sidebar_tree.root.add_leaf("[dim]No data yet.[/dim]")
                yield sidebar_tree
        yield Input(
            placeholder="Enter a concept to explore...",
            id="prompt",
        )
        yield Footer()

    def on_mount(self) -> None:
        init_db()
        self._refresh_subtitle()
        if self._initial_concept:
            self._start_exploration(self._initial_concept)

    def _refresh_subtitle(self) -> None:
        """Update the header subtitle with the current model info."""
        model_str = os.environ.get("CONCEPT_DFS_MODEL") or get_saved_model()
        if model_str:
            self.sub_title = model_str
        else:
            self.sub_title = "no model selected"

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "prompt":
            return
        value = event.value.strip()
        event.input.clear()

        if value.startswith("/"):
            self._handle_command(value)
            return

        if self._state == "initial":
            if not value:
                self._log("[red]Please enter a concept to get started.[/red]")
                return
            self._start_exploration(value)

        elif self._state == "waiting_custom":
            # User typed a custom concept name in the Input
            if not value:
                # Empty input — cancel custom, re-focus the selector
                self._state = "waiting_selection"
                try:
                    selector = self.query_one(".keyword-selector", KeywordSelector)
                    selector.focus()
                    self._set_placeholder("")
                except Exception:
                    self._process_next()
                return
            if self._current_concept is not None:
                self._stack.append((self._current_concept, value))
            else:
                self._stack.append((None, value))
            self._log(f"[bold cyan]Added custom concept:[/bold cyan] {value}")
            self._remove_active_selector()
            self._process_next()

        elif self._state == "waiting_cache":
            if not value:
                self._process_next()

        elif self._state == "waiting_error":
            if not value:
                # Retry — _current_concept is always set when in error state
                if self._current_concept is not None:
                    self._stack.append((self._current_parent, self._current_concept))
                self._process_next()
            elif value.lower() == "skip":
                self._process_next()
            else:
                self._log(
                    "[red]Press Enter to retry, 'skip' to skip, "
                    "or use a /command.[/red]"
                )

    # ------------------------------------------------------------------
    # Exploration logic
    # ------------------------------------------------------------------

    def _start_exploration(self, concept: str) -> None:
        if self._session_id is None:
            # New exploration — create a new session
            self._session_id = create_session(concept)
            self._log(
                f"\n[bold blue]Starting exploration for: {concept}[/bold blue]  "
                f"[dim](session #{self._session_id})[/dim]\n"
            )
        else:
            # Continuing within a resumed session
            self._log(
                f"\n[bold blue]Exploring: {concept}[/bold blue]  "
                f"[dim](session #{self._session_id})[/dim]\n"
            )
        self._stack = [(None, concept)]
        self._maybe_refresh_sidebar()
        self._process_next()

    def _process_next(self) -> None:
        if not self._stack:
            self._current_concept = None
            self._current_parent = None
            self._state = "initial"
            session_note = ""
            if self._session_id is not None:
                session_note = f"  [dim](session #{self._session_id})[/dim]"
            self._log(
                f"\n[bold green]Exploration complete! Stack is empty.[/bold green]{session_note}"
            )
            self._log(
                "[dim]Use /export or Ctrl+E to export the report. "
                "Enter a new concept to start a new session, "
                "or /resume to continue a previous one.[/dim]"
            )
            self._maybe_refresh_sidebar()
            self._session_id = None
            self._set_placeholder("Enter a concept to explore, or /command...")
            return

        parent, concept = self._stack.pop()
        self._current_parent = parent
        self._current_concept = concept
        self._maybe_refresh_sidebar()

        log = self.query_one("#log")
        log.mount(Rule())
        log.mount(Static(f"[bold magenta]Exploring: {concept}[/bold magenta]"))

        # Check cache first
        node = get_node(concept)
        if node:
            explanation = node["explanation"]
            cached_keywords: List[str] = node.get("keywords") or []
            insert_node(
                parent,
                concept,
                explanation,
                session_id=self._session_id,
                keywords=cached_keywords,
            )
            self._maybe_refresh_sidebar()
            log.mount(Static("[dim italic]Loaded from local cache.[/dim italic]"))
            log.mount(Markdown(explanation))

            if cached_keywords:
                self._current_keywords = cached_keywords
                self._show_keyword_selector(cached_keywords)
                self._state = "waiting_selection"
            else:
                self._state = "waiting_cache"
                self._set_placeholder("Press Enter to continue stack, or /command...")
            self._scroll_down()
            return

        # Fetch from LLM — but first ensure model + API key are available
        self._state = "exploring"
        self._ensure_ready_then_fetch(concept, parent)

    def _ensure_ready_then_fetch(self, concept: str, parent: Optional[str]) -> None:
        """Check that a model is selected and API key is present before fetching.

        Shows modal screens for model/auth if needed, then starts the fetch.
        """
        # 1. Check model
        model_str = os.environ.get("CONCEPT_DFS_MODEL") or get_saved_model()
        if not model_str:
            self._log("[yellow]No model configured. Please select one.[/yellow]")
            self.push_screen(
                ModelSelectScreen(),
                lambda result: self._on_preflight_model_selected(
                    result, concept, parent
                ),
            )
            return

        # 2. Check API key
        provider_id = model_str.split(":")[0] if ":" in model_str else model_str
        key = get_api_key(provider_id)
        if not key:
            info = PROVIDERS.get(provider_id)
            name = info.name if info else provider_id
            self._log(
                f"[yellow]No API key found for {name}. Please enter one.[/yellow]"
            )
            self.push_screen(
                AuthScreen(),
                lambda result: self._on_preflight_auth_done(result, concept, parent),
            )
            return

        # 3. Both ready — start fetch
        log = self.query_one("#log")
        log.mount(Static("[yellow]Fetching explanation from LLM...[/yellow]"))
        self._set_placeholder("Fetching...")
        self.query_one("#prompt", Input).disabled = True
        self._scroll_down()
        self._fetch_concept(concept, parent)

    def _on_preflight_model_selected(
        self, model_str: Optional[str], concept: str, parent: Optional[str]
    ) -> None:
        if model_str:
            self._log(f"[bold green]Model set to {model_str}[/bold green]")
            self._refresh_subtitle()
            # Model set — now re-check (API key might still be missing)
            self._ensure_ready_then_fetch(concept, parent)
        else:
            # User cancelled — abort this fetch, go back to waiting
            self._state = "initial"
            self._log("[dim]Model selection cancelled.[/dim]")
            self._set_placeholder("Enter a concept to explore, or /command...")
            self.query_one("#prompt", Input).focus()

    def _on_preflight_auth_done(
        self, key: Optional[str], concept: str, parent: Optional[str]
    ) -> None:
        if key:
            self._log("[bold green]API key saved.[/bold green]")
            # Key set — proceed with fetch
            self._ensure_ready_then_fetch(concept, parent)
        else:
            # User cancelled
            self._state = "initial"
            self._log("[dim]Auth cancelled.[/dim]")
            self._set_placeholder("Enter a concept to explore, or /command...")
            self.query_one("#prompt", Input).focus()

    @work(thread=True, exclusive=True)
    def _fetch_concept(self, concept: str, parent: Optional[str]) -> None:
        try:
            history: Optional[List[Tuple[str, str]]] = None
            if parent:
                path = get_path_to_root(parent, session_id=self._session_id)
                history = [(n["concept"], n["explanation"]) for n in path]

            # Mount an empty Markdown widget for streaming into
            self._stream_widget_id += 1
            widget_id = f"stream-md-{self._stream_widget_id}"
            self.call_from_thread(self._mount_stream_widget, widget_id)

            # Stream explanation + inline keywords in a single LLM call
            accumulated = ""
            for token in stream_explanation(concept, history=history):
                accumulated += token
                self.call_from_thread(
                    self._update_stream_widget, widget_id, accumulated
                )

            if not accumulated:
                # Fallback: no tokens received, try non-streaming
                response = fetch_concept(concept, history=history)
                explanation = response.explanation
                keywords = response.keywords or []
            else:
                # Parse inline KEYWORDS: line from the streamed text
                explanation, keywords = parse_keywords_from_text(accumulated)

                if not keywords:
                    # Fallback: LLM didn't include KEYWORDS line, fetch separately
                    self.call_from_thread(
                        self._log, "[dim]Fetching related concepts...[/dim]"
                    )
                    keywords = fetch_keywords(concept, explanation) or []

            # Update the stream widget to show only the explanation (without
            # the raw KEYWORDS: line)
            self.call_from_thread(self._update_stream_widget, widget_id, explanation)

            final = ConceptResponse(explanation=explanation, keywords=keywords)
            self.call_from_thread(self._on_fetch_complete, parent, concept, final)
        except Exception as e:
            self.call_from_thread(self._on_fetch_error, parent, concept, str(e))

    def _mount_stream_widget(self, widget_id: str) -> None:
        """Mount an empty Markdown widget that will be updated as tokens stream in."""
        log = self.query_one("#log")
        log.mount(Markdown("", id=widget_id))
        self._scroll_down()

    def _update_stream_widget(self, widget_id: str, content: str) -> None:
        """Update the streaming Markdown widget with new content."""
        try:
            md = self.query_one(f"#{widget_id}", Markdown)
            md.update(content)
            self._scroll_down()
        except Exception:
            pass  # Widget may have been removed

    def _on_fetch_complete(self, parent, concept, response) -> None:
        """Called when streaming is done. Save to DB and show keywords."""
        explanation = response.explanation
        keywords = response.keywords
        insert_node(
            parent,
            concept,
            explanation,
            session_id=self._session_id,
            keywords=keywords,
        )
        self._maybe_refresh_sidebar()

        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()

        if keywords:
            self._current_keywords = keywords
            self._show_keyword_selector(keywords)
            self._state = "waiting_selection"
        else:
            self._current_keywords = []
            self._process_next()

        self._scroll_down()

    def _show_keyword_selector(self, keywords: List[str]) -> None:
        """Mount a SelectionList widget with the given keywords."""
        self._selector_id += 1
        selector_id = f"kw-selector-{self._selector_id}"

        selections: list[tuple[str, str]] = []
        for kw in keywords:
            selections.append((kw, kw))
        selections.append(
            (
                "[bold italic]+ Custom concept...[/bold italic]",
                self._custom_concept_sentinel,
            )
        )

        sel_list = KeywordSelector(
            *selections,
            id=selector_id,
            classes="keyword-selector",
        )
        sel_list.border_title = "Select concepts (Space to toggle, Enter to confirm)"

        log = self.query_one("#log")
        log.mount(sel_list)
        sel_list.focus()
        self._set_placeholder("")
        self._scroll_down()

    def _remove_active_selector(self) -> None:
        """Remove the active keyword SelectionList from the DOM."""
        try:
            selector = self.query_one(".keyword-selector", KeywordSelector)
            selector.remove()
        except Exception:
            pass

    def action_confirm_selection(self) -> None:
        """Handle Enter key when it bubbles up from SelectionList."""
        if self._state != "waiting_selection":
            return

        try:
            selector = self.query_one(".keyword-selector", KeywordSelector)
        except Exception:
            return

        selected_values = selector.selected

        if not selected_values:
            # Nothing selected — skip (equivalent to pressing Enter on empty input before)
            self._remove_active_selector()
            self._process_next()
            return

        # Check if custom concept sentinel is among selected
        has_custom = self._custom_concept_sentinel in selected_values
        keyword_values = [
            v for v in selected_values if v != self._custom_concept_sentinel
        ]

        # Push selected keywords onto the stack
        for kw in reversed(keyword_values):
            if self._current_concept is not None:
                self._stack.append((self._current_concept, kw))

        if keyword_values:
            names = ", ".join(keyword_values)
            self._log(f"[bold cyan]Selected:[/bold cyan] {names}")

        if has_custom:
            # Switch to custom input mode
            self._remove_active_selector()
            self._state = "waiting_custom"
            self._set_placeholder(
                "Type a custom concept name and press Enter (empty to cancel)..."
            )
            self.query_one("#prompt", Input).focus()
        else:
            self._remove_active_selector()
            self._process_next()

    def on_keyword_selector_confirmed(self, event: KeywordSelector.Confirmed) -> None:
        """Handle Enter key from KeywordSelector widget."""
        self.action_confirm_selection()

    def _on_fetch_error(self, parent, concept, error) -> None:
        log = self.query_one("#log")
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()

        log.mount(Static(f"[bold red]Error fetching concept: {error}[/bold red]"))
        self._state = "waiting_error"
        self._set_placeholder("Enter to retry, 'skip' to skip, or /command...")
        self._scroll_down()

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    def _handle_command(self, cmd: str) -> None:
        cmd_lower = cmd.lower().strip()

        if cmd_lower == "/help":
            self._log(
                "\n[cyan]Available Commands:[/cyan]\n"
                "  [bold]/sessions[/bold]      - List all exploration sessions\n"
                "  [bold]/resume[/bold]        - Resume a previous session\n"
                "  [bold]/model[/bold]         - Switch LLM provider and model\n"
                "  [bold]/auth[/bold]          - Re-enter API key\n"
                "  [bold]/graph[/bold]         - Show exploration graph for current session\n"
                "  [bold]/graph all[/bold]     - Show exploration graph for all sessions\n"
                "  [bold]/export[/bold]        - Export current session to markdown\n"
                "  [bold]/export all[/bold]    - Export all sessions to markdown\n"
                "  [bold]/help[/bold]          - Show this help\n"
                "  [bold]/exit[/bold]          - Quit\n"
                "\n"
                "  [bold]Ctrl+G[/bold]         - Toggle graph sidebar\n"
                "  [bold]Ctrl+Q[/bold]         - Quit\n"
                "  [bold]Ctrl+E[/bold]         - Export report\n"
                "\n"
                "[dim]At the keyword prompt, use arrow keys to navigate, "
                "Space to toggle, Enter to confirm. "
                "Select '+ Custom concept...' to add your own.[/dim]\n"
            )
        elif cmd_lower == "/sessions":
            self._show_sessions()
        elif cmd_lower == "/resume":
            self.push_screen(SessionListScreen(), self._on_session_selected)
        elif cmd_lower == "/export all":
            self._export_report(session_id=None)
        elif cmd_lower == "/export":
            self._export_report(session_id=self._session_id)
        elif cmd_lower == "/graph all":
            self.push_screen(GraphScreen(session_id=None), self._on_graph_closed)
        elif cmd_lower == "/graph":
            if self._session_id is None:
                self._log(
                    "[yellow]No active session. Use /graph all to view all sessions.[/yellow]"
                )
            else:
                self.push_screen(
                    GraphScreen(
                        session_id=self._session_id,
                        current_concept=self._current_concept,
                        current_parent=self._current_parent,
                        pending_stack=list(self._stack),
                    ),
                    self._on_graph_closed,
                )
        elif cmd_lower in ("/exit", "/quit"):
            self.exit()
        elif cmd_lower == "/model":
            self.push_screen(ModelSelectScreen(), self._on_model_selected)
        elif cmd_lower == "/auth":
            self.push_screen(AuthScreen(), self._on_auth_done)
        elif cmd_lower.startswith("/"):
            self._log(
                f"[red]Unknown command: {cmd_lower}[/red]. Type /help for options."
            )

    def _on_model_selected(self, model_str: Optional[str]) -> None:
        if model_str:
            self._log(f"[bold green]Model set to {model_str}[/bold green]")
            self._refresh_subtitle()
        self.query_one("#prompt", Input).focus()

    def _on_auth_done(self, key: Optional[str]) -> None:
        if key:
            self._log("[bold green]API key saved.[/bold green]")
        self.query_one("#prompt", Input).focus()

    def _on_graph_closed(self, result: None) -> None:
        self.query_one("#prompt", Input).focus()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _show_sessions(self) -> None:
        """Display a list of all sessions in the log."""
        sessions = list_sessions()
        if not sessions:
            self._log("[dim]No sessions found.[/dim]")
            return

        lines = "\n[bold cyan]Exploration Sessions:[/bold cyan]\n"
        for s in sessions:
            root_node = get_session_root(s["id"])
            root_label = root_node["concept"] if root_node else s["name"]
            active = (
                " [bold green]\u25c0 active[/bold green]"
                if s["id"] == self._session_id
                else ""
            )
            lines += (
                f"  [bold green]#{s['id']}[/bold green]  {root_label}  "
                f"[dim]({s['created_at']})[/dim]{active}\n"
            )
        lines += "\n[dim]Use /resume to continue a previous session.[/dim]\n"
        self._log(lines)

    def _on_session_selected(self, session_id: Optional[int]) -> None:
        """Callback when a session is selected from the SessionListScreen."""
        if session_id is None:
            self.query_one("#prompt", Input).focus()
            return

        session = get_session(session_id)
        if not session:
            self._log("[red]Session not found.[/red]")
            self.query_one("#prompt", Input).focus()
            return

        self._session_id = session_id
        root_node = get_session_root(session_id)
        root_label = root_node["concept"] if root_node else session["name"]
        self._current_concept = root_label
        self._state = "initial"

        self._log(
            f"\n[bold green]Resumed session #{session_id}:[/bold green] {root_label}  "
            f"[dim]({session['created_at']})[/dim]\n"
            f"[dim]New explorations will be added to this session. "
            f"Type a concept to continue exploring.[/dim]"
        )
        self._set_placeholder("Enter a concept to add to this session, or /command...")
        self.query_one("#prompt", Input).focus()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def action_export(self) -> None:
        """Ctrl+E binding: exports current session if active, else all."""
        self._export_report(session_id=self._session_id)

    def action_graph(self) -> None:
        """Ctrl+G binding: toggle the graph sidebar panel."""
        sidebar = self.query_one("#graph-sidebar")
        if sidebar.has_class("visible"):
            sidebar.remove_class("visible")
        else:
            self._refresh_graph_sidebar()
            sidebar.add_class("visible")

    def _refresh_graph_sidebar(self) -> None:
        """Rebuild the sidebar tree with visited, current, and pending state."""
        tree = self.query_one("#sidebar-tree", Tree)
        tree.clear()

        sid = self._session_id
        if sid is None:
            tree.root.add_leaf("[dim]No active session.[/dim]")
            return

        nodes = get_session_nodes(sid)
        edges = get_session_edges(sid)

        if not nodes and not self._current_concept and not self._stack:
            session = get_session(sid)
            if session:
                tree.root.add_leaf(f"[bold cyan]{session['name']}[/bold cyan]")
            else:
                tree.root.add_leaf("[dim]No data yet.[/dim]")
            return

        _build_combined_concept_tree(
            tree.root,
            nodes,
            edges,
            current_concept=self._current_concept,
            current_parent=self._current_parent,
            pending_stack=self._stack,
        )
        tree.root.expand_all()

    def _maybe_refresh_sidebar(self) -> None:
        """Refresh the graph sidebar if it is currently visible."""
        sidebar = self.query_one("#graph-sidebar")
        if sidebar.has_class("visible"):
            self._refresh_graph_sidebar()

    def _export_report(self, session_id: Optional[int] = None) -> None:
        content = build_report(session_id=session_id)
        if content is None:
            self._log("[yellow]No concepts found to export.[/yellow]")
            return
        with open("report.md", "w", encoding="utf-8") as f:
            f.write(content)
        scope = f"session #{session_id}" if session_id else "all sessions"
        self._log(f"[bold green]Report exported to report.md ({scope})[/bold green]")

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _log(self, text: str) -> None:
        log = self.query_one("#log")
        log.mount(Static(text))
        self._scroll_down()

    def _set_placeholder(self, text: str) -> None:
        self.query_one("#prompt", Input).placeholder = text

    def _scroll_down(self) -> None:
        self.query_one("#log").scroll_end(animate=False)
