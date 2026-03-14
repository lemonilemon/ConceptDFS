import os
import sqlite3
import pytest
from concept_dfs.db import (
    init_db,
    get_node,
    insert_node,
    get_all_nodes,
    get_all_edges,
    get_path_to_root,
    create_session,
    list_sessions,
    get_session,
    get_session_nodes,
    get_session_edges,
    get_session_root,
    get_db_path,
)


@pytest.fixture(autouse=True)
def test_db(tmp_path):
    db_file = tmp_path / "test_concepts.db"
    os.environ["CONCEPT_DFS_DB"] = str(db_file)
    init_db()
    yield
    if db_file.exists():
        os.remove(db_file)
    os.environ.pop("CONCEPT_DFS_DB", None)


# ---------------------------------------------------------------------------
# Original node/edge tests (backward-compatible, no session_id)
# ---------------------------------------------------------------------------


def test_insert_and_get_node():
    concept = "Test Concept"
    explanation = "This is a test explanation."
    insert_node(None, concept, explanation)

    node = get_node(concept)
    assert node is not None
    assert node["concept"] == concept
    assert node["explanation"] == explanation


def test_insert_duplicate_node():
    concept = "Duplicate"
    explanation1 = "Explanation 1"
    explanation2 = "Explanation 2"

    insert_node(None, concept, explanation1)
    insert_node(None, concept, explanation2)

    node = get_node(concept)
    assert node is not None
    assert node["explanation"] == explanation1  # INSERT OR IGNORE means first one wins


def test_insert_node_with_parent():
    parent = "Parent"
    child = "Child"
    explanation = "explanation"

    insert_node(None, parent, explanation)
    insert_node(parent, child, explanation)

    nodes = get_all_nodes()
    assert len(nodes) == 2

    edges = get_all_edges()
    assert len(edges) == 1

    # Verify IDs match
    parent_id = next(n["id"] for n in nodes if n["concept"] == parent)
    child_id = next(n["id"] for n in nodes if n["concept"] == child)

    assert edges[0]["parent_id"] == parent_id
    assert edges[0]["child_id"] == child_id


def test_get_node_not_found():
    assert get_node("Nonexistent") is None


def test_get_all_nodes_empty():
    assert get_all_nodes() == []


def test_get_all_edges_empty():
    assert get_all_edges() == []


def test_get_path_to_root():
    insert_node(None, "Root", "Root explanation")
    insert_node("Root", "Middle", "Middle explanation")
    insert_node("Middle", "Leaf", "Leaf explanation")

    path = get_path_to_root("Leaf")

    assert len(path) == 3
    assert path[0]["concept"] == "Root"
    assert path[1]["concept"] == "Middle"
    assert path[2]["concept"] == "Leaf"


def test_get_path_to_root_single_node():
    insert_node(None, "Alone", "Alone explanation")

    path = get_path_to_root("Alone")

    assert len(path) == 1
    assert path[0]["concept"] == "Alone"


def test_get_path_to_root_nonexistent():
    path = get_path_to_root("Ghost")
    assert path == []


# ---------------------------------------------------------------------------
# Session CRUD tests
# ---------------------------------------------------------------------------


def test_create_session():
    sid = create_session("Graph Theory")
    assert isinstance(sid, int)
    assert sid >= 1


def test_create_multiple_sessions():
    s1 = create_session("Session A")
    s2 = create_session("Session B")
    assert s1 != s2


def test_get_session():
    sid = create_session("My Session")
    session = get_session(sid)
    assert session is not None
    assert session["id"] == sid
    assert session["name"] == "My Session"
    assert "created_at" in session


def test_get_session_not_found():
    assert get_session(9999) is None


def test_list_sessions_empty():
    sessions = list_sessions()
    assert sessions == []


def test_list_sessions():
    s1 = create_session("First")
    s2 = create_session("Second")

    sessions = list_sessions()
    assert len(sessions) == 2

    # Both sessions should be present
    ids = {s["id"] for s in sessions}
    assert ids == {s1, s2}


def test_list_sessions_includes_node_count():
    sid = create_session("Counted")
    insert_node(None, "Root", "Root explanation", session_id=sid)
    insert_node("Root", "Child", "Child explanation", session_id=sid)

    sessions = list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["node_count"] >= 2


# ---------------------------------------------------------------------------
# Session-scoped insert_node and edge queries
# ---------------------------------------------------------------------------


def test_insert_node_with_session():
    sid = create_session("Session 1")
    insert_node(None, "Root", "Root explanation", session_id=sid)
    insert_node("Root", "Child", "Child explanation", session_id=sid)

    # Edges should be session-scoped
    edges = get_all_edges(session_id=sid)
    assert len(edges) == 1

    # Without session filter, same edges appear
    all_edges = get_all_edges()
    assert len(all_edges) == 1


def test_edges_isolated_between_sessions():
    s1 = create_session("Session A")
    s2 = create_session("Session B")

    insert_node(None, "Root", "Root explanation", session_id=s1)
    insert_node("Root", "Child A", "Child A explanation", session_id=s1)

    insert_node(None, "Root", "Root explanation", session_id=s2)  # Node reused
    insert_node("Root", "Child B", "Child B explanation", session_id=s2)

    edges_s1 = get_all_edges(session_id=s1)
    edges_s2 = get_all_edges(session_id=s2)

    assert len(edges_s1) == 1
    assert len(edges_s2) == 1

    # Total edges = 2
    all_edges = get_all_edges()
    assert len(all_edges) == 2


def test_duplicate_edge_prevented_within_session():
    """Inserting the same parent->child twice in the same session should not create duplicate edges."""
    sid = create_session("Dedup")
    insert_node(None, "P", "P exp", session_id=sid)
    insert_node("P", "C", "C exp", session_id=sid)
    insert_node("P", "C", "C exp", session_id=sid)  # duplicate

    edges = get_all_edges(session_id=sid)
    assert len(edges) == 1


def test_same_edge_allowed_in_different_sessions():
    """Same parent->child edge can exist in two different sessions."""
    s1 = create_session("S1")
    s2 = create_session("S2")

    insert_node(None, "P", "P exp", session_id=s1)
    insert_node("P", "C", "C exp", session_id=s1)

    insert_node(None, "P", "P exp", session_id=s2)
    insert_node("P", "C", "C exp", session_id=s2)

    assert len(get_all_edges(session_id=s1)) == 1
    assert len(get_all_edges(session_id=s2)) == 1
    assert len(get_all_edges()) == 2


# ---------------------------------------------------------------------------
# Session-scoped node queries
# ---------------------------------------------------------------------------


def test_get_all_nodes_with_session():
    s1 = create_session("S1")
    s2 = create_session("S2")

    insert_node(None, "A", "A exp", session_id=s1)
    insert_node("A", "B", "B exp", session_id=s1)

    insert_node(None, "C", "C exp", session_id=s2)
    insert_node("C", "D", "D exp", session_id=s2)

    nodes_s1 = get_all_nodes(session_id=s1)
    nodes_s2 = get_all_nodes(session_id=s2)
    all_nodes = get_all_nodes()

    assert len(nodes_s1) == 2
    assert len(nodes_s2) == 2
    assert len(all_nodes) == 4

    s1_concepts = {n["concept"] for n in nodes_s1}
    assert s1_concepts == {"A", "B"}

    s2_concepts = {n["concept"] for n in nodes_s2}
    assert s2_concepts == {"C", "D"}


def test_get_session_nodes():
    sid = create_session("Test")
    insert_node(None, "Root", "Root exp", session_id=sid)
    insert_node("Root", "Leaf", "Leaf exp", session_id=sid)

    nodes = get_session_nodes(sid)
    assert len(nodes) == 2
    concepts = {n["concept"] for n in nodes}
    assert concepts == {"Root", "Leaf"}


def test_get_session_edges():
    sid = create_session("Test")
    insert_node(None, "Root", "Root exp", session_id=sid)
    insert_node("Root", "C1", "C1 exp", session_id=sid)
    insert_node("Root", "C2", "C2 exp", session_id=sid)

    edges = get_session_edges(sid)
    assert len(edges) == 2


# ---------------------------------------------------------------------------
# get_session_root
# ---------------------------------------------------------------------------


def test_get_session_root():
    sid = create_session("Test")
    insert_node(None, "Root", "Root exp", session_id=sid)
    insert_node("Root", "Child", "Child exp", session_id=sid)
    insert_node("Child", "Grandchild", "GC exp", session_id=sid)

    root = get_session_root(sid)
    assert root is not None
    assert root["concept"] == "Root"


def test_get_session_root_no_edges():
    sid = create_session("Empty Session")
    root = get_session_root(sid)
    assert root is None


# ---------------------------------------------------------------------------
# Session-scoped get_path_to_root
# ---------------------------------------------------------------------------


def test_get_path_to_root_with_session():
    sid = create_session("Path Test")
    insert_node(None, "Root", "Root exp", session_id=sid)
    insert_node("Root", "Mid", "Mid exp", session_id=sid)
    insert_node("Mid", "Leaf", "Leaf exp", session_id=sid)

    path = get_path_to_root("Leaf", session_id=sid)
    assert len(path) == 3
    assert path[0]["concept"] == "Root"
    assert path[1]["concept"] == "Mid"
    assert path[2]["concept"] == "Leaf"


def test_get_path_to_root_session_isolation():
    """Path-to-root should only follow edges within the given session."""
    s1 = create_session("S1")
    s2 = create_session("S2")

    # Session 1: Root -> Mid -> Leaf
    insert_node(None, "Root", "Root exp", session_id=s1)
    insert_node("Root", "Mid", "Mid exp", session_id=s1)
    insert_node("Mid", "Leaf", "Leaf exp", session_id=s1)

    # Session 2: OtherRoot -> Leaf (same Leaf node, different parent)
    insert_node(None, "OtherRoot", "Other exp", session_id=s2)
    insert_node("OtherRoot", "Leaf", "Leaf exp", session_id=s2)

    path_s1 = get_path_to_root("Leaf", session_id=s1)
    assert len(path_s1) == 3
    assert path_s1[0]["concept"] == "Root"

    path_s2 = get_path_to_root("Leaf", session_id=s2)
    assert len(path_s2) == 2
    assert path_s2[0]["concept"] == "OtherRoot"


# ---------------------------------------------------------------------------
# Schema migration test
# ---------------------------------------------------------------------------


def test_migration_adds_session_id_column(tmp_path):
    """init_db should add session_id column to an existing edges table that lacks it."""
    db_file = tmp_path / "migrate_test.db"
    os.environ["CONCEPT_DFS_DB"] = str(db_file)

    # Create an old-style DB without session_id
    conn = sqlite3.connect(str(db_file))
    conn.execute("""
        CREATE TABLE nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept TEXT UNIQUE NOT NULL,
            explanation TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE edges (
            parent_id INTEGER,
            child_id INTEGER,
            FOREIGN KEY(parent_id) REFERENCES nodes(id),
            FOREIGN KEY(child_id) REFERENCES nodes(id)
        )
    """)
    conn.commit()
    conn.close()

    # Now call init_db — it should migrate by adding session_id
    init_db()

    conn = sqlite3.connect(str(db_file))
    cursor = conn.execute("PRAGMA table_info(edges)")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()

    assert "session_id" in columns


def test_migration_adds_keywords_column(tmp_path):
    """init_db should add keywords column to an existing nodes table that lacks it."""
    db_file = tmp_path / "migrate_keywords_test.db"
    os.environ["CONCEPT_DFS_DB"] = str(db_file)

    # Create a DB without keywords column on nodes
    conn = sqlite3.connect(str(db_file))
    conn.execute("""
        CREATE TABLE nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept TEXT UNIQUE NOT NULL,
            explanation TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE edges (
            parent_id INTEGER,
            child_id INTEGER,
            session_id INTEGER,
            FOREIGN KEY(parent_id) REFERENCES nodes(id),
            FOREIGN KEY(child_id) REFERENCES nodes(id),
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()
    conn.close()

    # Now call init_db — it should migrate by adding keywords
    init_db()

    conn = sqlite3.connect(str(db_file))
    cursor = conn.execute("PRAGMA table_info(nodes)")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()

    assert "keywords" in columns


# ---------------------------------------------------------------------------
# Keywords storage tests
# ---------------------------------------------------------------------------


def test_insert_node_with_keywords():
    """insert_node stores keywords and get_node returns them as a list."""
    keywords = ["DFS", "BFS", "Dijkstra"]
    insert_node(None, "Graph Traversal", "Visiting nodes.", keywords=keywords)

    node = get_node("Graph Traversal")
    assert node is not None
    assert node["keywords"] == ["DFS", "BFS", "Dijkstra"]


def test_insert_node_without_keywords():
    """insert_node without keywords stores None; get_node returns empty list."""
    insert_node(None, "Simple", "No keywords here.")

    node = get_node("Simple")
    assert node is not None
    assert node["keywords"] == []


def test_insert_node_keywords_update_on_reinsert():
    """If a node is reinserted with keywords after initially having none, keywords are updated."""
    insert_node(None, "Evolving", "Initial explanation.")
    node = get_node("Evolving")
    assert node is not None
    assert node["keywords"] == []

    # Reinsert with keywords — INSERT OR IGNORE won't change the row,
    # but the UPDATE should fill in keywords
    insert_node(None, "Evolving", "Initial explanation.", keywords=["A", "B"])
    node = get_node("Evolving")
    assert node is not None
    assert node["keywords"] == ["A", "B"]


def test_insert_node_keywords_not_overwritten():
    """If a node already has keywords, reinserting without keywords does not clear them."""
    insert_node(None, "Stable", "Explanation.", keywords=["X", "Y"])
    node = get_node("Stable")
    assert node is not None
    assert node["keywords"] == ["X", "Y"]

    # Reinsert without keywords — should not overwrite existing
    insert_node(None, "Stable", "Explanation.")
    node = get_node("Stable")
    assert node is not None
    assert node["keywords"] == ["X", "Y"]


def test_get_path_to_root_includes_keywords():
    """get_path_to_root returns keywords in each node dict."""
    insert_node(None, "Root", "Root exp", keywords=["A", "B"])
    insert_node("Root", "Child", "Child exp", keywords=["C", "D"])

    path = get_path_to_root("Child")
    assert len(path) == 2
    assert path[0]["keywords"] == ["A", "B"]
    assert path[1]["keywords"] == ["C", "D"]
