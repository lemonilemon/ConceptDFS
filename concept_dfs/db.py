import sqlite3
import json
import os
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
from concept_dfs.paths import DATA_DIR


def get_db_path():
    return os.environ.get("CONCEPT_DFS_DB", str(DATA_DIR / "concepts.db"))


@contextmanager
def get_db_connection():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()


def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Create nodes table (global concept cache)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept TEXT UNIQUE NOT NULL,
            explanation TEXT NOT NULL,
            keywords TEXT
        )
        """)

        # Create sessions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """)

        # Create edges table with optional session_id
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            parent_id INTEGER,
            child_id INTEGER,
            session_id INTEGER,
            FOREIGN KEY(parent_id) REFERENCES nodes(id),
            FOREIGN KEY(child_id) REFERENCES nodes(id),
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
        """)

        # Migrate: add session_id column if missing (existing databases)
        cursor.execute("PRAGMA table_info(edges)")
        columns = [row["name"] for row in cursor.fetchall()]
        if "session_id" not in columns:
            cursor.execute(
                "ALTER TABLE edges ADD COLUMN session_id INTEGER REFERENCES sessions(id)"
            )

        # Migrate: add keywords column to nodes if missing (existing databases)
        cursor.execute("PRAGMA table_info(nodes)")
        node_columns = [row["name"] for row in cursor.fetchall()]
        if "keywords" not in node_columns:
            cursor.execute("ALTER TABLE nodes ADD COLUMN keywords TEXT")

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_concept ON nodes(concept)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_parent_child ON edges(parent_id, child_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_edges_session ON edges(session_id)"
        )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def _deserialize_node(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a DB row to a dict, deserializing the keywords JSON field."""
    d = dict(row)
    if "keywords" in d:
        raw = d.get("keywords")
        d["keywords"] = json.loads(raw) if raw else []
    return d


def create_session(name: str) -> int:
    """Create a new exploration session and return its ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sessions (name) VALUES (?)", (name,))
        row_id = cursor.lastrowid
        if row_id is None:
            raise RuntimeError("INSERT did not set lastrowid")
        return row_id


def list_sessions() -> List[Dict[str, Any]]:
    """Return all sessions ordered by most recent first."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.id, s.name, s.created_at,
                   COUNT(DISTINCT e.child_id) + COUNT(DISTINCT CASE WHEN e.parent_id NOT IN
                       (SELECT child_id FROM edges WHERE session_id = s.id) THEN e.parent_id END)
                   AS node_count
            FROM sessions s
            LEFT JOIN edges e ON e.session_id = s.id
            GROUP BY s.id
            ORDER BY s.created_at DESC
        """)
        return [dict(row) for row in cursor.fetchall()]


def get_session(session_id: int) -> Optional[Dict[str, Any]]:
    """Get a single session by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, created_at FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_session_nodes(session_id: int) -> List[Dict[str, Any]]:
    """Return all nodes that appear in edges for a given session."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT n.id, n.concept, n.explanation, n.keywords
            FROM nodes n
            WHERE n.id IN (
                SELECT parent_id FROM edges WHERE session_id = ?
                UNION
                SELECT child_id FROM edges WHERE session_id = ?
            )
        """,
            (session_id, session_id),
        )
        return [_deserialize_node(row) for row in cursor.fetchall()]


def get_session_edges(session_id: int) -> List[Dict[str, Any]]:
    """Return all edges for a given session."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT parent_id, child_id FROM edges WHERE session_id = ?",
            (session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_session_root(session_id: int) -> Optional[Dict[str, Any]]:
    """Return the root node of a session (node that appears as parent but not as child)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Root is a parent node that is never a child within this session
        cursor.execute(
            """
            SELECT DISTINCT n.id, n.concept, n.explanation, n.keywords
            FROM nodes n
            JOIN edges e ON e.parent_id = n.id AND e.session_id = ?
            WHERE n.id NOT IN (
                SELECT child_id FROM edges WHERE session_id = ?
            )
            LIMIT 1
        """,
            (session_id, session_id),
        )
        row = cursor.fetchone()
        return _deserialize_node(row) if row else None


# ---------------------------------------------------------------------------
# Node operations
# ---------------------------------------------------------------------------


def get_node(concept: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, concept, explanation, keywords FROM nodes WHERE concept = ?",
            (concept,),
        )
        row = cursor.fetchone()
        if row:
            return _deserialize_node(row)
        return None


def insert_node(
    parent_concept: Optional[str],
    concept: str,
    explanation: str,
    session_id: Optional[int] = None,
    keywords: Optional[List[str]] = None,
) -> None:
    with get_db_connection() as conn:
        cursor = conn.cursor()

        keywords_json = json.dumps(keywords) if keywords is not None else None

        # Insert or ignore the new concept
        cursor.execute(
            "INSERT OR IGNORE INTO nodes (concept, explanation, keywords) VALUES (?, ?, ?)",
            (concept, explanation, keywords_json),
        )

        # If the node already existed but had no keywords, update them
        if keywords_json:
            cursor.execute(
                "UPDATE nodes SET keywords = ? WHERE concept = ? AND (keywords IS NULL OR keywords = '[]')",
                (keywords_json, concept),
            )

        if parent_concept:
            # We need to get both IDs
            cursor.execute("SELECT id FROM nodes WHERE concept = ?", (concept,))
            child_row = cursor.fetchone()
            if not child_row:
                return
            child_id = child_row["id"]

            cursor.execute("SELECT id FROM nodes WHERE concept = ?", (parent_concept,))
            parent_row = cursor.fetchone()
            if parent_row:
                parent_id = parent_row["id"]

                # Check if edge already exists (within this session) to prevent duplicates
                if session_id is not None:
                    cursor.execute(
                        "SELECT 1 FROM edges WHERE parent_id = ? AND child_id = ? AND session_id = ?",
                        (parent_id, child_id, session_id),
                    )
                else:
                    cursor.execute(
                        "SELECT 1 FROM edges WHERE parent_id = ? AND child_id = ? AND session_id IS NULL",
                        (parent_id, child_id),
                    )
                if not cursor.fetchone():
                    cursor.execute(
                        "INSERT INTO edges (parent_id, child_id, session_id) VALUES (?, ?, ?)",
                        (parent_id, child_id, session_id),
                    )


def get_all_nodes(session_id: Optional[int] = None) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if session_id is not None:
            cursor.execute(
                """
                SELECT DISTINCT n.id, n.concept, n.explanation, n.keywords
                FROM nodes n
                WHERE n.id IN (
                    SELECT parent_id FROM edges WHERE session_id = ?
                    UNION
                    SELECT child_id FROM edges WHERE session_id = ?
                )
            """,
                (session_id, session_id),
            )
        else:
            cursor.execute("SELECT id, concept, explanation, keywords FROM nodes")
        return [_deserialize_node(row) for row in cursor.fetchall()]


def get_all_edges(session_id: Optional[int] = None) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if session_id is not None:
            cursor.execute(
                "SELECT parent_id, child_id FROM edges WHERE session_id = ?",
                (session_id,),
            )
        else:
            cursor.execute("SELECT parent_id, child_id FROM edges")
        return [dict(row) for row in cursor.fetchall()]


def get_path_to_root(
    concept: str, session_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Walk edges upward from a concept to the root, returning the path in root-to-node order.

    Returns a list of dicts with 'concept' and 'explanation' keys,
    ordered from root to the given concept (inclusive).

    If session_id is provided, only follows edges within that session.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        path = []
        current = concept
        visited: set[str] = set()

        while current:
            if current in visited:
                break
            visited.add(current)
            cursor.execute(
                "SELECT id, concept, explanation, keywords FROM nodes WHERE concept = ?",
                (current,),
            )
            row = cursor.fetchone()
            if not row:
                break

            raw_kw = row["keywords"]
            kw_list = json.loads(raw_kw) if raw_kw else []
            path.append(
                {
                    "concept": row["concept"],
                    "explanation": row["explanation"],
                    "keywords": kw_list,
                }
            )

            # Find parent via edges (optionally within session)
            if session_id is not None:
                cursor.execute(
                    """SELECT n.concept FROM nodes n
                       JOIN edges e ON e.parent_id = n.id
                       WHERE e.child_id = ? AND e.session_id = ?""",
                    (row["id"], session_id),
                )
            else:
                cursor.execute(
                    """SELECT n.concept FROM nodes n
                       JOIN edges e ON e.parent_id = n.id
                       WHERE e.child_id = ?""",
                    (row["id"],),
                )
            parent_row = cursor.fetchone()
            current = parent_row["concept"] if parent_row else None

        # Reverse to get root-to-node order
        path.reverse()
        return path
