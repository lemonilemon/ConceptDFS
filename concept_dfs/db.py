import sqlite3
import os
from contextlib import contextmanager
from typing import Optional, Dict, Any, List

DB_PATH = os.environ.get("CONCEPT_DFS_DB", "concepts.db")

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()

def init_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create nodes table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept TEXT UNIQUE NOT NULL,
            explanation TEXT NOT NULL
        )
        ''')
        
        # Create edges table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS edges (
            parent_id INTEGER,
            child_id INTEGER,
            FOREIGN KEY(parent_id) REFERENCES nodes(id),
            FOREIGN KEY(child_id) REFERENCES nodes(id)
        )
        ''')
        
        # Indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_concept ON nodes(concept)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_parent_child ON edges(parent_id, child_id)')

def get_node(concept: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, concept, explanation FROM nodes WHERE concept = ?", (concept,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

def insert_node(parent_concept: Optional[str], concept: str, explanation: str) -> None:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Insert or ignore the new concept
        cursor.execute(
            "INSERT OR IGNORE INTO nodes (concept, explanation) VALUES (?, ?)", 
            (concept, explanation)
        )
        
        if parent_concept:
            # We need to get both IDs
            cursor.execute("SELECT id FROM nodes WHERE concept = ?", (concept,))
            child_row = cursor.fetchone()
            if not child_row:
                return
            child_id = child_row['id']
            
            cursor.execute("SELECT id FROM nodes WHERE concept = ?", (parent_concept,))
            parent_row = cursor.fetchone()
            if parent_row:
                parent_id = parent_row['id']
                
                # Check if edge already exists to prevent duplicate relationships
                cursor.execute(
                    "SELECT 1 FROM edges WHERE parent_id = ? AND child_id = ?",
                    (parent_id, child_id)
                )
                if not cursor.fetchone():
                    cursor.execute(
                        "INSERT INTO edges (parent_id, child_id) VALUES (?, ?)",
                        (parent_id, child_id)
                    )

def get_all_nodes() -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, concept, explanation FROM nodes")
        return [dict(row) for row in cursor.fetchall()]

def get_all_edges() -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT parent_id, child_id FROM edges")
        return [dict(row) for row in cursor.fetchall()]
