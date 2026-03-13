import os
import pytest
from src.db import init_db, get_node, insert_node, get_all_nodes, get_all_edges

@pytest.fixture(autouse=True)
def test_db(tmp_path):
    db_file = tmp_path / "test_concepts.db"
    os.environ["CONCEPT_DFS_DB"] = str(db_file)
    init_db()
    yield
    if db_file.exists():
        os.remove(db_file)

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
