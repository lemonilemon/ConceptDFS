import os
import subprocess
import pytest
from concept_dfs.db import init_db, insert_node

@pytest.fixture
def test_db_setup(tmp_path):
    db_file = tmp_path / "test_inspect.db"
    os.environ["CONCEPT_DFS_DB"] = str(db_file)
    init_db()
    insert_node(None, "SQL Test", "Checking if script works.")
    yield str(db_file)
    if db_file.exists():
        os.remove(db_file)

def test_inspect_db_script(test_db_setup):
    # Run the shell script with a SELECT query
    script_path = "tools/inspect_db.sh"
    query = "SELECT concept FROM nodes WHERE concept='SQL Test';"
    
    # We must ensure the env var is passed to the subprocess
    env = os.environ.copy()
    
    result = subprocess.run(
        ["bash", script_path, query],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert result.returncode == 0
    assert "SQL Test" in result.stdout

def test_inspect_db_script_error(test_db_setup):
    script_path = "tools/inspect_db.sh"
    invalid_query = "SELECT * FROM non_existent_table;"
    
    env = os.environ.copy()
    result = subprocess.run(
        ["bash", script_path, invalid_query],
        capture_output=True,
        text=True,
        env=env
    )
    
    assert result.returncode != 0
    assert "Error:" in result.stderr
