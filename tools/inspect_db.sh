#!/bin/bash
# A simple wrapper to execute SQL against the local cache using Python.
DB_PATH=${CONCEPT_DFS_DB:-${XDG_DATA_HOME:-$HOME/.local/share}/concept-dfs/concepts.db}
python3 -c "
import sqlite3, sys
conn = sqlite3.connect('$DB_PATH')
cursor = conn.cursor()
try:
    cursor.execute(sys.argv[1])
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    conn.commit()
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
" "$1"
