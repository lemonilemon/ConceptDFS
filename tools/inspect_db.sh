#!/bin/bash
# A simple wrapper to execute SQL against the local cache.
# Providing an absolute path is better but keeping to the design: 
# it expects concepts.db in the cwd where it gets called.
sqlite3 concepts.db "$1"
