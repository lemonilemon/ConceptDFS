# ConceptDFS

A minimal, keyboard-driven CLI tool designed to solve a researcher's problem: efficiently exploring branching knowledge using a Stack-based Depth-First Search (DFS) over an LLM-generated knowledge graph.

It relies on a SQLite caching layer to ensure fast traversal, reduced token costs, and protection against circular referencing or infinite loops.

## Features
- **LIFO Stack-based DFS**: Explore deep concept branches naturally.
- **Local SQLite Caching**: Prevent redundant LLM calls and guard against infinite loops.
- **Keyboard-driven Interface**: Built with rich for a fast, TUI-like experience.
- **Markdown Export**: Generate report.md, optionally with Mermaid diagrams representing your exploration graph.

## Running
You can execute the application directly using `uvx`:
```bash
uvx --from . concept-dfs
```
