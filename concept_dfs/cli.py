import sys
from rich.console import Console
from concept_dfs.db import init_db, list_sessions, get_session_root
from concept_dfs.provider import select_model, force_auth
from concept_dfs.app import build_report, ConceptDFSApp

console = Console()


def export_report(session_id=None):
    """Export the exploration report to report.md (standalone CLI command)."""
    content = build_report(session_id=session_id)
    if content is None:
        console.print("[yellow]No concepts found in database to export.[/yellow]")
        return

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(content)

    scope = f"session #{session_id}" if session_id else "all sessions"
    console.print(
        f"[bold green]Success: Report exported to report.md ({scope})[/bold green]"
    )


def show_sessions():
    """List all exploration sessions."""
    sessions = list_sessions()
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    console.print("\n[bold cyan]Exploration Sessions:[/bold cyan]")
    for s in sessions:
        root_node = get_session_root(s["id"])
        root_label = root_node["concept"] if root_node else s["name"]
        console.print(
            f"  [bold green]#{s['id']}[/bold green]  {root_label}  "
            f"[dim]({s['created_at']})[/dim]"
        )
    console.print()


def main():
    if len(sys.argv) >= 2:
        first_arg = sys.argv[1].lower()

        if first_arg == "export":
            init_db()
            # Check for --session N flag
            sid = None
            if "--session" in sys.argv:
                idx = sys.argv.index("--session")
                if idx + 1 < len(sys.argv):
                    try:
                        sid = int(sys.argv[idx + 1])
                    except ValueError:
                        console.print("[red]Invalid session ID.[/red]")
                        return
            export_report(session_id=sid)
            return
        elif first_arg == "sessions":
            init_db()
            show_sessions()
            return
        elif first_arg == "model":
            select_model()
            return
        elif first_arg == "auth":
            force_auth()
            return
        elif first_arg in ("--help", "-h", "help"):
            console.print("[cyan]ConceptDFS[/cyan]")
            console.print("\n[bold]Usage:[/bold]")
            console.print("  concept-dfs              - Launch the interactive TUI")
            console.print(
                "  concept-dfs <concept>    - Launch the TUI with an initial concept"
            )
            console.print("  concept-dfs model        - Select LLM provider and model")
            console.print("  concept-dfs auth         - Configure API keys")
            console.print("  concept-dfs sessions     - List all exploration sessions")
            console.print(
                "  concept-dfs export       - Export all sessions to report.md"
            )
            console.print(
                "  concept-dfs export --session N  - Export a specific session"
            )
            return
        else:
            # Treat remaining args as the initial concept
            initial_concept = " ".join(sys.argv[1:])
            app = ConceptDFSApp(initial_concept=initial_concept)
            app.run()
            return

    # No arguments — launch the interactive Textual TUI
    app = ConceptDFSApp()
    app.run()


if __name__ == "__main__":
    main()
