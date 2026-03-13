import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from concept_dfs.db import init_db, get_node, insert_node, get_all_nodes, get_all_edges
from concept_dfs.llm import fetch_concept

console = Console()

def export_report():
    nodes = get_all_nodes()
    edges = get_all_edges()
    
    if not nodes:
        console.print("[yellow]No concepts found in database to export.[/yellow]")
        return
        
    lines = ["# ConceptDFS Exploration Report\n"]
    
    lines.append("## Knowledge Graph\n")
    lines.append("```mermaid")
    lines.append("graph TD;")
    
    id_to_concept = {n['id']: n['concept'] for n in nodes}
    
    for edge in edges:
        parent = id_to_concept.get(edge['parent_id'])
        child = id_to_concept.get(edge['child_id'])
        if parent and child:
            p_safe = parent.replace('"', '')
            c_safe = child.replace('"', '')
            lines.append(f'  {edge["parent_id"]}["{p_safe}"] --> {edge["child_id"]}["{c_safe}"]')
    
    lines.append("```\n")
    
    lines.append("## Concepts Explained\n")
    for n in nodes:
        lines.append(f"### {n['concept']}\n")
        lines.append(n['explanation'])
        lines.append("\n")
        
    with open("report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    console.print("[bold green]Success: Report exported to report.md[/bold green]")

def main():
    if len(sys.argv) < 2:
        console.print("[bold red]Usage: uvx --from . concept-dfs '<initial concept>'[/bold red]")
        sys.exit(1)
        
    initial_query = " ".join(sys.argv[1:])
    
    init_db()
    
    if initial_query.lower() == "export":
        export_report()
        sys.exit(0)
    
    stack = [(None, initial_query)]
    
    console.print(f"[bold blue]Starting ConceptDFS for: {initial_query}[/bold blue]\n")
    
    while stack:
        parent, concept = stack.pop()
        
        console.rule(f"[bold magenta]Exploring: {concept}[/bold magenta]")
        
        node = get_node(concept)
        if node:
            explanation = node['explanation']
            insert_node(parent, concept, explanation)
            console.print("[dim italic]Loaded from local cache.[/dim italic]\n")
            console.print(Markdown(explanation))
            console.print()
            
            console.print("[dim]Type 'export' to save report, or press Enter to continue stack...[/dim]")
            user_input = Prompt.ask("Action").strip()
            if user_input.lower() == 'export':
                export_report()
            continue
            
        console.print(f"[yellow]Fetching explanation from LLM...[/yellow]")
        try:
            response = fetch_concept(concept)
            explanation = response.explanation
            keywords = response.keywords
            
            insert_node(parent, concept, explanation)
            
        except Exception as e:
            console.print(f"[bold red]Error fetching concept: {e}[/bold red]")
            continue
            
        console.print(Markdown(explanation))
        console.print()
        
        if keywords:
            console.print("[bold cyan]Related concepts to explore:[/bold cyan]")
            for i, kw in enumerate(keywords, 1):
                console.print(f"  [bold green]{i}.[/bold green] {kw}")
                
            console.print("\n[dim]Enter space-separated numbers to push them onto the stack (e.g., '1 3'), 'export' to save report, or press Enter to skip.[/dim]")
            
            while True:
                user_input = Prompt.ask("Your choice").strip()
                
                if user_input.lower() == 'export':
                    export_report()
                    continue
                    
                if not user_input:
                    break
                    
                parts = user_input.split()
                valid = True
                selected_indices = []
                for p in parts:
                    if p.isdigit():
                        idx = int(p)
                        if 1 <= idx <= len(keywords):
                            selected_indices.append(idx - 1)
                        else:
                            console.print(f"[red]Invalid number: {idx}[/red]")
                            valid = False
                    else:
                        console.print(f"[red]Invalid input: {p}[/red]")
                        valid = False
                        
                if valid:
                    # Push descending order so they are popped in ascending order
                    for idx in reversed(selected_indices):
                        stack.append((concept, keywords[idx]))
                    break

    console.print("[bold green]Exploration complete! Stack is empty.[/bold green]")
    export_now = Prompt.ask("Would you like to export the final report? (y/n)").strip().lower()
    if export_now in ['y', 'yes', 'export']:
        export_report()

if __name__ == "__main__":
    main()
