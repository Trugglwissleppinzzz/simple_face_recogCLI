import os
# Fix Qt Wayland + font warning on Arch Linux / Hyprland
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_FONT_DPI"] = "96"

import pickle
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from collections import Counter

console = Console()

ENCODINGS_FILE = 'encodings.pickle'

def main():
    console.print(Panel.fit(
        "[bold magenta]FACE RECOGNITION MODEL TESTER[/bold magenta]",
        border_style="magenta",
        title="Test & Inspect Model",
        subtitle="Check trained encoder"
    ))
    
    console.print("[yellow]Loading encodings.pickle...[/yellow]")
    
    if not os.path.exists(ENCODINGS_FILE):
        console.print("[red]❌ File encodings.pickle not found![/red]")
        console.print("[red]Please run: python encode_faces.py first[/red]")
        return

    # Load model
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    # Parse data
    encodings = data.get("encodings", [])
    names = data.get("names", [])
    total_encodings = len(encodings)
    encodings_per_person = Counter(names)
    unique_people = sorted(encodings_per_person.keys())
    num_people = len(unique_people)

    # Validation
    is_valid = len(names) == total_encodings
    encoding_dim = len(encodings[0]) if encodings else 0

    console.print("[green]✅ Model loaded successfully![/green]\n")

    console.print(f"[bold]Total face encodings:[/bold] [cyan]{total_encodings}[/cyan]")
    console.print(f"[bold]Number of people trained:[/bold] [yellow]{num_people}[/yellow]")
    console.print(f"[bold]Encoding vector size:[/bold] [cyan]{encoding_dim} dimensions[/cyan]")
    console.print(f"[bold]Model valid:[/bold] [{'green' if is_valid else 'red'}]{is_valid}[/]\n")

    if total_encodings > 0:
        most_trained = encodings_per_person.most_common(1)[0]
        least_trained = min(encodings_per_person.items(), key=lambda x: x[1])
        avg_per_person = total_encodings / num_people if num_people > 0 else 0

        console.print(f"[bold]Most trained person :[/bold] [yellow]{most_trained[0]}[/yellow] ({most_trained[1]} encodings)")
        console.print(f"[bold]Least trained person:[/bold] [yellow]{least_trained[0]}[/yellow] ({least_trained[1]} encodings)")
        console.print(f"[bold]Average encodings/person:[/bold] [cyan]{avg_per_person:.1f}[/cyan]\n")

    table = Table(title="Encodings per Person", show_header=True, header_style="bold cyan")
    table.add_column("Person Name", style="yellow", justify="left")
    table.add_column("Encodings", style="cyan", justify="center")
    table.add_column("% of Total", style="magenta", justify="center")

    for person, count in sorted(encodings_per_person.items()):
        percentage = (count / total_encodings * 100) if total_encodings > 0 else 0
        table.add_row(person, str(count), f"{percentage:.1f}%")
    
    console.print(table)

    console.print(f"\n[bold]All trained people ({num_people} people):[/bold]")
    console.print(f"[yellow]{', '.join(unique_people)}[/yellow]\n")

    console.print("[bold green]✅ Model test completed! Ready for real-time recognition.[/bold green]")
    console.print("[dim]Tip: Run 'python recognize_faces_video.py' to start camera recognition[/dim]")

if __name__ == "__main__":
    main()