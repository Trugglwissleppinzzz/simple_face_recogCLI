import os
# Fix Qt Wayland + font warning on Arch Linux / Hyprland
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_FONT_DPI"] = "96"

import cv2
from imutils import paths
import face_recognition
import pickle
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from collections import Counter   

console = Console()

DATASET_DIR = 'dataset'
ENCODINGS_FILE = 'encodings.pickle'
DETECTION_METHOD = 'hog'

def main():
    console.print(Panel.fit(
        "[bold magenta]FACE RECOGNITION TRAINING CLI[/bold magenta]",
        border_style="magenta"
    ))
    
    if not os.path.exists(DATASET_DIR):
        console.print("[red] Dataset folder not found![/red]")
        return

    person_folders = [f for f in os.listdir(DATASET_DIR) 
                      if os.path.isdir(os.path.join(DATASET_DIR, f))]
    
    if not person_folders:
        console.print("[red] No persons found in dataset/[/red]")
        return

    console.print(f"[bold]Found {len(person_folders)} person(s):[/bold] {person_folders}\n")

    known_encodings = []
    known_names = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        
        for folder in person_folders:
            image_paths = list(paths.list_images(os.path.join(DATASET_DIR, folder)))
            if not image_paths:
                console.print(f"[yellow] No images in folder: {folder}[/yellow]")
                continue
                
            task = progress.add_task(f"[cyan]Processing {folder}...", total=len(image_paths))
            
            for image_path in image_paths:
                name = folder
                image = cv2.imread(image_path)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                boxes = face_recognition.face_locations(rgb, model=DETECTION_METHOD)
                encodings = face_recognition.face_encodings(rgb, boxes)
                
                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(name)
                
                progress.advance(task)

            console.print(f"[green] Finished folder: {folder} ({len(image_paths)} images)[/green]")

    # Save encodings
    data = {"encodings": known_encodings, "names": known_names}
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    encodings_per_person = Counter(known_names)
    console.print("\n[bold green] TRAINING COMPLETED SUCCESSFULLY![/bold green]")
    console.print(f"   Total face encodings: [cyan]{len(known_encodings)}[/cyan]")
    console.print(f"   Encodings per person  : [cyan]{dict(encodings_per_person)}[/cyan]")  # ← Dictionary bạn yêu cầu
    console.print(f"   People trained        : [yellow]{list(encodings_per_person.keys())}[/yellow]")
    console.print(f"   Encodings saved to    : [blue]{ENCODINGS_FILE}[/blue]")

if __name__ == "__main__":
    main()