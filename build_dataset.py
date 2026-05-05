import os
# Fix Qt Wayland + font warning on Arch Linux / Hyprland
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_FONT_DPI"] = "96"

import cv2
import glob
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

DATASET_DIR = 'dataset'

def get_existing_folders():
    """Get list of existing person folders"""
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        return []
    folders = [f for f in os.listdir(DATASET_DIR) 
               if os.path.isdir(os.path.join(DATASET_DIR, f))]
    return sorted(folders)

def get_next_index(person_name):
    """Get the next available index number for the person's images"""
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.exists(person_dir):
        return 0
    
    # Find all files matching the pattern: name_xxxxx.png
    image_files = glob.glob(os.path.join(person_dir, f"{person_name}_*.png"))
    if not image_files:
        return 0
    
    indices = []
    for img_file in image_files:
        try:
            filename = os.path.basename(img_file)
            # Extract number after the last underscore
            number_str = filename.split('_')[-1].split('.')[0]
            indices.append(int(number_str))
        except:
            continue
    
    return max(indices) + 1 if indices else 0

def save_captured_images(person_name, images, start_idx):
    """Save captured images with progress"""
    if not images:
        return 0
    
    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    with Progress(SpinnerColumn(), TextColumn("Saving images...")) as progress:
        task = progress.add_task("", total=len(images))
        idx = start_idx
        for img in images:
            filename = f"{person_name}_{idx:05d}.png"
            cv2.imwrite(os.path.join(person_dir, filename), img)
            idx += 1
            progress.advance(task)
    
    console.print(f"[green] Saved {len(images)} images to dataset/{person_name}/[/green]")
    return len(images)

def main():
    console.print(Panel.fit("[bold cyan]FACE DATASET BUILDER CLI[/bold cyan]", border_style="blue"))
    
    # Open camera only once
    console.print("[yellow]Opening camera...[/yellow]")
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        video = cv2.VideoCapture(0, cv2.CAP_V4L2)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow("Dataset Builder CLI", cv2.WINDOW_NORMAL)
    console.print("[bold green]Camera ready![/bold green]\n")

    # Main loop - support adding many persons continuously
    while True:
        existing = get_existing_folders()
        if existing:
            console.print("[bold]Existing persons:[/bold]")
            for p in existing:
                console.print(f"   • [yellow]{p}[/yellow]")
        else:
            console.print("[dim]No persons yet.[/dim]")

        name = console.input("\n[bold]Enter person's name[/bold] (or Q to quit): ").strip()
        
        if name.lower() == 'q':
            console.print("[green]Exiting program...[/green]")
            break

        if not name:
            console.print("[red]Name cannot be empty![/red]")
            continue

        # Check if person already exists
        is_new = not os.path.exists(os.path.join(DATASET_DIR, name))
        idx = get_next_index(name)

        if is_new:
            console.print(f"[green] Created new folder for: {name}[/green]")
        else:
            console.print(f"[green] Adding more images to existing folder: {name}[/green]")
        
        console.print(f"[green]   Starting from image: {idx:05d}\n[/green]")

        total_captured = 0
        images = []

        console.print("[bold]Press [red]P[/red] = Capture   [red]M[/red] = Next person   [red]Q[/red] = Quit[/bold]")

        while True:
            ret, frame = video.read()
            if not ret:
                console.print("[red]Failed to grab frame![/red]")
                break

            frame = cv2.flip(frame, 1)

            display = frame.copy()
            cv2.putText(display, "P: Capture  M: Next person  Q: Quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Person: {name} | Captured: {total_captured}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Dataset Builder CLI", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('p') or key == ord('P'):
                images.append(frame.copy())
                total_captured += 1
                console.print(f"[cyan]Captured {total_captured}[/cyan]")

            elif key == ord('m') or key == ord('M'):
                save_captured_images(name, images, idx)
                console.print("[yellow]→ Switching to new person...[/yellow]\n")
                break

            elif key == ord('q') or key == ord('Q'):
                save_captured_images(name, images, idx)
                console.print("[green]Exiting program...[/green]")
                video.release()
                cv2.destroyAllWindows()
                return

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()