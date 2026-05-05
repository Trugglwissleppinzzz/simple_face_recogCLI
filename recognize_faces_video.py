import os
# Fix Qt Wayland + font warning on Arch Linux / Hyprland
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_FONT_DPI"] = "96"

import cv2
import pickle
import face_recognition
import imutils
import time
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.status import Status

console = Console()

# ================== CONFIG ==================
ENCODINGS_FILE = 'encodings.pickle'
DETECTION_METHOD = 'hog'
RESIZE_WIDTH = 320
SKIP_FRAMES = 5
TOLERANCE = 0.45
# ===========================================

def main():
    console.print(Panel.fit(
        "[bold cyan]FACE RECOGNITION CLI[/bold cyan]",
        border_style="cyan"
    ))
    
    if not os.path.exists(ENCODINGS_FILE):
        console.print("[red] encodings.pickle not found![/red]")
        return

    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    
    console.print(f"[green]Loaded {len(data['encodings'])} encodings for {len(set(data['names']))} people[/green]\n")

    console.print("[yellow]Opening camera...[/yellow]")
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        video = cv2.VideoCapture(0, cv2.CAP_V4L2)

    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video.set(cv2.CAP_PROP_FPS, 60)
    cv2.namedWindow("Face Recognition CLI", cv2.WINDOW_NORMAL)
    console.print("[bold green] Camera started! Press Q to quit[/bold green]\n")

    frame_count = 0
    last_boxes = []
    last_names = []
    start_time = time.time()
    fps = 0

    with Status("[bold yellow]Running face recognition...[/bold yellow] (Multi-people supported)", 
                spinner="dots", console=console, refresh_per_second=10):
        
        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_count += 1

            if frame_count % SKIP_FRAMES == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = imutils.resize(rgb, width=RESIZE_WIDTH)
                r = frame.shape[1] / float(rgb.shape[1])

                boxes = face_recognition.face_locations(
                    rgb, model=DETECTION_METHOD, number_of_times_to_upsample=0
                )
                encodings = face_recognition.face_encodings(rgb, boxes)

                names = []
                for encoding in encodings:
                    distances = face_recognition.face_distance(data["encodings"], encoding)
                    if len(distances) > 0:
                        best_idx = np.argmin(distances)
                        name = data["names"][best_idx] if distances[best_idx] <= TOLERANCE else "Unknown"
                    else:
                        name = "Unknown"
                    names.append(name)

                last_boxes = boxes
                last_names = names

            # Draw results
            for ((top, right, bottom, left), name) in zip(last_boxes, last_names):
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - start_time)
                start_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Face Recognition CLI", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()