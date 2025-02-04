import cv2
import psutil
import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s}{size_name[i]}"

def load_icon(path, size=20):
    icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if icon is None:
        raise FileNotFoundError(f"Icon not found: {path}")
    icon = cv2.resize(icon, (size, size))
    return icon

def overlay_icon(frame, icon, position):
    x, y = position
    h, w = icon.shape[:2]
    
    # Handle alpha channel if present
    if icon.shape[2] == 4:
        alpha = icon[:, :, 3] / 255.0
        icon = icon[:, :, :3]  # Extract BGR channels
    else:
        alpha = 1.0
    
    # Overlay the icon on the frame
    for c in range(3):
        frame[y:y+h, x:x+w, c] = (alpha * icon[:, :, c] + 
                                  (1 - alpha) * frame[y:y+h, x:x+w, c])

# Load icons (ensure correct paths)
icons = {
    'memory': load_icon('assets/icons/memory.png'),
    'cpu': load_icon('assets/icons/cpu.png'),
    'network': load_icon('assets/icons/network.png'),
    'clock': load_icon('assets/icons/clock.png')
}

cap = cv2.VideoCapture(0)
prev_net_io = psutil.net_io_counters()
prev_time = time.time()
psutil.cpu_percent(interval=0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Overlay icons on the OpenCV frame first
    y_start = 10
    x_start = 10
    icon_size = 20
    spacing = 8

    for i, icon_key in enumerate(['memory', 'cpu', 'network', 'clock']):
        y_pos = y_start + i * (icon_size + spacing)
        overlay_icon(frame, icons[icon_key], (x_start, y_pos))

    # Convert to PIL for text drawing
    pil_im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.truetype("assets/fonts/inter.ttf", 15)

    # Get system metrics
    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent()
    current_net_io = psutil.net_io_counters()
    current_time = time.time()
    
    time_diff = current_time - prev_time
    sent_speed = (current_net_io.bytes_sent - prev_net_io.bytes_sent) / time_diff
    recv_speed = (current_net_io.bytes_recv - prev_net_io.bytes_recv) / time_diff
    prev_net_io = current_net_io
    prev_time = current_time

    current_time_str = time.strftime("%H:%M:%S")

    texts = [
        f"{mem.percent:.1f}% [{convert_size(mem.used)}/{convert_size(mem.total)}]",
        f"{cpu_percent:.1f}%",
        f"▲ {convert_size(sent_speed)}/s ▼ {convert_size(recv_speed)}/s",
        current_time_str
    ]

    # Draw text next to icons
    for i, text in enumerate(texts):
        y_pos = y_start + i * (icon_size + spacing)
        text_position = (x_start + icon_size + 5, y_pos)  # Adjusted vertical alignment
        draw.text(text_position, text, font=font, fill=(0, 255, 0))

    # Convert back to OpenCV format
    frame_with_overlays = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    
    cv2.imshow('System Monitor', frame_with_overlays)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()