
import cv2
import os

def extract_frames(video_path, output_folder, frame_name="frame"):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  
        # Save frame as PNG
        frame_filename = os.path.join(output_folder, f"{frame_name}_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        print(f'\rFrame {frame_count} done ', end='', flush=True)
        frame_count += 1
        
    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")

extract_frames(r"wasd.mp4", r"frames/")
