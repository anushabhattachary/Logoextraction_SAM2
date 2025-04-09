# scripts/extract_frames.py

import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % int(cap.get(cv2.CAP_PROP_FPS) // frame_rate) == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1
    cap.release()
    print(f"🎞️ Saved {saved} frames to {output_dir}")
