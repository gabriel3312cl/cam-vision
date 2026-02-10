import cv2
import numpy as np

def create_dummy_video(filename="dummy_stream.mp4", duration=5, fps=20):
    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    # Draw a moving circle (simulating a face)
    for i in range(duration * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Move simulated face from left to right
        x = int((i / (duration * fps)) * (width - 100)) + 50
        y = height // 2
        
        # Draw "Face" (Green Circle)
        cv2.circle(frame, (x, y), 40, (200, 200, 200), -1)
        
        # Draw "Eyes" and "Mouth" to trigger face detector (hopefully, though simple circles might not works with YuNet)
        # YuNet is robust but specific. 
        # Actually, let's just make a white background with a "face-like" structure or use a real sample image moved around.
        # But for simpler buffer testing, just a moving noise blob might not trigger face detection.
        # However, we can test the VideoStream logic with this.
        
        cv2.circle(frame, (x - 15, y - 10), 5, (0, 0, 0), -1)
        cv2.circle(frame, (x + 15, y - 10), 5, (0, 0, 0), -1)
        cv2.ellipse(frame, (x, y + 15), (20, 10), 0, 0, 180, (0, 0, 0), 2)

        out.write(frame)

    out.release()
    print(f"Dummy video saved to {filename}")

if __name__ == "__main__":
    create_dummy_video()
