import cv2
import mediapipe as mp
import json
import os

video_path = "input/vid4.mp4"  # use your working video
output_path = "output/pose_data.json"

# Setup
cap = cv2.VideoCapture(video_path)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

pose_data = []
frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (640, 480))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        landmarks = [
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            }
            for lm in results.pose_landmarks.landmark
        ]
        pose_data.append({
            "frame": frame_idx,
            "landmarks": landmarks
        })

    frame_idx += 1

cap.release()
pose.close()

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(pose_data, f, indent=2)

print(f"✅ Pose data saved to {output_path}")
print(f"📊 Total frames with pose: {len(pose_data)} / {frame_idx}")
