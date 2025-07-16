import cv2
import mediapipe as mp
import json
import os


video_path = "../input/inp_vid.mp4"  
output_path = "pose_data.json"
cap = cv2.VideoCapture(video_path)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

pose_data = []

frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

 
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            })
        pose_data.append({
            'frame': frame_idx,
            'landmarks': landmarks
        })

    frame_idx += 1

cap.release()
pose.close()



if os.path.dirname(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(pose_data, f, indent=2)

print(f"Pose data saved to {output_path}")
print("Checking if file exists...")
print(os.path.exists(output_path))