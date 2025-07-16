import cv2
import mediapipe as mp
import os

video_path = "input/vid4.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError(f"❌ Video not found: {video_path}")

cap = cv2.VideoCapture(video_path)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

success_count = 0
total_frames = 0

while total_frames < 50 and cap.isOpened():  # test first 50 frames
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (640, 480))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        print(f"✅ Pose detected in frame {total_frames}")
        success_count += 1
    else:
        print(f"❌ No pose detected in frame {total_frames}")

    total_frames += 1

cap.release()
pose.close()
print(f"\n🎯 Summary: {success_count}/{total_frames} frames had valid poses.")
