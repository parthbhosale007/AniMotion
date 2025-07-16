import bpy
import json
import os

# === CONFIG ===
json_path = bpy.path.abspath("//output/pose_data.json")  # relative to .blend file
video_width = 640
video_height = 480
scale_factor = 2.0  # adjust for Blender unit scale
start_frame = 1

# === Load pose data ===
with open(json_path, "r") as f:
    pose_data = json.load(f)

if not pose_data:
    raise ValueError("No pose data found in JSON!")

num_landmarks = len(pose_data[0]["landmarks"])
print(f"✅ Loaded {len(pose_data)} frames with {num_landmarks} landmarks per frame.")

# === Create armature ===
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.add(type='ARMATURE', enter_editmode=True)
armature = bpy.context.object
armature.name = "MediaPipeArmature"
edit_bones = armature.data.edit_bones

for i in range(num_landmarks):
    bone = edit_bones.new(f"Bone_{i}")
    bone.head = (0, 0, i * 0.05)  # stacked vertically to avoid zero-length bones
    bone.tail = (0, 0.1, i * 0.05)

bpy.ops.object.mode_set(mode='POSE')

# === Animate ===
for frame_data in pose_data:
    frame_num = frame_data["frame"] + start_frame
    for i, lm in enumerate(frame_data["landmarks"]):
        bone = armature.pose.bones[f"Bone_{i}"]

        # Convert normalized coordinates to Blender space
        x = (lm["x"] - 0.5) * video_width / 100 * scale_factor
        y = (lm["y"] - 0.5) * video_height / 100 * scale_factor
        z = -lm["z"] * scale_factor  # Invert for Blender coordinate system

        bone.location = (x, y, z)
        bone.keyframe_insert(data_path="location", frame=frame_num)

print("✅ Animation imported successfully!")
