import bpy
import json
import os

# === CONFIG ===
json_path = bpy.path.abspath("//output/pose_data.json")
video_width = 640
video_height = 480
scale_factor = 2.0
start_frame = 1
sphere_size = 0.04

# === Load pose data ===
with open(json_path, "r") as f:
    pose_data = json.load(f)

if not pose_data:
    raise ValueError("No pose data found!")

num_frames = len(pose_data)
num_landmarks = len(pose_data[0]["landmarks"])
print(f"✅ Loaded {num_frames} frames with {num_landmarks} landmarks per frame.")

# === Create armature ===
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.add(type='ARMATURE', enter_editmode=True)
arm = bpy.context.object
arm.name = "MediaPipeArmature"
ebones = arm.data.edit_bones

for i in range(num_landmarks):
    bone = ebones.new(f"Bone_{i}")
    bone.head = (0, 0, i * 0.05)
    bone.tail = (0, 0.1, i * 0.05)

bpy.ops.object.mode_set(mode='POSE')

# === Animate ===
for frame_data in pose_data:
    frame_num = frame_data["frame"] + start_frame
    for i, lm in enumerate(frame_data["landmarks"]):
        pb = arm.pose.bones[f"Bone_{i}"]

        # Convert normalized coords to Blender space
        x = (lm["x"] - 0.5) * video_width / 100 * scale_factor
        y = (lm["y"] - 0.5) * video_height / 100 * scale_factor
        z = -lm["z"] * scale_factor

        pb.location = (x, y, z)
        pb.keyframe_insert(data_path="location", frame=frame_num)

# === Set end frame ===
bpy.context.scene.frame_end = start_frame + num_frames

# === Create spheres at each bone and parent to bone ===
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')
spheres = []

for i in range(num_landmarks):
    bone_name = f"Bone_{i}"
    bone_head = arm.data.bones[bone_name].head_local
    world_head = arm.matrix_world @ bone_head

    bpy.ops.mesh.primitive_uv_sphere_add(radius=sphere_size, location=world_head)
    sphere = bpy.context.object
    sphere.name = f"Sphere_{i}"

    sphere.parent = arm
    sphere.parent_type = 'BONE'
    sphere.parent_bone = bone_name
    spheres.append(sphere)

# === Optional: Export
bpy.ops.object.select_all(action='DESELECT')
arm.select_set(True)
for s in spheres: s.select_set(True)
bpy.context.view_layer.objects.active = arm

export_path = bpy.path.abspath("//output/pose_vis.fbx")
bpy.ops.export_scene.fbx(
    filepath=export_path,
    use_selection=True,
    bake_anim=True,
    add_leaf_bones=False,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False
)

print(f"✅ Exported animation with skin mesh to: {export_path}")
