import bpy
import json
import os

# === CONFIG ===
json_path = bpy.path.abspath("//output/pose_data.json")
character_path = bpy.path.abspath("//assets/T-posee.fbx")
export_path = bpy.path.abspath("//output/skinned_animation.fbx")

video_width, video_height = 640, 480
scale_factor, start_frame = 2.0, 1

# === Load pose data ===
with open(json_path, "r") as f:
    pose_data = json.load(f)

num_frames = len(pose_data)
num_landmarks = len(pose_data[0]["landmarks"])
print(f"✅ Loaded {num_frames} frames with {num_landmarks} landmarks per frame.")

# === Create MediaPipe armature ===
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.armature_add(enter_editmode=True)
arm = bpy.context.object
arm.name = "MediaPipeArmature"
ebones = arm.data.edit_bones

for i in range(num_landmarks):
    bone = ebones.new(f"Bone_{i}")
    bone.head = (0, 0, i * 0.05)
    bone.tail = (0, 0.1, i * 0.05)

bpy.ops.object.mode_set(mode='POSE')

# === Animate bones ===
for frame_data in pose_data:
    frame_num = frame_data["frame"] + start_frame
    for i, lm in enumerate(frame_data["landmarks"]):
        pb = arm.pose.bones[f"Bone_{i}"]
        x = (lm["x"] - 0.5) * video_width / 100 * scale_factor
        y = (lm["y"] - 0.5) * video_height / 100 * scale_factor
        z = -lm["z"] * scale_factor
        pb.location = (x, y, z)
        pb.keyframe_insert(data_path="location", frame=frame_num)

bpy.context.scene.frame_end = start_frame + num_frames

# === Import character mesh ===
bpy.ops.import_scene.fbx(filepath=character_path)
character_rig = [obj for obj in bpy.context.selected_objects if obj.type == 'ARMATURE'][0]
character_rig.name = "CharacterArmature"

# === Retarget animation via constraints ===
bpy.ops.object.select_all(action='DESELECT')
bpy.context.view_layer.objects.active = character_rig
bpy.ops.object.mode_set(mode='POSE')

# Simple 1:1 bone mapping if same count — you can customize mapping here
for i in range(min(len(character_rig.pose.bones), num_landmarks)):
    src_bone = arm.pose.bones[f"Bone_{i}"]
    tgt_bone = character_rig.pose.bones[i]

    con = tgt_bone.constraints.new(type='COPY_LOCATION')
    con.target = arm
    con.subtarget = src_bone.name

    con = tgt_bone.constraints.new(type='COPY_ROTATION')
    con.target = arm
    con.subtarget = src_bone.name

bpy.ops.object.mode_set(mode='OBJECT')

# === Export final animated mesh ===
bpy.ops.object.select_all(action='DESELECT')
character_rig.select_set(True)
bpy.context.view_layer.objects.active = character_rig

bpy.ops.export_scene.fbx(
    filepath=export_path,
    use_selection=True,
    bake_anim=True,
    add_leaf_bones=False,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False
)

print(f"✅ Exported character animation to: {export_path}")
