import bpy
import json
import os

# === CONFIG ===
json_path = bpy.path.abspath("//output/pose_data.json")
character_path = bpy.path.abspath("//assets/character.fbx")
export_path = bpy.path.abspath("//output/skinned_animation.fbx")

video_width, video_height = 640, 480
scale_factor, start_frame = 1.0, 1

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

# === Animate MediaPipe bones ===
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

# === Import character FBX ===
bpy.ops.import_scene.fbx(filepath=character_path)
character_rig = [obj for obj in bpy.context.selected_objects if obj.type == 'ARMATURE'][0]
character_rig.name = "CharacterArmature"
print(f"✅ Found armature: {character_rig.name}")

# === Retarget selected bones ===
bone_map = {
    "Hips": 0,
    "Spine": 11,
    "Spine1": 12,
    "Spine2": 23,
    "Neck": 24,
    "Head": 0,
    "LeftShoulder": 11,
    "LeftArm": 13,
    "LeftForeArm": 15,
    "LeftHand": 17,
    "RightShoulder": 12,
    "RightArm": 14,
    "RightForeArm": 16,
    "RightHand": 18,
    "LeftUpLeg": 23,
    "LeftLeg": 25,
    "LeftFoot": 27,
    "RightUpLeg": 24,
    "RightLeg": 26,
    "RightFoot": 28,
}

bpy.ops.object.select_all(action='DESELECT')
bpy.context.view_layer.objects.active = character_rig
bpy.ops.object.mode_set(mode='POSE')

for tgt_name_base, src_idx in bone_map.items():
    tgt_name = f"mixamorig:{tgt_name_base}"
    if tgt_name in character_rig.pose.bones:
        tgt_bone = character_rig.pose.bones[tgt_name]
        src_bone = arm.pose.bones.get(f"Bone_{src_idx}")
        if src_bone:
            loc = tgt_bone.constraints.new(type='COPY_LOCATION')
            loc.target = arm
            loc.subtarget = src_bone.name
            rot = tgt_bone.constraints.new(type='COPY_ROTATION')
            rot.target = arm
            rot.subtarget = src_bone.name
            print(f"🔗 Retargeted {tgt_name} to Bone_{src_idx}")
        else:
            print(f"⚠️ Source Bone_{src_idx} not found")
    else:
        print(f"⚠️ Target bone {tgt_name} not found")

bpy.ops.object.mode_set(mode='OBJECT')

# === Export final animated FBX ===
bpy.ops.object.select_all(action='DESELECT')
character_rig.select_set(True)
bpy.context.view_layer.objects.active = character_rig

print(f"\nFBX export starting... '{export_path}'")
bpy.ops.export_scene.fbx(
    filepath=export_path,
    use_selection=True,
    bake_anim=True,
    add_leaf_bones=False,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False
)
print(f"✅ Exported character animation to: {export_path}")
