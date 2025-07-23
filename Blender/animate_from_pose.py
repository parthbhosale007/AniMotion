import bpy, json, os

# === CONFIG ===
json_path = bpy.path.abspath("//output/pose_data.json")
video_width = 640
video_height = 480
scale_factor = 2.0
start_frame = 1
sphere_size = 0.02      # radius of each joint-sphere

# === Load pose data ===
with open(json_path, "r") as f:
    pose_data = json.load(f)
if not pose_data:
    raise ValueError("No pose data found!")

num_landmarks = len(pose_data[0]["landmarks"])
print(f"✅ Loaded {len(pose_data)} frames with {num_landmarks} landmarks per frame.")

# === Create armature ===
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.add(type='ARMATURE', enter_editmode=True)
arm = bpy.context.object
arm.name = "MediaPipeArmature"
ebones = arm.data.edit_bones

for i in range(num_landmarks):
    b = ebones.new(f"Bone_{i}")
    b.head = (0, 0, i * 0.05)
    b.tail = (0, 0.1, i * 0.05)

bpy.ops.object.mode_set(mode='OBJECT')

# === Create helper spheres and parent to bones ===
spheres = []
for i, eb in enumerate(ebones):
    # compute world-space head location
    head_local = eb.head.copy()
    world_loc = arm.matrix_world @ head_local
    
    # add sphere at joint location
    bpy.ops.mesh.primitive_uv_sphere_add(radius=sphere_size, location=world_loc)
    sph = bpy.context.object
    sph.name = f"Joint_{i}"
    
    # parent to armature bone
    sph.parent = arm
    sph.parent_type = 'BONE'
    sph.parent_bone = eb.name
    
    spheres.append(sph)

# === Pose-mode animation ===
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')

for frame_info in pose_data:
    frame = frame_info["frame"] + start_frame
    for i, lm in enumerate(frame_info["landmarks"]):
        pb = arm.pose.bones[f"Bone_{i}"]
        # convert to Blender space
        x = (lm["x"] - 0.5) * video_width / 100 * scale_factor
        y = (lm["y"] - 0.5) * video_height / 100 * scale_factor
        z = -lm["z"] * scale_factor
        
        pb.location = (x, y, z)
        pb.keyframe_insert(data_path="location", frame=frame)

# === (Optional) Export FBX ===
bpy.ops.object.mode_set(mode='OBJECT')
export_path = bpy.path.abspath("//output/pose_vis.fbx")
# select only armature and spheres
bpy.ops.object.select_all(action='DESELECT')
arm.select_set(True)
for sph in spheres: sph.select_set(True)
bpy.context.view_layer.objects.active = arm

bpy.ops.export_scene.fbx(
    filepath=export_path,
    use_selection=True,
    bake_anim=True,
    add_leaf_bones=False,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False
)
print(f"✅ Exported visualized animation to {export_path}")
