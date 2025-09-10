# Blender/auto_oneclick.py - ROBUST VERSION
# One-click, headless: JSON -> Mixamo rig baked -> FBX + MP4
# Requires Blender 4.x (bundled NumPy OK). No SciPy needed.

import bpy, json, os, math
import numpy as np
import mathutils

# ------------------- CONFIG -------------------
JSON_PATH   = bpy.path.abspath("//output/pose_data.json")
CHAR_FBX    = bpy.path.abspath("//assets/character.fbx")   # Mixamo character (T-pose)
OUT_FBX     = bpy.path.abspath("//output/skinned_animation.fbx")
OUT_MP4     = bpy.path.abspath("//output/anim.mp4")

VIDEO_W, VIDEO_H = 640, 480
SCALE          = 1.0  # Reduced scale to start
START_FRAME    = 1
SMOOTH_WINDOW  = 5   # Reduced smoothing
FPS            = 30
RENDER_PREVIEW = True
DEBUG_MODE     = True  # Enable detailed logging
# ------------------------------------------------

def debug_print(msg):
    if DEBUG_MODE:
        print(f"🔍 DEBUG: {msg}")

def safe_clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for b in bpy.data.actions: 
        bpy.data.actions.remove(b)
    debug_print("Scene cleared")

def load_pose_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not data:
        raise ValueError("Pose JSON empty.")
    debug_print(f"Loaded {len(data)} frames from JSON")
    return data

def moving_average(arr, win):
    if win < 3 or win % 2 == 0: 
        return arr
    k = np.ones(win) / win
    pad = win // 2
    arr_p = np.pad(arr, ((pad,pad),(0,0),(0,0)), mode='edge')
    out = np.empty_like(arr)
    for i in range(arr.shape[2]):  # x,y,z
        out[:,:,i] = np.apply_along_axis(lambda m: np.convolve(m, k, mode='valid'), 0, arr_p[:,:,i])
    return out

def norm_to_world(lm, sx=1, sy=1, sz=1):
    # Improved coordinate conversion - MediaPipe to Blender world space
    # MediaPipe: x=[0,1] left-to-right, y=[0,1] top-to-bottom, z=depth
    # Blender: x=left-right, y=front-back, z=up-down
    
    x = (lm[0] - 0.5) * 10 * SCALE * sx  # Scale to reasonable world size
    z = (0.5 - lm[1]) * 10 * SCALE * sy  # Flip Y to Z (MediaPipe Y down = Blender Z up)
    y = -lm[2] * 10 * SCALE * sz         # MediaPipe Z depth = Blender Y front/back
    return (x, y, z)

def mid(a, b):
    return ((a[0]+b[0])/2.0, (a[1]+b[1])/2.0, (a[2]+b[2])/2.0)

def ensure_camera_light():
    # Clear existing cameras and lights for clean setup
    for obj in list(bpy.data.objects):
        if obj.type in ('CAMERA', 'LIGHT'):
            bpy.data.objects.remove(obj)
    
    # Create camera
    cam_data = bpy.data.cameras.new("AutoCamera")
    cam = bpy.data.objects.new("AutoCamera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (15, -15, 10)
    cam.rotation_euler = (1.1, 0, 0.78)
    bpy.context.scene.camera = cam
    debug_print(f"Camera created at {cam.location}")

    # Create sun light
    light_data = bpy.data.lights.new("AutoLight", type='SUN')
    light_data.energy = 5
    light = bpy.data.objects.new("AutoLight", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (10, -10, 15)
    debug_print("Sun light created")

def import_mixamo_robust(path):
    """Import Mixamo FBX and return armature + mesh objects"""
    debug_print(f"Importing FBX: {path}")
    pre_objs = set(bpy.data.objects)
    
    try:
        bpy.ops.import_scene.fbx(
            filepath=path,
            use_manual_orientation=True,
            global_scale=1.0,
            bake_space_transform=False
        )
    except RuntimeError as e:
        if "Version 6100 unsupported" in str(e):
            print("❌ FBX version too old! Please re-download from Mixamo.")
        raise e
    
    post_objs = set(bpy.data.objects)
    new_objs = list(post_objs - pre_objs)
    
    # Find armature and meshes
    armature = None
    meshes = []
    
    for obj in new_objs:
        if obj.type == 'ARMATURE':
            armature = obj
        elif obj.type == 'MESH':
            meshes.append(obj)
    
    if not armature:
        raise RuntimeError("No armature found in FBX")
    
    armature.name = "CharacterArmature"
    debug_print(f"Found armature: {armature.name}")
    debug_print(f"Found {len(meshes)} mesh objects")
    
    # Ensure meshes are properly bound to armature
    for mesh in meshes:
        mesh.name = f"CharacterMesh_{mesh.name}"
        # Check for armature modifier
        has_armature_mod = any(mod.type == 'ARMATURE' for mod in mesh.modifiers)
        if not has_armature_mod:
            arm_mod = mesh.modifiers.new("Armature", 'ARMATURE')
            arm_mod.object = armature
            debug_print(f"Added armature modifier to {mesh.name}")
        
        # Parent mesh to armature (use standard OBJECT parenting)
        mesh.parent = armature
        mesh.parent_type = 'ARMATURE'
        debug_print(f"Parented {mesh.name} to armature")
    
    return armature, meshes

def get_bone(armature, name):
    """Get bone from armature pose"""
    return armature.pose.bones.get(name)

def resolve_bone(armature, name):
    """Try to find bone with various naming conventions"""
    # Try with mixamorig: prefix
    bone = get_bone(armature, name)
    if bone:
        return bone
    
    # Try without prefix
    simple_name = name.split(":")[-1] if ":" in name else name
    bone = get_bone(armature, simple_name)
    if bone:
        return bone
    
    # Try common variations
    variations = [
        name.replace("mixamorig:", ""),
        name.replace("_", ""),
        name.lower(),
        name.upper()
    ]
    
    for var in variations:
        bone = get_bone(armature, var)
        if bone:
            return bone
    
    return None

def calculate_bone_rotation(from_pos, to_pos, rest_direction=(0, 1, 0)):
    """Calculate rotation quaternion to align bone from rest pose to target direction"""
    direction = to_pos - from_pos
    if direction.length < 0.001:
        return mathutils.Quaternion()  # Identity quaternion
    
    direction.normalize()
    rest_vec = mathutils.Vector(rest_direction).normalized()
    
    # Calculate rotation from rest direction to target direction
    rotation_quat = rest_vec.rotation_difference(direction)
    return rotation_quat

def apply_animation_robust(armature, landmark_data, frame_count):
    """Apply animation using improved bone mapping and rotation calculation"""
    debug_print(f"Applying animation to {frame_count} frames")
    
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # Clear existing animation
    if armature.animation_data:
        armature.animation_data_clear()
    
    # MediaPipe landmark indices
    mp_indices = {
        'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
        'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
        'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
        'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16, 'left_pinky': 17, 'right_pinky': 18,
        'left_index': 19, 'right_index': 20, 'left_thumb': 21, 'right_thumb': 22,
        'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30,
        'left_foot_index': 31, 'right_foot_index': 32
    }
    
    # Enhanced bone mappings with better rest directions
    bone_mappings = [
        # Core bones
        ("mixamorig:Hips", 'hip_center', 'shoulder_center', (0, 0, 1)),
        ("mixamorig:Spine", 'hip_center', 'shoulder_center', (0, 0, 1)),
        ("mixamorig:Spine1", 'hip_center', 'shoulder_center', (0, 0, 1)),
        ("mixamorig:Neck", 'shoulder_center', 'nose', (0, 0, 1)),
        ("mixamorig:Head", 'shoulder_center', 'nose', (0, 0, 1)),
        
        # Arms
        ("mixamorig:LeftShoulder", 'left_shoulder', 'left_elbow', (-1, 0, 0)),
        ("mixamorig:LeftArm", 'left_shoulder', 'left_elbow', (-1, 0, 0)),
        ("mixamorig:LeftForeArm", 'left_elbow', 'left_wrist', (-1, 0, 0)),
        
        ("mixamorig:RightShoulder", 'right_shoulder', 'right_elbow', (1, 0, 0)),
        ("mixamorig:RightArm", 'right_shoulder', 'right_elbow', (1, 0, 0)),
        ("mixamorig:RightForeArm", 'right_elbow', 'right_wrist', (1, 0, 0)),
        
        # Legs
        ("mixamorig:LeftUpLeg", 'left_hip', 'left_knee', (0, 0, -1)),
        ("mixamorig:LeftLeg", 'left_knee', 'left_ankle', (0, 0, -1)),
        ("mixamorig:LeftFoot", 'left_ankle', 'left_foot_index', (0, 1, 0)),
        
        ("mixamorig:RightUpLeg", 'right_hip', 'right_knee', (0, 0, -1)),
        ("mixamorig:RightLeg", 'right_knee', 'right_ankle', (0, 0, -1)),
        ("mixamorig:RightFoot", 'right_ankle', 'right_foot_index', (0, 1, 0)),
    ]
    
    # Find available bones
    available_bones = []
    for bone_name, from_key, to_key, rest_dir in bone_mappings:
        bone = resolve_bone(armature, bone_name)
        if bone:
            available_bones.append((bone_name, from_key, to_key, rest_dir, bone))
            debug_print(f"✅ Found bone: {bone.name}")
        else:
            debug_print(f"❌ Missing bone: {bone_name}")
    
    if not available_bones:
        print("💥 ERROR: No bones found!")
        print("Available bones in armature:")
        for bone in armature.pose.bones[:10]:  # Show first 10
            print(f"  - {bone.name}")
        return
    
    debug_print(f"Will animate {len(available_bones)} bones")
    
    # Process all frames
    for frame_idx in range(frame_count):
        frame_num = START_FRAME + frame_idx
        bpy.context.scene.frame_set(frame_num)
        
        if frame_idx % 30 == 0:  # Progress every 30 frames
            debug_print(f"Processing frame {frame_num}/{frame_count}")
        
        # Get landmarks for this frame
        landmarks = landmark_data[frame_idx]
        
        # Calculate derived positions
        hip_center = mid(landmarks[mp_indices['left_hip']], landmarks[mp_indices['right_hip']])
        shoulder_center = mid(landmarks[mp_indices['left_shoulder']], landmarks[mp_indices['right_shoulder']])
        
        # Create position lookup
        positions = {
            'hip_center': mathutils.Vector(hip_center),
            'shoulder_center': mathutils.Vector(shoulder_center),
            'nose': mathutils.Vector(landmarks[mp_indices['nose']]),
            'left_shoulder': mathutils.Vector(landmarks[mp_indices['left_shoulder']]),
            'right_shoulder': mathutils.Vector(landmarks[mp_indices['right_shoulder']]),
            'left_elbow': mathutils.Vector(landmarks[mp_indices['left_elbow']]),
            'right_elbow': mathutils.Vector(landmarks[mp_indices['right_elbow']]),
            'left_wrist': mathutils.Vector(landmarks[mp_indices['left_wrist']]),
            'right_wrist': mathutils.Vector(landmarks[mp_indices['right_wrist']]),
            'left_hip': mathutils.Vector(landmarks[mp_indices['left_hip']]),
            'right_hip': mathutils.Vector(landmarks[mp_indices['right_hip']]),
            'left_knee': mathutils.Vector(landmarks[mp_indices['left_knee']]),
            'right_knee': mathutils.Vector(landmarks[mp_indices['right_knee']]),
            'left_ankle': mathutils.Vector(landmarks[mp_indices['left_ankle']]),
            'right_ankle': mathutils.Vector(landmarks[mp_indices['right_ankle']]),
            'left_foot_index': mathutils.Vector(landmarks[mp_indices['left_foot_index']]),
            'right_foot_index': mathutils.Vector(landmarks[mp_indices['right_foot_index']]),
        }
        
        # Apply bone transformations
        for bone_name, from_key, to_key, rest_dir, bone in available_bones:
            from_pos = positions.get(from_key)
            to_pos = positions.get(to_key)
            
            if not from_pos or not to_pos:
                continue
            
            # For hips, also set location
            if "Hips" in bone_name:
                hip_local = armature.matrix_world.inverted() @ positions['hip_center']
                bone.location = hip_local
                bone.keyframe_insert("location", frame=frame_num)
            
            # Calculate and apply rotation
            rotation = calculate_bone_rotation(from_pos, to_pos, rest_dir)
            bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = rotation
            bone.keyframe_insert("rotation_quaternion", frame=frame_num)
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Set scene frame range
    bpy.context.scene.frame_start = START_FRAME
    bpy.context.scene.frame_end = START_FRAME + frame_count - 1
    debug_print(f"Animation complete! Frames: {bpy.context.scene.frame_start}-{bpy.context.scene.frame_end}")

def bake_animation(armature, start_frame, end_frame):
    """Bake the animation"""
    debug_print(f"Baking animation frames {start_frame}-{end_frame}")
    bpy.context.view_layer.objects.active = armature
    try:
        bpy.ops.nla.bake(
            frame_start=start_frame,
            frame_end=end_frame,
            only_selected=False,
            visual_keying=True,
            clear_constraints=True,
            use_current_action=True,
            bake_types={'POSE'}
        )
        debug_print("Animation baked successfully")
    except Exception as e:
        debug_print(f"Baking failed: {e}")

def setup_render(fps, out_mp4, frame_end):
    s = bpy.context.scene
    s.render.engine = 'BLENDER_EEVEE_NEXT' 
    s.render.resolution_x = 1280
    s.render.resolution_y = 720
    s.render.fps = fps
    s.frame_start = START_FRAME
    s.frame_end = START_FRAME + frame_end - 1
    s.render.image_settings.file_format = 'FFMPEG'
    s.render.ffmpeg.format = 'MPEG4'
    s.render.ffmpeg.codec = 'H264'
    s.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    s.render.ffmpeg.ffmpeg_preset = 'GOOD'
    s.render.filepath = out_mp4
    debug_print("Render settings configured")

def check_inputs():
    """Check if input files exist"""
    json_abs = bpy.path.abspath(JSON_PATH)
    fbx_abs = bpy.path.abspath(CHAR_FBX)
    
    print("\n=== INPUT CHECK ===")
    print(f"JSON: {json_abs} {'✅' if os.path.exists(json_abs) else '❌'}")
    print(f"FBX:  {fbx_abs} {'✅' if os.path.exists(fbx_abs) else '❌'}")
    print(f"Blend: {bpy.data.filepath or '(unsaved)'}")
    print("==================\n")
    
    if not os.path.exists(json_abs):
        raise FileNotFoundError(f"JSON not found: {json_abs}")
    if not os.path.exists(fbx_abs):
        raise FileNotFoundError(f"FBX not found: {fbx_abs}")

# ================= MAIN PIPELINE =================

try:
    print("🚀 Starting Blender animation pipeline...")
    
    # 1. Check inputs
    check_inputs()
    
    # 2. Clear scene and setup
    safe_clear_scene()
    ensure_camera_light()
    
    # 3. Load and process pose data
    pose_data = load_pose_json(JSON_PATH)
    frame_count = len(pose_data)
    landmark_count = len(pose_data[0]["landmarks"])
    
    debug_print(f"Processing {frame_count} frames, {landmark_count} landmarks each")
    
    # Convert to numpy array and smooth
    arr = np.array([[[lm["x"], lm["y"], lm["z"]] for lm in f["landmarks"]] for f in pose_data])
    arr_smoothed = moving_average(arr, SMOOTH_WINDOW)
    
    # Convert to world coordinates
    landmark_world_data = {}
    for frame_idx in range(frame_count):
        frame_landmarks = []
        for lm_idx in range(landmark_count):
            world_pos = norm_to_world(arr_smoothed[frame_idx, lm_idx])
            frame_landmarks.append(world_pos)
        landmark_world_data[frame_idx] = frame_landmarks
    
    # 4. Import character
    armature, meshes = import_mixamo_robust(CHAR_FBX)
    
    # 5. Apply animation
    apply_animation_robust(armature, landmark_world_data, frame_count)
    
    # 6. Bake animation
    bake_animation(armature, START_FRAME, START_FRAME + frame_count - 1)
    
    # 7. Export FBX
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    for mesh in meshes:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature
    
    debug_print(f"Exporting FBX with {len(meshes)} meshes...")
    bpy.ops.export_scene.fbx(
        filepath=OUT_FBX,
        use_selection=True,
        bake_anim=True,
        add_leaf_bones=False
    )
    print(f"✅ FBX exported: {OUT_FBX}")
    
    # 8. Render preview
    if RENDER_PREVIEW:
        setup_render(FPS, OUT_MP4, frame_count)
        print(f"🎥 Rendering preview...")
        bpy.ops.render.render(animation=True)
        print(f"✅ Video rendered: {OUT_MP4}")
    
    print("🎉 Pipeline completed successfully!")

except Exception as e:
    print(f"💥 ERROR: {e}")
    import traceback
    traceback.print_exc()
    raise