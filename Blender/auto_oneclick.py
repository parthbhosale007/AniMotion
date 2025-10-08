# Blender/auto_oneclick.py
# One-click, headless: JSON -> Mixamo rig baked -> FBX + MP4
# Requires Blender 4.x (bundled NumPy OK)

import bpy, json, os, math
import numpy as np
import mathutils

# ------------------- CONFIG -------------------
JSON_PATH   = bpy.path.abspath("//output/pose_data.json")
CHAR_FBX    = bpy.path.abspath("//assets/Remy.fbx")   # Mixamo character (T-pose)
OUT_FBX     = bpy.path.abspath("//output/skinned_animation.fbx")
OUT_MP4     = bpy.path.abspath("//output/anim.mp4")

VIDEO_W, VIDEO_H = 640, 480
SCALE          = 0.05  # Reduced scale for better fitting
START_FRAME    = 1
SMOOTH_WINDOW  = 5     # Smaller smoothing window
FPS            = 30
RENDER_PREVIEW = True
# ------------------------------------------------

def safe_clear_scene():
    """Clear scene properly"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # Clear orphan data
    for block in [bpy.data.actions, bpy.data.meshes, bpy.data.materials]:
        for item in block:
            if not item.users:
                block.remove(item)

def load_pose_json(path):
    """Load and validate pose JSON"""
    with open(path, "r") as f:
        data = json.load(f)
    
    if not data:
        raise ValueError("Pose JSON empty.")
    
    print(f"📁 Loaded {len(data)} frames, {len(data[0]['landmarks'])} landmarks each")
    return data

def moving_average(arr, win):
    """Simple smoothing"""
    if win < 3:
        return arr
    
    padded = np.pad(arr, ((win//2, win//2), (0,0), (0,0)), mode='edge')
    smoothed = np.zeros_like(arr)
    
    for i in range(arr.shape[0]):
        smoothed[i] = np.mean(padded[i:i+win], axis=0)
    
    return smoothed

def norm_to_world(lm, scale_x=1, scale_y=1, scale_z=1):
    """Convert normalized MediaPipe coordinates to world space"""
    x = (lm[0] - 0.5) * VIDEO_W * SCALE * scale_x
    y = (lm[1] - 0.5) * VIDEO_H * SCALE * scale_y  
    z = lm[2] * VIDEO_W * SCALE * scale_z  # Positive Z up
    
    return (x, y, z)

def ensure_camera_light():
    """Setup camera and lighting"""
    # Camera
    if "AutoCamera" not in bpy.data.objects:
        bpy.ops.object.camera_add(location=(8, -8, 6))
        cam = bpy.context.object
        cam.name = "AutoCamera"
    else:
        cam = bpy.data.objects["AutoCamera"]
    
    cam.location = (8, -8, 6)
    cam.rotation_euler = (1.0, 0, 0.8)
    bpy.context.scene.camera = cam
    
    # Light
    if "AutoLight" not in bpy.data.objects:
        bpy.ops.object.light_add(type='SUN', location=(5, -5, 8))
        light = bpy.context.object
        light.name = "AutoLight"
        light.data.energy = 3.0
    
    print("✅ Camera & light setup complete")

def import_mixamo_character(fbx_path):
    """Import Mixamo character and find armature"""
    # Store existing objects
    existing_objects = set(bpy.data.objects)
    
    # Import FBX
    try:
        bpy.ops.import_scene.fbx(filepath=fbx_path)
    except Exception as e:
        print(f"❌ FBX import failed: {e}")
        return None
    
    # Find new objects
    new_objects = set(bpy.data.objects) - existing_objects
    
    # Look for armature
    armature = None
    for obj in new_objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break
    
    # If not found in new objects, search all armatures
    if not armature:
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE':
                armature = obj
                break
    
    if armature:
        armature.name = "Mixamo_Rig"
        print(f"✅ Imported armature: {armature.name}")
        
        # Find and parent meshes
        meshes = []
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj != armature:
                # Ensure armature modifier exists
                has_armature_mod = False
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE':
                        mod.object = armature
                        has_armature_mod = True
                        break
                
                if not has_armature_mod:
                    mod = obj.modifiers.new("Armature", 'ARMATURE')
                    mod.object = armature
                
                # Parent to armature
                obj.parent = armature
                meshes.append(obj)
                print(f"   └─ Linked mesh: {obj.name}")
        
        return armature
    else:
        print("❌ No armature found in FBX file")
        return None

def get_bone(armature, bone_name):
    """Safely get bone from armature with fallback names"""
    # Try exact name
    if bone_name in armature.pose.bones:
        return armature.pose.bones[bone_name]
    
    # Try without mixamorig: prefix
    simple_name = bone_name.replace("mixamorig:", "")
    if simple_name in armature.pose.bones:
        return armature.pose.bones[simple_name]
    
    # Try common variations
    variations = [
        bone_name,
        bone_name.replace("mixamorig:", ""),
        bone_name.replace("mixamorig:", "mixamorig2:"),
        bone_name.lower(),
        bone_name.upper()
    ]
    
    for var in variations:
        if var in armature.pose.bones:
            return armature.pose.bones[var]
    
    return None

def calculate_bone_rotation(start_pos, end_pos, bone_axis=(0, 1, 0)):
    """Calculate rotation to align bone with target direction"""
    if (end_pos - start_pos).length < 0.001:
        return mathutils.Quaternion()  # Identity
    
    target_direction = (end_pos - start_pos).normalized()
    bone_axis_vector = mathutils.Vector(bone_axis).normalized()
    
    return bone_axis_vector.rotation_difference(target_direction)

def apply_simple_animation(armature, landmark_data_world, frame_count):
    """Apply animation using simple bone direction approach"""
    print("🎬 Starting animation application...")
    
    # Set pose mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # Clear existing animation
    armature.animation_data_clear()
    
    # MediaPipe landmark indices
    MP_INDICES = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28
    }
    
    # Bone mappings - simplified but complete
    BONE_MAPPINGS = [
        # Spine chain
        ("Hips", 'hip_center', None),  # Location only
        ("Spine", 'hip_center', 'spine_mid'),
        ("Spine1", 'spine_mid', 'spine_upper'),
        ("Spine2", 'spine_upper', 'neck_base'),
        
        # Left Arm
        ("LeftArm", 'left_shoulder', 'left_elbow'),
        ("LeftForeArm", 'left_elbow', 'left_wrist'),
        
        # Right Arm
        ("RightArm", 'right_shoulder', 'right_elbow'),
        ("RightForeArm", 'right_elbow', 'right_wrist'),
        
        # Left Leg
        ("LeftUpLeg", 'left_hip', 'left_knee'),
        ("LeftLeg", 'left_knee', 'left_ankle'),
        
        # Right Leg
        ("RightUpLeg", 'right_hip', 'right_knee'),
        ("RightLeg", 'right_knee', 'right_ankle'),
        
        # Head
        ("Neck", 'neck_base', 'head_base'),
        ("Head", 'head_base', 'head_top'),
    ]
    
    # Verify bone existence
    valid_bones = []
    for bone_name, start_point, end_point in BONE_MAPPINGS:
        bone = get_bone(armature, bone_name)
        if bone:
            valid_bones.append((bone, bone_name, start_point, end_point))
            print(f"   ✅ Bone found: {bone_name}")
        else:
            print(f"   ❌ Bone missing: {bone_name}")
    
    print(f"🎯 Animating {len(valid_bones)} bones over {frame_count} frames")
    
    # Animation loop
    for frame_idx in range(frame_count):
        bpy.context.scene.frame_set(START_FRAME + frame_idx)
        
        # Get current frame landmarks
        frame_landmarks = landmark_data_world[frame_idx]
        
        # Calculate derived positions
        left_shoulder = mathutils.Vector(frame_landmarks[MP_INDICES['left_shoulder']])
        right_shoulder = mathutils.Vector(frame_landmarks[MP_INDICES['right_shoulder']])
        left_hip = mathutils.Vector(frame_landmarks[MP_INDICES['left_hip']])
        right_hip = mathutils.Vector(frame_landmarks[MP_INDICES['right_hip']])
        nose = mathutils.Vector(frame_landmarks[MP_INDICES['nose']])
        
        # Derived positions
        hip_center = (left_hip + right_hip) * 0.5
        shoulder_center = (left_shoulder + right_shoulder) * 0.5
        spine_mid = (hip_center + shoulder_center) * 0.5
        spine_upper = shoulder_center
        neck_base = shoulder_center + (nose - shoulder_center) * 0.3
        head_base = shoulder_center + (nose - shoulder_center) * 0.6
        head_top = nose
        
        positions = {
            'left_shoulder': left_shoulder,
            'right_shoulder': right_shoulder,
            'left_elbow': mathutils.Vector(frame_landmarks[MP_INDICES['left_elbow']]),
            'right_elbow': mathutils.Vector(frame_landmarks[MP_INDICES['right_elbow']]),
            'left_wrist': mathutils.Vector(frame_landmarks[MP_INDICES['left_wrist']]),
            'right_wrist': mathutils.Vector(frame_landmarks[MP_INDICES['right_wrist']]),
            'left_hip': left_hip,
            'right_hip': right_hip,
            'left_knee': mathutils.Vector(frame_landmarks[MP_INDICES['left_knee']]),
            'right_knee': mathutils.Vector(frame_landmarks[MP_INDICES['right_knee']]),
            'left_ankle': mathutils.Vector(frame_landmarks[MP_INDICES['left_ankle']]),
            'right_ankle': mathutils.Vector(frame_landmarks[MP_INDICES['right_ankle']]),
            'hip_center': hip_center,
            'spine_mid': spine_mid,
            'spine_upper': spine_upper,
            'neck_base': neck_base,
            'head_base': head_base,
            'head_top': head_top,
        }
        
        # Apply to each bone
        for bone, bone_name, start_point, end_point in valid_bones:
            bone.rotation_mode = 'QUATERNION'
            
            if bone_name == "Hips":
                # Hips get location only
                bone_location = armature.matrix_world.inverted() @ hip_center
                bone.location = bone_location
                bone.keyframe_insert(data_path="location", frame=START_FRAME + frame_idx)
                bone.rotation_quaternion = mathutils.Quaternion()
            elif end_point:  # Bones with direction
                start_pos = positions.get(start_point)
                end_pos = positions.get(end_point)
                
                if start_pos and end_pos:
                    rotation = calculate_bone_rotation(start_pos, end_pos)
                    bone.rotation_quaternion = rotation
                else:
                    bone.rotation_quaternion = mathutils.Quaternion()  # Identity
            else:
                bone.rotation_quaternion = mathutils.Quaternion()  # Identity
            
            bone.keyframe_insert(data_path="rotation_quaternion", frame=START_FRAME + frame_idx)
    
    # Set scene frame range
    bpy.context.scene.frame_start = START_FRAME
    bpy.context.scene.frame_end = START_FRAME + frame_count - 1
    print(f"📊 Frame range: {bpy.context.scene.frame_start} - {bpy.context.scene.frame_end}")

def bake_animation(armature, start_frame, end_frame):
    """Bake animation to keyframes"""
    print("🍳 Baking animation...")
    
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # Select all bones
    bpy.ops.pose.select_all(action='SELECT')
    
    # Bake animation
    bpy.ops.nla.bake(
        frame_start=start_frame,
        frame_end=end_frame,
        step=1,
        only_selected=True,
        visual_keying=True,
        clear_constraints=False,
        clear_parents=False,
        use_current_action=True,
        bake_types={'POSE'}
    )
    
    print("✅ Baking complete")

def setup_render_settings(fps, output_path, frame_end):
    """Configure render settings"""
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.fps = fps
    scene.frame_start = START_FRAME
    scene.frame_end = START_FRAME + frame_end - 1
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    
    if RENDER_PREVIEW:
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.codec = 'H264'
        scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
        scene.render.filepath = output_path

def export_animated_fbx(armature, output_path):
    """Export character with animation"""
    print("📤 Exporting FBX...")
    
    # Select armature and all its children
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    
    # Select all meshes parented to armature
    for obj in bpy.data.objects:
        if obj.parent == armature and obj.type == 'MESH':
            obj.select_set(True)
    
    bpy.context.view_layer.objects.active = armature
    
    # Export settings
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=True,
        apply_scale_options='FBX_SCALE_ALL',
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        add_leaf_bones=False,
        mesh_smooth_type='FACE'
    )
    
    print(f"✅ FBX exported: {output_path}")

def check_input_files():
    """Verify all required files exist"""
    json_path = bpy.path.abspath(JSON_PATH)
    fbx_path = bpy.path.abspath(CHAR_FBX)
    
    print("\n🔍 Checking input files...")
    print(f"   JSON: {json_path} - {'✅ EXISTS' if os.path.exists(json_path) else '❌ MISSING'}")
    print(f"   FBX:  {fbx_path} - {'✅ EXISTS' if os.path.exists(fbx_path) else '❌ MISSING'}")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Pose JSON not found: {json_path}")
    if not os.path.exists(fbx_path):
        raise FileNotFoundError(f"Character FBX not found: {fbx_path}")

# ------------------- MAIN PIPELINE -------------------
def main():
    try:
        print("🚀 Starting Mixamo Auto-Rigger...")
        
        # 1. Check inputs
        check_input_files()
        
        # 2. Clear scene
        safe_clear_scene()
        
        # 3. Setup scene
        ensure_camera_light()
        
        # 4. Load and process pose data
        pose_data = load_pose_json(JSON_PATH)
        total_frames = len(pose_data)
        landmarks_per_frame = len(pose_data[0]["landmarks"])
        
        print(f"📊 Processing {total_frames} frames with {landmarks_per_frame} landmarks each")
        
        # Convert to numpy array
        raw_landmarks = np.array([
            [[lm["x"], lm["y"], lm["z"]] for lm in frame["landmarks"]] 
            for frame in pose_data
        ], dtype=np.float32)
        
        # Smooth data
        smoothed_landmarks = moving_average(raw_landmarks, SMOOTH_WINDOW)
        
        # Convert to world coordinates
        landmark_data_world = []
        for frame_idx in range(total_frames):
            frame_world = []
            for lm_idx in range(landmarks_per_frame):
                world_pos = norm_to_world(smoothed_landmarks[frame_idx, lm_idx])
                frame_world.append(world_pos)
            landmark_data_world.append(frame_world)
        
        print("✅ Pose data processed")
        
        # 5. Import character
        armature = import_mixamo_character(CHAR_FBX)
        if not armature:
            print("❌ Failed to import character")
            return
        
        # 6. Position character
        armature.location = (0, 0, 0)
        armature.rotation_euler = (0, 0, 0)
        
        # 7. Apply animation
        apply_simple_animation(armature, landmark_data_world, total_frames)
        
        # 8. Bake animation
        bake_animation(armature, START_FRAME, START_FRAME + total_frames - 1)
        
        # 9. Export FBX
        export_animated_fbx(armature, OUT_FBX)
        
        # 10. Setup and render MP4
        if RENDER_PREVIEW:
            setup_render_settings(FPS, OUT_MP4, total_frames)
            print("🎥 Rendering animation preview...")
            bpy.ops.render.render(animation=True)
            print(f"✅ MP4 rendered: {OUT_MP4}")
        
        print("\n🎉 PIPELINE COMPLETE! 🎉")
        print(f"   FBX: {OUT_FBX}")
        if RENDER_PREVIEW:
            print(f"   MP4: {OUT_MP4}")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

# Run the pipeline
if __name__ == "__main__":
    main()