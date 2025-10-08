# Blender/auto_oneclick.py - FIXED ROTATION VERSION
import bpy, json, os, math
import numpy as np
import mathutils

# ------------------- CONFIG -------------------
JSON_PATH   = "output/pose_data.json"
CHAR_FBX    = "assets/Remy.fbx"
OUT_FBX     = "output/animated_character.fbx"
OUT_MP4     = "output/anim_preview.mp4"

VIDEO_W, VIDEO_H = 640, 480
SCALE          = 0.05
START_FRAME    = 1
SMOOTH_WINDOW  = 5
FPS            = 30
RENDER_PREVIEW = False
# ------------------------------------------------

def safe_clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def load_pose_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    print(f"📁 Loaded {len(data)} frames, {len(data[0]['landmarks'])} landmarks each")
    return data

def moving_average(arr, win):
    if win < 3:
        return arr
    padded = np.pad(arr, ((win//2, win//2), (0,0), (0,0)), mode='edge')
    smoothed = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        smoothed[i] = np.mean(padded[i:i+win], axis=0)
    return smoothed

def norm_to_world(lm, scale_x=1, scale_y=1, scale_z=1):
    x = (lm[0] - 0.5) * VIDEO_W * SCALE * scale_x
    y = (lm[1] - 0.5) * VIDEO_H * SCALE * scale_y  
    z = lm[2] * VIDEO_W * SCALE * scale_z
    return (x, y, z)

def ensure_camera_light():
    if "AutoCamera" not in bpy.data.objects:
        bpy.ops.object.camera_add(location=(8, -8, 6))
        cam = bpy.context.object
        cam.name = "AutoCamera"
    else:
        cam = bpy.data.objects["AutoCamera"]
    cam.location = (8, -8, 6)
    cam.rotation_euler = (1.0, 0, 0.8)
    bpy.context.scene.camera = cam
    
    if "AutoLight" not in bpy.data.objects:
        bpy.ops.object.light_add(type='SUN', location=(5, -5, 8))
        light = bpy.context.object
        light.name = "AutoLight"
        light.data.energy = 3.0

def import_mixamo_character(fbx_path):
    existing_objects = set(bpy.data.objects)
    
    try:
        bpy.ops.import_scene.fbx(filepath=fbx_path)
    except Exception as e:
        print(f"❌ FBX import failed: {e}")
        return None
    
    new_objects = set(bpy.data.objects) - existing_objects
    armature = None
    
    for obj in new_objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break
    
    if not armature:
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE':
                armature = obj
                break
    
    if armature:
        armature.name = "Mixamo_Rig"
        print(f"✅ Imported armature: {armature.name}")
        
        meshes = []
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj != armature:
                has_armature_mod = False
                for mod in obj.modifiers:
                    if mod.type == 'ARMATURE':
                        mod.object = armature
                        has_armature_mod = True
                        break
                if not has_armature_mod:
                    mod = obj.modifiers.new("Armature", 'ARMATURE')
                    mod.object = armature
                obj.parent = armature
                meshes.append(obj)
                print(f"   └─ Linked mesh: {obj.name}")
        
        return armature
    else:
        print("❌ No armature found in FBX file")
        return None

def get_bone(armature, bone_name):
    if bone_name in armature.pose.bones:
        return armature.pose.bones[bone_name]
    return None

def calculate_limb_rotation(start_pos, end_pos, up_vector=mathutils.Vector((0, 0, 1))):
    """Better rotation calculation for limbs"""
    if (end_pos - start_pos).length < 0.001:
        return mathutils.Quaternion()
    
    # Calculate direction
    direction = (end_pos - start_pos).normalized()
    
    # Calculate right vector (cross product of up and direction)
    right = up_vector.cross(direction).normalized()
    
    # Recalculate proper up vector
    up = direction.cross(right).normalized()
    
    # Create rotation matrix
    rot_matrix = mathutils.Matrix((right, direction, up)).transposed()
    
    return rot_matrix.to_quaternion()

def apply_corrected_animation(armature, landmark_data_world, frame_count):
    """Fixed animation with proper rotations"""
    print("🎬 Starting CORRECTED animation application...")
    
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    armature.animation_data_clear()
    
    MP_INDICES = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28
    }
    
    # SIMPLIFIED BONE MAPPINGS - Focus on major bones first
    BONE_MAPPINGS = [
        # Hips - location only
        ("mixamorig:Hips", 'hip_center', None, 'LOCATION'),
        
        # Spine chain
        ("mixamorig:Spine", 'hip_center', 'spine_mid', 'ROTATION'),
        ("mixamorig:Spine1", 'spine_mid', 'spine_upper', 'ROTATION'),
        ("mixamorig:Spine2", 'spine_upper', 'neck_base', 'ROTATION'),
        
        # Arms - Left
        ("mixamorig:LeftArm", 'left_shoulder', 'left_elbow', 'ROTATION'),
        ("mixamorig:LeftForeArm", 'left_elbow', 'left_wrist', 'ROTATION'),
        
        # Arms - Right
        ("mixamorig:RightArm", 'right_shoulder', 'right_elbow', 'ROTATION'),
        ("mixamorig:RightForeArm", 'right_elbow', 'right_wrist', 'ROTATION'),
        
        # Legs - Left
        ("mixamorig:LeftUpLeg", 'left_hip', 'left_knee', 'ROTATION'),
        ("mixamorig:LeftLeg", 'left_knee', 'left_ankle', 'ROTATION'),
        
        # Legs - Right
        ("mixamorig:RightUpLeg", 'right_hip', 'right_knee', 'ROTATION'),
        ("mixamorig:RightLeg", 'right_knee', 'right_ankle', 'ROTATION'),
        
        # Head
        ("mixamorig:Neck", 'neck_base', 'head_base', 'ROTATION'),
        ("mixamorig:Head", 'head_base', 'head_top', 'ROTATION'),
    ]
    
    # Find valid bones
    valid_bones = []
    for bone_name, start_point, end_point, anim_type in BONE_MAPPINGS:
        bone = get_bone(armature, bone_name)
        if bone:
            valid_bones.append((bone, bone_name, start_point, end_point, anim_type))
            print(f"   ✅ Bone found: {bone_name}")
    
    print(f"🎯 Animating {len(valid_bones)} bones over {frame_count} frames")
    
    # Store initial rotations for reference
    initial_rotations = {}
    for bone, bone_name, start_point, end_point, anim_type in valid_bones:
        initial_rotations[bone_name] = bone.rotation_quaternion.copy()
    
    # Animation loop - TEST WITH FIRST 20 FRAMES
    test_frames = min(20, frame_count)
    for frame_idx in range(test_frames):
        frame_num = START_FRAME + frame_idx
        bpy.context.scene.frame_set(frame_num)
        
        frame_landmarks = landmark_data_world[frame_idx]
        
        # Calculate key positions
        left_shoulder = mathutils.Vector(frame_landmarks[MP_INDICES['left_shoulder']])
        right_shoulder = mathutils.Vector(frame_landmarks[MP_INDICES['right_shoulder']])
        left_hip = mathutils.Vector(frame_landmarks[MP_INDICES['left_hip']])
        right_hip = mathutils.Vector(frame_landmarks[MP_INDICES['right_hip']])
        nose = mathutils.Vector(frame_landmarks[MP_INDICES['nose']])
        
        # Derived positions with better calculations
        hip_center = (left_hip + right_hip) * 0.5
        shoulder_center = (left_shoulder + right_shoulder) * 0.5
        
        # Spine positions with proper proportions
        spine_height = (shoulder_center - hip_center).length
        spine_mid = hip_center + (shoulder_center - hip_center) * 0.33
        spine_upper = hip_center + (shoulder_center - hip_center) * 0.66
        neck_base = shoulder_center
        head_base = shoulder_center + (nose - shoulder_center) * 0.5
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
        
        for bone, bone_name, start_point, end_point, anim_type in valid_bones:
            bone.rotation_mode = 'QUATERNION'
            
            if anim_type == 'LOCATION' and bone_name == "mixamorig:Hips":
                # Only move hips slightly to avoid extreme movements
                hip_local = armature.matrix_world.inverted() @ hip_center
                # Reduce hip movement to 50% to prevent sliding
                reduced_location = hip_local * 0.5
                bone.location = reduced_location
                bone.keyframe_insert(data_path="location", frame=frame_num)
                # Keep hips rotation neutral
                bone.rotation_quaternion = mathutils.Quaternion()
                
            elif anim_type == 'ROTATION' and end_point:
                start_pos = positions.get(start_point)
                end_pos = positions.get(end_point)
                
                if start_pos and end_pos and (end_pos - start_pos).length > 0.01:
                    # Use improved rotation calculation
                    rotation = calculate_limb_rotation(start_pos, end_pos)
                    
                    # Apply rotation with constraints
                    if "Spine" in bone_name:
                        # Limit spine rotation
                        rotation = rotation.slerp(mathutils.Quaternion(), 0.3)
                    elif "Neck" in bone_name or "Head" in bone_name:
                        # Limit head rotation
                        rotation = rotation.slerp(mathutils.Quaternion(), 0.5)
                    
                    bone.rotation_quaternion = rotation
                else:
                    # Default pose
                    bone.rotation_quaternion = mathutils.Quaternion()
            
            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
    
    bpy.context.scene.frame_start = START_FRAME
    bpy.context.scene.frame_end = START_FRAME + test_frames - 1
    print(f"📊 Frame range: {bpy.context.scene.frame_start} - {bpy.context.scene.frame_end}")

def bake_animation(armature, start_frame, end_frame):
    print("🍳 Baking animation...")
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    
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

def export_animated_fbx(armature, output_path):
    print("📤 Exporting FBX...")
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    
    for obj in bpy.context.scene.objects:
        if obj.parent == armature and obj.type == 'MESH':
            obj.select_set(True)
    
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
    print("\n🔍 Checking input files...")
    print(f"   JSON: {JSON_PATH} - {'✅ EXISTS' if os.path.exists(JSON_PATH) else '❌ MISSING'}")
    print(f"   FBX:  {CHAR_FBX} - {'✅ EXISTS' if os.path.exists(CHAR_FBX) else '❌ MISSING'}")
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"Pose JSON not found: {JSON_PATH}")
    if not os.path.exists(CHAR_FBX):
        raise FileNotFoundError(f"Character FBX not found: {CHAR_FBX}")

def main():
    try:
        print("🚀 Starting IMPROVED Mixamo Auto-Rigger...")
        check_input_files()
        safe_clear_scene()
        ensure_camera_light()
        
        pose_data = load_pose_json(JSON_PATH)
        total_frames = len(pose_data)
        landmarks_per_frame = len(pose_data[0]["landmarks"])
        
        print(f"📊 Processing {total_frames} frames with {landmarks_per_frame} landmarks each")
        
        raw_landmarks = np.array([
            [[lm["x"], lm["y"], lm["z"]] for lm in frame["landmarks"]] 
            for frame in pose_data
        ], dtype=np.float32)
        
        smoothed_landmarks = moving_average(raw_landmarks, SMOOTH_WINDOW)
        
        landmark_data_world = []
        for frame_idx in range(total_frames):
            frame_world = []
            for lm_idx in range(landmarks_per_frame):
                world_pos = norm_to_world(smoothed_landmarks[frame_idx, lm_idx])
                frame_world.append(world_pos)
            landmark_data_world.append(frame_world)
        
        print("✅ Pose data processed")
        
        armature = import_mixamo_character(CHAR_FBX)
        if not armature:
            print("❌ Failed to import character")
            return
        
        # Reset character position
        armature.location = (0, 0, 0)
        armature.rotation_euler = (0, 0, 0)
        
        # Apply CORRECTED animation
        apply_corrected_animation(armature, landmark_data_world, total_frames)
        
        # Bake and export
        test_frames = min(20, total_frames)
        bake_animation(armature, START_FRAME, START_FRAME + test_frames - 1)
        export_animated_fbx(armature, OUT_FBX)
        
        print("\n🎉 IMPROVED PIPELINE COMPLETE! 🎉")
        print(f"   FBX: {OUT_FBX}")
        print("   🔧 Changes made:")
        print("   - Better rotation calculations")
        print("   - Limited hip movement")
        print("   - Constrained spine/head rotations")
        print("   - Testing first 20 frames only")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()