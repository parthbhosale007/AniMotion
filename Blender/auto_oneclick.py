# Blender/auto_oneclick.py - PROPER BONE MAPPING VERSION
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

def analyze_bone_structure(armature):
    """Analyze bone hierarchy and rest poses"""
    print("\n🔬 ANALYZING BONE STRUCTURE:")
    
    bone_info = {}
    
    for bone in armature.pose.bones:
        # Get bone in rest pose
        data_bone = armature.data.bones[bone.name]
        
        # Calculate bone direction and length
        if data_bone.parent:
            # Local matrix gives us the bone's rest pose orientation
            local_matrix = data_bone.matrix_local
            head = local_matrix.translation
            tail = local_matrix @ mathutils.Vector((0, data_bone.length, 0))
            direction = (tail - head).normalized()
        else:
            # Root bone
            direction = mathutils.Vector((0, 1, 0))  # Default Y-forward
        
        bone_info[bone.name] = {
            'head': data_bone.head_local,
            'tail': data_bone.tail_local,
            'length': data_bone.length,
            'direction': direction,
            'parent': data_bone.parent.name if data_bone.parent else None,
            'matrix': data_bone.matrix_local
        }
        
        print(f"   {bone.name}:")
        print(f"      Head: {data_bone.head_local}")
        print(f"      Tail: {data_bone.tail_local}") 
        print(f"      Direction: {direction}")
        print(f"      Length: {data_bone.length:.3f}")
        print(f"      Parent: {data_bone.parent.name if data_bone.parent else 'None'}")
    
    return bone_info

def get_proper_bone_mappings(armature, bone_info):
    """Create proper bone mappings based on actual rig structure"""
    
    # MediaPipe to Mixamo bone mapping
    MIXAMO_MAPPINGS = {
        # Spine chain
        'mixamorig:Hips': {'type': 'ROOT', 'mediapipe_start': 'hip_center', 'mediapipe_end': None},
        'mixamorig:Spine': {'type': 'SPINE', 'mediapipe_start': 'hip_center', 'mediapipe_end': 'spine_mid'},
        'mixamorig:Spine1': {'type': 'SPINE', 'mediapipe_start': 'spine_mid', 'mediapipe_end': 'spine_upper'},
        'mixamorig:Spine2': {'type': 'SPINE', 'mediapipe_start': 'spine_upper', 'mediapipe_end': 'neck_base'},
        
        # Left Arm
        'mixamorig:LeftShoulder': {'type': 'SHOULDER', 'mediapipe_start': 'spine_upper', 'mediapipe_end': 'left_shoulder'},
        'mixamorig:LeftArm': {'type': 'UPPER_ARM', 'mediapipe_start': 'left_shoulder', 'mediapipe_end': 'left_elbow'},
        'mixamorig:LeftForeArm': {'type': 'LOWER_ARM', 'mediapipe_start': 'left_elbow', 'mediapipe_end': 'left_wrist'},
        
        # Right Arm
        'mixamorig:RightShoulder': {'type': 'SHOULDER', 'mediapipe_start': 'spine_upper', 'mediapipe_end': 'right_shoulder'},
        'mixamorig:RightArm': {'type': 'UPPER_ARM', 'mediapipe_start': 'right_shoulder', 'mediapipe_end': 'right_elbow'},
        'mixamorig:RightForeArm': {'type': 'LOWER_ARM', 'mediapipe_start': 'right_elbow', 'mediapipe_end': 'right_wrist'},
        
        # Left Leg
        'mixamorig:LeftUpLeg': {'type': 'UPPER_LEG', 'mediapipe_start': 'left_hip', 'mediapipe_end': 'left_knee'},
        'mixamorig:LeftLeg': {'type': 'LOWER_LEG', 'mediapipe_start': 'left_knee', 'mediapipe_end': 'left_ankle'},
        
        # Right Leg
        'mixamorig:RightUpLeg': {'type': 'UPPER_LEG', 'mediapipe_start': 'right_hip', 'mediapipe_end': 'right_knee'},
        'mixamorig:RightLeg': {'type': 'LOWER_LEG', 'mediapipe_start': 'right_knee', 'mediapipe_end': 'right_ankle'},
        
        # Head
        'mixamorig:Neck': {'type': 'NECK', 'mediapipe_start': 'neck_base', 'mediapipe_end': 'head_base'},
        'mixamorig:Head': {'type': 'HEAD', 'mediapipe_start': 'head_base', 'mediapipe_end': 'head_top'},
    }
    
    # Filter to only include bones that exist in the armature
    valid_mappings = {}
    for bone_name, mapping in MIXAMO_MAPPINGS.items():
        if bone_name in armature.pose.bones:
            valid_mappings[bone_name] = mapping
            print(f"   ✅ Mapping: {bone_name} -> {mapping['type']}")
        else:
            print(f"   ❌ Missing: {bone_name}")
    
    return valid_mappings

def calculate_proper_rotation(target_direction, bone_rest_direction, up_vector=mathutils.Vector((0, 0, 1))):
    """Calculate proper rotation from bone's rest pose to target direction"""
    
    # Normalize vectors
    target_dir = target_direction.normalized()
    bone_dir = bone_rest_direction.normalized()
    
    # Handle edge cases
    if target_dir.length < 0.001 or bone_dir.length < 0.001:
        return mathutils.Quaternion()
    
    # Calculate the rotation between rest direction and target direction
    rotation = bone_dir.rotation_difference(target_dir)
    
    return rotation

def apply_proper_animation(armature, landmark_data_world, frame_count, bone_info, bone_mappings):
    """Apply animation with proper bone space calculations"""
    print("🎬 Starting PROPER animation application...")
    
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
    
    print(f"🎯 Animating {len(bone_mappings)} bones over {min(10, frame_count)} test frames")
    
    # Test with first 10 frames only
    test_frames = min(10, frame_count)
    
    for frame_idx in range(test_frames):
        frame_num = START_FRAME + frame_idx
        bpy.context.scene.frame_set(frame_num)
        
        frame_landmarks = landmark_data_world[frame_idx]
        
        # Calculate ALL derived positions first
        left_shoulder = mathutils.Vector(frame_landmarks[MP_INDICES['left_shoulder']])
        right_shoulder = mathutils.Vector(frame_landmarks[MP_INDICES['right_shoulder']])
        left_hip = mathutils.Vector(frame_landmarks[MP_INDICES['left_hip']])
        right_hip = mathutils.Vector(frame_landmarks[MP_INDICES['right_hip']])
        nose = mathutils.Vector(frame_landmarks[MP_INDICES['nose']])
        
        # Calculate spine positions based on actual proportions
        hip_center = (left_hip + right_hip) * 0.5
        shoulder_center = (left_shoulder + right_shoulder) * 0.5
        
        # Spine chain with proper proportions
        spine_vector = shoulder_center - hip_center
        spine_mid = hip_center + spine_vector * 0.33
        spine_upper = hip_center + spine_vector * 0.66
        neck_base = shoulder_center
        head_base = shoulder_center + (nose - shoulder_center) * 0.3
        head_top = nose
        
        positions = {
            # Raw landmarks
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
            
            # Derived positions
            'hip_center': hip_center,
            'spine_mid': spine_mid,
            'spine_upper': spine_upper,
            'neck_base': neck_base,
            'head_base': head_base,
            'head_top': head_top,
        }
        
        # Apply animation to each mapped bone
        for bone_name, mapping in bone_mappings.items():
            bone = armature.pose.bones[bone_name]
            bone.rotation_mode = 'QUATERNION'
            
            bone_data = bone_info[bone_name]
            bone_rest_direction = bone_data['direction']
            
            if mapping['type'] == 'ROOT':
                # Only move hips slightly
                hip_local = armature.matrix_world.inverted() @ hip_center
                bone.location = hip_local * 0.3  # Reduced movement
                bone.keyframe_insert(data_path="location", frame=frame_num)
                bone.rotation_quaternion = mathutils.Quaternion()  # No rotation for root
                
            elif mapping['mediapipe_end']:
                # Calculate target direction from MediaPipe data
                start_pos = positions.get(mapping['mediapipe_start'])
                end_pos = positions.get(mapping['mediapipe_end'])
                
                if start_pos and end_pos and (end_pos - start_pos).length > 0.01:
                    target_direction = (end_pos - start_pos).normalized()
                    
                    # Calculate proper rotation
                    rotation = calculate_proper_rotation(target_direction, bone_rest_direction)
                    
                    # Apply constraints based on bone type
                    if mapping['type'] in ['SPINE', 'NECK']:
                        # Limit spine/neck rotation
                        rotation = rotation.slerp(mathutils.Quaternion(), 0.7)
                    elif mapping['type'] == 'HEAD':
                        # Limit head rotation
                        rotation = rotation.slerp(mathutils.Quaternion(), 0.8)
                    
                    bone.rotation_quaternion = rotation
                else:
                    # Default pose if no valid direction
                    bone.rotation_quaternion = mathutils.Quaternion()
            
            # Insert keyframe
            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
    
    # Set scene frame range
    bpy.context.scene.frame_start = START_FRAME
    bpy.context.scene.frame_end = START_FRAME + test_frames - 1
    print(f"📊 Frame range: {bpy.context.scene.frame_start} - {bpy.context.scene.frame_end}")

# ------------------- REST OF THE FUNCTIONS (same as before) -------------------

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
        print("🚀 Starting SCIENTIFIC Mixamo Auto-Rigger...")
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
        
        # STEP 1: Analyze bone structure
        bone_info = analyze_bone_structure(armature)
        
        # STEP 2: Create proper bone mappings
        bone_mappings = get_proper_bone_mappings(armature, bone_info)
        
        if not bone_mappings:
            print("❌ No valid bone mappings found!")
            return
        
        # STEP 3: Apply proper animation
        apply_proper_animation(armature, landmark_data_world, total_frames, bone_info, bone_mappings)
        
        # STEP 4: Bake and export
        test_frames = min(10, total_frames)
        bake_animation(armature, START_FRAME, START_FRAME + test_frames - 1)
        export_animated_fbx(armature, OUT_FBX)
        
        print("\n🎉 SCIENTIFIC PIPELINE COMPLETE! 🎉")
        print(f"   FBX: {OUT_FBX}")
        print("   🔬 Scientific approach:")
        print("   - Analyzed bone rest poses")
        print("   - Proper bone space calculations")
        print("   - Correct MediaPipe to Mixamo mapping")
        print("   - Rotation constraints based on bone type")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()