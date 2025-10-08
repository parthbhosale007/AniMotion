# Blender/auto_oneclick.py - SYSTEMATIC AXIS TESTING
import bpy, json, os, math
import numpy as np
import mathutils

# ------------------- CONFIG -------------------
JSON_PATH   = "output/pose_data.json"
CHAR_FBX    = "assets/Remy.fbx"
OUT_FBX     = "output/animated_character.fbx"

VIDEO_W, VIDEO_H = 640, 480
SCALE          = 0.08
START_FRAME    = 1
SMOOTH_WINDOW  = 5
USE_FRAMES     = 100  # Reduced for faster testing
ROTATION_STRENGTH = 0.8  # 80% strength for clear motion

# AXIS MAPPING OPTIONS - WE'LL TRY THEM ALL
AXIS_OPTIONS = {
    "OPTION_1": {"desc": "MP X→X, MP Y→Z, MP Z→-Y", "func": lambda x, y, z: (x, -z, y)},
    "OPTION_2": {"desc": "MP X→X, MP Y→Z, MP Z→Y", "func": lambda x, y, z: (x, z, y)},
    "OPTION_3": {"desc": "MP X→X, MP Y→-Z, MP Z→Y", "func": lambda x, y, z: (x, z, -y)},
    "OPTION_4": {"desc": "MP X→-X, MP Y→Z, MP Z→Y", "func": lambda x, y, z: (-x, z, y)},
    "OPTION_5": {"desc": "MP X→Y, MP Y→Z, MP Z→X", "func": lambda x, y, z: (y, x, z)},
    "OPTION_6": {"desc": "MP X→-Y, MP Y→Z, MP Z→X", "func": lambda x, y, z: (-y, x, z)},
}

# SELECT WHICH OPTION TO TEST (change this number to test different options)
SELECTED_OPTION = 1  # Change this from 1 to 6 to test different mappings
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

def norm_to_world(lm, axis_option):
    """Convert using selected axis mapping"""
    # Normalized MediaPipe coordinates
    mp_x = (lm[0] - 0.5) * VIDEO_W * SCALE
    mp_y = (lm[1] - 0.5) * VIDEO_H * SCALE  
    mp_z = lm[2] * VIDEO_W * SCALE
    
    # Apply selected axis mapping
    option_key = f"OPTION_{SELECTED_OPTION}"
    if option_key in AXIS_OPTIONS:
        mapping_func = AXIS_OPTIONS[option_key]["func"]
        blender_x, blender_y, blender_z = mapping_func(mp_x, mp_y, mp_z)
        return (blender_x, blender_y, blender_z)
    else:
        # Default fallback
        return (mp_x, mp_z, mp_y)

def ensure_camera_light():
    if "AutoCamera" not in bpy.data.objects:
        bpy.ops.object.camera_add(location=(6, -8, 4))
        cam = bpy.context.object
        cam.name = "AutoCamera"
    else:
        cam = bpy.data.objects["AutoCamera"]
    cam.location = (6, -8, 4)
    cam.rotation_euler = (1.2, 0, 0.7)
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
        
        return armature
    else:
        print("❌ No armature found in FBX file")
        return None

def get_bone_rest_direction(armature, bone_name):
    bone = armature.data.bones[bone_name]
    return (bone.tail_local - bone.head_local).normalized()

def calculate_proper_rotation(armature, bone_name, target_direction_world):
    bone_rest_direction_local = get_bone_rest_direction(armature, bone_name)
    armature_matrix = armature.matrix_world
    target_direction_local = armature_matrix.inverted().to_3x3() @ target_direction_world.normalized()
    
    if target_direction_local.length > 0.001 and bone_rest_direction_local.length > 0.001:
        return bone_rest_direction_local.rotation_difference(target_direction_local)
    else:
        return mathutils.Quaternion()

def apply_animation(armature, landmark_data_world, frame_count):
    print(f"🎬 Testing AXIS OPTION {SELECTED_OPTION}")
    
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    armature.animation_data_clear()
    
    MP_INDICES = {
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28
    }
    
    ANIMATED_BONES = [
        "mixamorig:LeftArm", "mixamorig:RightArm",
        "mixamorig:LeftForeArm", "mixamorig:RightForeArm", 
        "mixamorig:LeftUpLeg", "mixamorig:RightUpLeg",
    ]
    
    for frame_idx in range(min(USE_FRAMES, frame_count)):
        frame_num = START_FRAME + frame_idx
        bpy.context.scene.frame_set(frame_num)
        
        frame_landmarks = landmark_data_world[frame_idx]
        
        # Get positions with current axis mapping
        left_shoulder = mathutils.Vector(frame_landmarks[MP_INDICES['left_shoulder']])
        right_shoulder = mathutils.Vector(frame_landmarks[MP_INDICES['right_shoulder']])
        left_elbow = mathutils.Vector(frame_landmarks[MP_INDICES['left_elbow']])
        right_elbow = mathutils.Vector(frame_landmarks[MP_INDICES['right_elbow']])
        left_wrist = mathutils.Vector(frame_landmarks[MP_INDICES['left_wrist']])
        right_wrist = mathutils.Vector(frame_landmarks[MP_INDICES['right_wrist']])
        left_hip = mathutils.Vector(frame_landmarks[MP_INDICES['left_hip']])
        right_hip = mathutils.Vector(frame_landmarks[MP_INDICES['right_hip']])
        left_knee = mathutils.Vector(frame_landmarks[MP_INDICES['left_knee']])
        right_knee = mathutils.Vector(frame_landmarks[MP_INDICES['right_knee']])
        
        for bone_name in ANIMATED_BONES:
            if bone_name not in armature.pose.bones:
                continue
                
            bone = armature.pose.bones[bone_name]
            bone.rotation_mode = 'QUATERNION'
            
            if bone_name == "mixamorig:LeftArm":
                target_dir = (left_elbow - left_shoulder)
            elif bone_name == "mixamorig:RightArm":
                target_dir = (right_elbow - right_shoulder)
            elif bone_name == "mixamorig:LeftForeArm":
                target_dir = (left_wrist - left_elbow)
            elif bone_name == "mixamorig:RightForeArm":
                target_dir = (right_wrist - right_elbow)
            elif bone_name == "mixamorig:LeftUpLeg":
                target_dir = (left_knee - left_hip)
            elif bone_name == "mixamorig:RightUpLeg":
                target_dir = (right_knee - right_hip)
            else:
                continue
            
            if target_dir.length > 0.1:
                rotation = calculate_proper_rotation(armature, bone_name, target_dir)
                final_rotation = rotation.slerp(mathutils.Quaternion(), 1.0 - ROTATION_STRENGTH)
                bone.rotation_quaternion = final_rotation
            else:
                bone.rotation_quaternion = mathutils.Quaternion()
            
            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
    
    bpy.context.scene.frame_start = START_FRAME
    bpy.context.scene.frame_end = START_FRAME + min(USE_FRAMES, frame_count) - 1

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

def main():
    try:
        option_key = f"OPTION_{SELECTED_OPTION}"
        option_desc = AXIS_OPTIONS[option_key]["desc"] if option_key in AXIS_OPTIONS else "UNKNOWN"
        
        print("🚀 SYSTEMATIC AXIS TESTING")
        print(f"🎯 TESTING OPTION {SELECTED_OPTION}: {option_desc}")
        print("📋 Available options:")
        for i in range(1, 7):
            key = f"OPTION_{i}"
            if key in AXIS_OPTIONS:
                print(f"   {i}. {AXIS_OPTIONS[key]['desc']}")
        
        # Check files
        if not os.path.exists(JSON_PATH):
            raise FileNotFoundError(f"JSON not found: {JSON_PATH}")
        if not os.path.exists(CHAR_FBX):
            raise FileNotFoundError(f"FBX not found: {CHAR_FBX}")
        
        safe_clear_scene()
        ensure_camera_light()
        
        pose_data = load_pose_json(JSON_PATH)
        total_frames = len(pose_data)
        
        raw_landmarks = np.array([
            [[lm["x"], lm["y"], lm["z"]] for lm in frame["landmarks"]] 
            for frame in pose_data
        ], dtype=np.float32)
        
        smoothed_landmarks = moving_average(raw_landmarks, SMOOTH_WINDOW)
        
        landmark_data_world = []
        for frame_idx in range(total_frames):
            frame_world = []
            for lm_idx in range(len(pose_data[0]["landmarks"])):
                world_pos = norm_to_world(smoothed_landmarks[frame_idx, lm_idx], option_key)
                frame_world.append(world_pos)
            landmark_data_world.append(frame_world)
        
        print("✅ Pose data processed")
        
        armature = import_mixamo_character(CHAR_FBX)
        if not armature:
            return
        
        armature.location = (0, 0, 0)
        armature.rotation_euler = (0, 0, 0)
        
        apply_animation(armature, landmark_data_world, total_frames)
        bake_frames = min(USE_FRAMES, total_frames)
        bake_animation(armature, START_FRAME, START_FRAME + bake_frames - 1)
        export_animated_fbx(armature, OUT_FBX)
        
        print(f"\n🎉 OPTION {SELECTED_OPTION} COMPLETE!")
        print(f"   FBX: {OUT_FBX}")
        print(f"   Mapping: {option_desc}")
        print("\n🔍 CHECK THE RESULT:")
        print("   - Do arms move UP/DOWN for jumping jacks?")
        print("   - Is motion in the correct vertical plane?")
        print("   - If not, change SELECTED_OPTION and run again!")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()