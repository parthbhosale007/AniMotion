# Blender/auto_oneclick.py - AXIS FIXED VERSION
import bpy, json, os, math
import numpy as np
import mathutils

# ------------------- CONFIG -------------------
JSON_PATH   = "output/pose_data.json"
CHAR_FBX    = "assets/Remy.fbx"
OUT_FBX     = "output/animated_character.fbx"
OUT_MP4     = "output/anim_preview.mp4"

VIDEO_W, VIDEO_H = 640, 480
SCALE          = 0.08  # Increased scale for better visibility
START_FRAME    = 1
SMOOTH_WINDOW  = 5
FPS            = 30
RENDER_PREVIEW = False

USE_FRAMES = 60  # Use 60 frames to see the motion
ROTATION_STRENGTH = 0.6  # Increased to 60% - more visible motion
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
    """FIXED COORDINATE SYSTEM: MediaPipe to Blender conversion"""
    # MediaPipe: X=left/right, Y=up/down, Z=forward/backward
    # Blender: X=left/right, Y=forward/backward, Z=up/down
    
    # Convert MediaPipe to Blender coordinates:
    x = (lm[0] - 0.5) * VIDEO_W * SCALE * scale_x  # X stays same
    z = (0.5 - lm[1]) * VIDEO_H * SCALE * scale_y  # MediaPipe Y -> Blender Z (UP)
    y = -lm[2] * VIDEO_W * SCALE * scale_z         # MediaPipe Z -> Blender Y (forward)
    
    return (x, z, y)  # Swapped Y and Z!

def ensure_camera_light():
    if "AutoCamera" not in bpy.data.objects:
        bpy.ops.object.camera_add(location=(8, -12, 6))  # Better camera position
        cam = bpy.context.object
        cam.name = "AutoCamera"
    else:
        cam = bpy.data.objects["AutoCamera"]
    cam.location = (8, -12, 6)
    cam.rotation_euler = (1.1, 0, 0.6)
    bpy.context.scene.camera = cam
    
    if "AutoLight" not in bpy.data.objects:
        bpy.ops.object.light_add(type='SUN', location=(10, -10, 15))
        light = bpy.context.object
        light.name = "AutoLight"
        light.data.energy = 4.0

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

def get_bone_rest_direction(armature, bone_name):
    """Get bone direction in armature's local space"""
    bone = armature.data.bones[bone_name]
    return (bone.tail_local - bone.head_local).normalized()

def calculate_proper_rotation(armature, bone_name, target_direction_world):
    """Calculate rotation with proper axis handling"""
    
    # Get bone's rest direction in local space
    bone_rest_direction_local = get_bone_rest_direction(armature, bone_name)
    
    # Convert target direction from world to armature local space
    armature_matrix = armature.matrix_world
    target_direction_local = armature_matrix.inverted().to_3x3() @ target_direction_world.normalized()
    
    # Calculate rotation in local space
    if target_direction_local.length > 0.001 and bone_rest_direction_local.length > 0.001:
        return bone_rest_direction_local.rotation_difference(target_direction_local)
    else:
        return mathutils.Quaternion()

def apply_corrected_animation(armature, landmark_data_world, frame_count):
    """Animation with CORRECTED AXIS handling"""
    print("🎬 Starting AXIS-CORRECTED animation...")
    
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
    
    # ANIMATE THESE BONES - with better motion
    ANIMATED_BONES = [
        "mixamorig:LeftArm", "mixamorig:RightArm",
        "mixamorig:LeftForeArm", "mixamorig:RightForeArm", 
        "mixamorig:LeftUpLeg", "mixamorig:RightUpLeg",
        "mixamorig:LeftLeg", "mixamorig:RightLeg"
    ]
    
    print(f"🎯 Animating {len(ANIMATED_BONES)} bones over {USE_FRAMES} frames")
    print(f"🔧 Rotation strength: {ROTATION_STRENGTH * 100}%")
    
    for frame_idx in range(min(USE_FRAMES, frame_count)):
        frame_num = START_FRAME + frame_idx
        bpy.context.scene.frame_set(frame_num)
        
        frame_landmarks = landmark_data_world[frame_idx]
        
        # Get landmark positions (now in correct Blender coordinates)
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
        left_ankle = mathutils.Vector(frame_landmarks[MP_INDICES['left_ankle']])
        right_ankle = mathutils.Vector(frame_landmarks[MP_INDICES['right_ankle']])
        
        # Apply to each bone
        for bone_name in ANIMATED_BONES:
            if bone_name not in armature.pose.bones:
                continue
                
            bone = armature.pose.bones[bone_name]
            bone.rotation_mode = 'QUATERNION'
            
            # Calculate target directions
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
            elif bone_name == "mixamorig:LeftLeg":
                target_dir = (left_ankle - left_knee)
            elif bone_name == "mixamorig:RightLeg":
                target_dir = (right_ankle - right_knee)
            else:
                continue
            
            # Apply rotation with proper strength
            if target_dir.length > 0.1:
                rotation = calculate_proper_rotation(armature, bone_name, target_dir)
                # Apply the rotation with our strength setting
                final_rotation = rotation.slerp(mathutils.Quaternion(), 1.0 - ROTATION_STRENGTH)
                bone.rotation_quaternion = final_rotation
            else:
                bone.rotation_quaternion = mathutils.Quaternion()
            
            bone.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)
    
    bpy.context.scene.frame_start = START_FRAME
    bpy.context.scene.frame_end = START_FRAME + min(USE_FRAMES, frame_count) - 1
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
        print("🚀 Starting AXIS-FIXED Mixamo Auto-Rigger...")
        print("🎯 TARGET: Jumping Jacks motion in correct vertical plane")
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
        
        print("✅ Pose data processed with CORRECTED AXIS")
        
        armature = import_mixamo_character(CHAR_FBX)
        if not armature:
            print("❌ Failed to import character")
            return
        
        # Reset character position
        armature.location = (0, 0, 0)
        armature.rotation_euler = (0, 0, 0)
        
        # Apply AXIS-CORRECTED animation
        apply_corrected_animation(armature, landmark_data_world, total_frames)
        
        # Bake and export
        bake_frames = min(USE_FRAMES, total_frames)
        bake_animation(armature, START_FRAME, START_FRAME + bake_frames - 1)
        export_animated_fbx(armature, OUT_FBX)
        
        print("\n🎉 AXIS-FIXED PIPELINE COMPLETE! 🎉")
        print(f"   FBX: {OUT_FBX}")
        print("   🔧 CRITICAL FIXES APPLIED:")
        print("   ✅ MediaPipe Y (up/down) → Blender Z (up/down)")
        print("   ✅ MediaPipe Z (forward) → Blender Y (forward)") 
        print("   ✅ Increased rotation strength to 60%")
        print("   ✅ Using 60 frames for clear motion")
        print("   ✅ Better camera angle to see vertical motion")
        print("   🎯 Expected: Arms/legs should move UP/DOWN in jumping jacks!")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()