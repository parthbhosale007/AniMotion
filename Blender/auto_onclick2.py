# Blender One-Click Remy Animation Pipeline
# Converts pose_data.json to full Mixamo character animation
# Run this script in Blender with your character.fbx and pose_data.json ready

import bpy
import json
import os
import math
import mathutils
from mathutils import Vector, Quaternion, Euler
import bmesh

print("🎬 REMY ANIMATION PIPELINE STARTING...")

# =================== CONFIGURATION ===================
JSON_PATH = bpy.path.abspath("//output/pose_data.json")
CHAR_FBX = bpy.path.abspath("//assets/character.fbx") 
OUT_FBX = bpy.path.abspath("//output/remy_animated.fbx")
OUT_MP4 = bpy.path.abspath("//output/remy_animation.mp4")

# Animation settings
FPS = 30
SCALE_FACTOR = 8.0  # Scale up movements for visibility
SMOOTH_FACTOR = 0.3  # Smoothing between frames
START_FRAME = 1

print(f"📁 JSON: {JSON_PATH}")
print(f"📁 FBX: {CHAR_FBX}")
print(f"📁 Output FBX: {OUT_FBX}")
print(f"📁 Output MP4: {OUT_MP4}")

# =================== UTILITY FUNCTIONS ===================
def clear_scene():
    """Complete scene cleanup"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Clear all data
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for armature in bpy.data.armatures:
        bpy.data.armatures.remove(armature)
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)
    
    print("🧹 Scene cleared")

def setup_scene():
    """Setup camera, lighting, and render settings"""
    # Create camera
    bpy.ops.object.camera_add(location=(10, -10, 8))
    camera = bpy.context.active_object
    camera.rotation_euler = (1.1, 0, 0.785)
    bpy.context.scene.camera = camera
    
    # Create lighting
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    light = bpy.context.active_object
    light.data.energy = 3
    
    # Render settings
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.fps = FPS
    
    print("🎥 Scene setup complete")

def load_pose_data(json_path):
    """Load and validate pose data"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Pose data not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if not data:
        raise ValueError("Pose data is empty")
    
    print(f"📊 Loaded {len(data)} frames with {len(data[0]['landmarks'])} landmarks each")
    return data

def import_mixamo_character(fbx_path):
    """Import Mixamo character and return armature"""
    if not os.path.exists(fbx_path):
        raise FileNotFoundError(f"Character FBX not found: {fbx_path}")
    
    print("🎭 Importing Mixamo character...")
    
    # Import FBX
    bpy.ops.import_scene.fbx(
        filepath=fbx_path,
        use_manual_orientation=True,
        global_scale=1.0,
        bake_space_transform=False
    )
    
    # Find armature and meshes
    armature = None
    meshes = []
    
    for obj in bpy.context.selected_objects:
        if obj.type == 'ARMATURE':
            armature = obj
        elif obj.type == 'MESH':
            meshes.append(obj)
    
    if not armature:
        # Try to find armature in scene
        for obj in bpy.data.objects:
            if obj.type == 'ARMATURE':
                armature = obj
                break
    
    if not armature:
        raise RuntimeError("No armature found in character FBX")
    
    armature.name = "RemyRig"
    print(f"✅ Found armature: {armature.name} with {len(armature.pose.bones)} bones")
    print(f"✅ Found {len(meshes)} mesh objects")
    
    return armature, meshes

def get_bone_safe(armature, bone_name):
    """Safely get bone with fallback names"""
    # Direct lookup
    if bone_name in armature.pose.bones:
        return armature.pose.bones[bone_name]
    
    # Try without prefix
    simple_name = bone_name.replace("mixamorig:", "")
    if simple_name in armature.pose.bones:
        return armature.pose.bones[simple_name]
    
    # Try common variations
    variations = [
        bone_name.replace("_", ""),
        bone_name.lower(),
        bone_name.upper(),
        f"mixamorig:{simple_name}",
        f"{simple_name}",
    ]
    
    for var in variations:
        if var in armature.pose.bones:
            return armature.pose.bones[var]
    
    return None

def convert_pose_to_world(landmarks, scale=SCALE_FACTOR):
    """Convert MediaPipe normalized coordinates to Blender world space"""
    world_landmarks = {}
    
    # MediaPipe landmark indices
    indices = {
        'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12, 
        'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28, 'left_foot_index': 31, 'right_foot_index': 32
    }
    
    for name, idx in indices.items():
        if idx < len(landmarks):
            lm = landmarks[idx]
            # Convert MediaPipe [0,1] normalized coords to Blender world
            x = (lm['x'] - 0.5) * scale
            y = -lm['z'] * scale  # MediaPipe Z becomes Blender Y (depth)
            z = (0.5 - lm['y']) * scale  # MediaPipe Y becomes Blender Z (flip up/down)
            world_landmarks[name] = Vector((x, y, z))
    
    return world_landmarks

def calculate_direction_rotation(from_pos, to_pos, bone_rest_dir=(0, 1, 0)):
    """Calculate rotation to point bone from one position to another"""
    direction = (to_pos - from_pos).normalized()
    rest_vector = Vector(bone_rest_dir).normalized()
    
    # Calculate rotation quaternion
    rotation = rest_vector.rotation_difference(direction)
    return rotation

def apply_remy_animation(armature, pose_frames):
    """Apply full body animation to Remy character"""
    print(f"🎬 Applying animation to {len(pose_frames)} frames...")
    
    # Set armature as active and enter pose mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # Clear existing animation
    if armature.animation_data:
        armature.animation_data_clear()
    
    # Reset all bones
    bpy.ops.pose.select_all(action='SELECT')
    bpy.ops.pose.rot_clear()
    bpy.ops.pose.loc_clear()
    
    # Define bone mapping with rest directions
    bone_configs = [
        # Core body
        ('mixamorig:Hips', None, None, True),  # Root bone - gets location
        ('mixamorig:Spine', 'left_hip|right_hip', 'left_shoulder|right_shoulder', False),
        ('mixamorig:Spine1', 'left_hip|right_hip', 'left_shoulder|right_shoulder', False),
        ('mixamorig:Spine2', 'left_hip|right_hip', 'left_shoulder|right_shoulder', False),
        ('mixamorig:Neck', 'left_shoulder|right_shoulder', 'nose', False),
        ('mixamorig:Head', 'left_shoulder|right_shoulder', 'nose', False),
        
        # Left arm
        ('mixamorig:LeftShoulder', 'left_shoulder', 'left_elbow', False),
        ('mixamorig:LeftArm', 'left_shoulder', 'left_elbow', False),
        ('mixamorig:LeftForeArm', 'left_elbow', 'left_wrist', False),
        
        # Right arm  
        ('mixamorig:RightShoulder', 'right_shoulder', 'right_elbow', False),
        ('mixamorig:RightArm', 'right_shoulder', 'right_elbow', False),
        ('mixamorig:RightForeArm', 'right_elbow', 'right_wrist', False),
        
        # Left leg
        ('mixamorig:LeftUpLeg', 'left_hip', 'left_knee', False),
        ('mixamorig:LeftLeg', 'left_knee', 'left_ankle', False),
        ('mixamorig:LeftFoot', 'left_ankle', 'left_foot_index', False),
        
        # Right leg
        ('mixamorig:RightUpLeg', 'right_hip', 'right_knee', False),
        ('mixamorig:RightLeg', 'right_knee', 'right_ankle', False),
        ('mixamorig:RightFoot', 'right_ankle', 'right_foot_index', False),
    ]
    
    # Find available bones
    active_bones = []
    for bone_name, from_landmark, to_landmark, uses_location in bone_configs:
        bone = get_bone_safe(armature, bone_name)
        if bone:
            active_bones.append((bone, from_landmark, to_landmark, uses_location))
            print(f"✅ Will animate: {bone.name}")
        else:
            print(f"⚠️  Missing: {bone_name}")
    
    if not active_bones:
        raise RuntimeError("No compatible bones found!")
    
    print(f"🦴 Animating {len(active_bones)} bones")
    
    # Process each frame
    frame_count = len(pose_frames)
    for frame_idx, frame_data in enumerate(pose_frames):
        current_frame = START_FRAME + frame_idx
        bpy.context.scene.frame_set(current_frame)
        
        if frame_idx % 30 == 0:
            print(f"⏱️  Processing frame {frame_idx + 1}/{frame_count}")
        
        # Convert landmarks to world positions
        landmarks = frame_data['landmarks']
        world_positions = convert_pose_to_world(landmarks)
        
        # Calculate derived positions
        if 'left_hip' in world_positions and 'right_hip' in world_positions:
            world_positions['hip_center'] = (world_positions['left_hip'] + world_positions['right_hip']) * 0.5
        
        if 'left_shoulder' in world_positions and 'right_shoulder' in world_positions:
            world_positions['shoulder_center'] = (world_positions['left_shoulder'] + world_positions['right_shoulder']) * 0.5
        
        # Apply transformations to each bone
        for bone, from_key, to_key, uses_location in active_bones:
            
            if uses_location and 'hip_center' in world_positions:
                # Root bone gets location animation
                world_pos = world_positions['hip_center']
                local_pos = armature.matrix_world.inverted() @ world_pos
                bone.location = local_pos
                bone.keyframe_insert(data_path="location", frame=current_frame)
            
            if from_key and to_key:
                # Get positions for rotation calculation
                from_pos = None
                to_pos = None
                
                # Handle combined positions (like left_hip|right_hip)
                if '|' in from_key:
                    keys = from_key.split('|')
                    positions = [world_positions.get(k) for k in keys if k in world_positions]
                    if positions:
                        from_pos = sum(positions, Vector()) / len(positions)
                else:
                    from_pos = world_positions.get(from_key)
                
                if '|' in to_key:
                    keys = to_key.split('|')
                    positions = [world_positions.get(k) for k in keys if k in world_positions]
                    if positions:
                        to_pos = sum(positions, Vector()) / len(positions)
                else:
                    to_pos = world_positions.get(to_key)
                
                if from_pos and to_pos:
                    # Calculate rotation
                    rotation = calculate_direction_rotation(from_pos, to_pos)
                    
                    # Apply with smoothing
                    if frame_idx > 0:
                        # Blend with previous rotation for smoothness
                        bone.rotation_mode = 'QUATERNION'
                        prev_rotation = bone.rotation_quaternion.copy()
                        new_rotation = prev_rotation.slerp(rotation, SMOOTH_FACTOR)
                        bone.rotation_quaternion = new_rotation
                    else:
                        bone.rotation_mode = 'QUATERNION'
                        bone.rotation_quaternion = rotation
                    
                    bone.keyframe_insert(data_path="rotation_quaternion", frame=current_frame)
    
    # Set frame range
    bpy.context.scene.frame_start = START_FRAME
    bpy.context.scene.frame_end = START_FRAME + frame_count - 1
    
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"✅ Animation applied! {frame_count} frames from {START_FRAME} to {START_FRAME + frame_count - 1}")

def export_results(armature, meshes):
    """Export animated FBX and render MP4 preview"""
    print("💾 Exporting results...")
    
    # Select all relevant objects
    bpy.ops.object.select_all(action='DESELECT')
    armature.select_set(True)
    for mesh in meshes:
        mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature
    
    # Export FBX with animation
    bpy.ops.export_scene.fbx(
        filepath=OUT_FBX,
        use_selection=True,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        add_leaf_bones=False,
        primary_bone_axis='Y',
        secondary_bone_axis='X'
    )
    print(f"✅ FBX exported: {OUT_FBX}")
    
    # Setup and render MP4
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.filepath = OUT_MP4
    
    print(f"🎥 Rendering MP4 preview...")
    bpy.ops.render.render(animation=True)
    print(f"✅ MP4 rendered: {OUT_MP4}")

# =================== MAIN EXECUTION ===================
def run_remy_pipeline():
    """Execute the complete Remy animation pipeline"""
    try:
        print("🚀 Starting Remy Animation Pipeline...")
        
        # Step 1: Clean and setup scene
        clear_scene()
        setup_scene()
        
        # Step 2: Load pose data
        pose_data = load_pose_data(JSON_PATH)
        
        # Step 3: Import character
        armature, meshes = import_mixamo_character(CHAR_FBX)
        
        # Step 4: Apply animation
        apply_remy_animation(armature, pose_data)
        
        # Step 5: Export results
        export_results(armature, meshes)
        
        print("🎉 REMY ANIMATION PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📁 Animated FBX: {OUT_FBX}")
        print(f"🎬 Preview Video: {OUT_MP4}")
        
    except Exception as e:
        print(f"💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

# Execute the pipeline
if __name__ == "__main__":
    run_remy_pipeline()