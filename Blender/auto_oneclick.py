# Blender/auto_oneclick.py
# One-click, headless: JSON -> Mixamo rig baked -> FBX + MP4
# Requires Blender 4.x (bundled NumPy OK). No SciPy needed.

import bpy, json, os, math
import numpy as np
import matutils

# ------------------- CONFIG -------------------
JSON_PATH   = bpy.path.abspath("//output/pose_data.json")
CHAR_FBX    = bpy.path.abspath("//assets/character.fbx")   # Mixamo character (T-pose)
OUT_FBX     = bpy.path.abspath("//output/skinned_animation.fbx")
OUT_MP4     = bpy.path.abspath("//output/anim.mp4")

VIDEO_W, VIDEO_H = 640, 480
SCALE          = 2.0
START_FRAME    = 1
SMOOTH_WINDOW  = 9   # odd, >=3
FPS            = 30
RENDER_PREVIEW = True   # set False to skip MP4
# ------------------------------------------------

def safe_clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for b in bpy.data.actions: bpy.data.actions.remove(b)

def load_pose_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    if not data:
        raise ValueError("Pose JSON empty.")
    return data

def moving_average(arr, win):
    if win < 3 or win % 2 == 0: return arr
    k = np.ones(win) / win
    # pad at ends to keep same length
    pad = win // 2
    arr_p = np.pad(arr, ((pad,pad),(0,0),(0,0)), mode='edge')
    out = np.empty_like(arr)
    for i in range(arr.shape[2]):  # x,y,z
        out[:,:,i] = np.apply_along_axis(lambda m: np.convolve(m, k, mode='valid'), 0, arr_p[:,:,i])
    return out

def norm_to_world(lm, sx, sy, sz):
    x = (lm[0] - 0.5) * VIDEO_W / 100.0 * SCALE * sx
    y = (lm[1] - 0.5) * VIDEO_H / 100.0 * SCALE * sy
    z = -lm[2] * SCALE * sz
    return (x, y, z)

def create_driver_empties(n_landmarks):
    empties = []
    for i in range(n_landmarks):
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0,0,0))
        e = bpy.context.object
        e.name = f"MP_{i}"
        empties.append(e)
    # make a parent to scale/move all drivers together
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0,0,0))
    root = bpy.context.object
    root.name = "MP_ROOT"
    for e in empties:
        e.parent = root
    return root, empties

def key_empties(empties, frames_xyz, sx=1, sy=1, sz=1):
    # frames_xyz shape: (T, 33, 3)
    for f_idx in range(frames_xyz.shape[0]):
        frame_num = START_FRAME + f_idx
        for i, e in enumerate(empties):
            e.location = norm_to_world(frames_xyz[f_idx, i], sx, sy, sz)
            e.keyframe_insert("location", frame=frame_num)

def mid(a, b):
    return ( (a[0]+b[0])/2.0, (a[1]+b[1])/2.0, (a[2]+b[2])/2.0 )

def ensure_camera_light():
    # Camera
    if not any(o for o in bpy.data.objects if o.type=="CAMERA"):
        bpy.ops.object.camera_add(location=(4, -6, 3), rotation=(math.radians(70), 0, math.radians(35)))
    # Light
    if not any(o for o in bpy.data.objects if o.type=="LIGHT"):
        bpy.ops.object.light_add(type='SUN', location=(0,0,10))

def import_mixamo(path):
    pre_objs = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=path)
    post_objs = set(bpy.data.objects)
    new = list(post_objs - pre_objs)
    arm = next((o for o in new if o.type=='ARMATURE'), None)
    if not arm:
        # try any armature in scene
        arm = next((o for o in bpy.data.objects if o.type=='ARMATURE'), None)
    if not arm:
        raise RuntimeError("No armature found after FBX import.")
    arm.name = "CharacterArmature"
    return arm

def get_bone(o, name):
    return o.pose.bones.get(name)

def add_clear_constraints(b):
    for c in list(b.constraints):
        b.constraints.remove(c)

def add_loc_constraint(b, tgt, sub):
    c = b.constraints.new('COPY_LOCATION')
    c.target = tgt
    c.subtarget = sub
    c.use_offset = True
    return c

def add_track_constraint(b, tgt, sub, axis='TRACK_Y'):
    c = b.constraints.new('DAMPED_TRACK')
    c.target = tgt
    c.subtarget = sub
    # Mixamo bones usually aim down local -Y, but Damped Track uses +Y by default.
    # We’ll stick with TRACK_Y; flip in-place if needed by setting use_invert_x/y/z.
    b.constraints.update()
    return c

def bake_pose(obj, f_start, f_end):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.nla.bake(
        frame_start=f_start,
        frame_end=f_end,
        only_selected=False,
        visual_keying=True,
        clear_constraints=True,
        use_current_action=True,
        bake_types={'POSE'}
    )

def setup_render(fps, out_mp4, frame_end):
    s = bpy.context.scene
    s.render.engine = 'BLENDER_EEVEE'
    s.render.fps = fps
    s.frame_start = START_FRAME
    s.frame_end = START_FRAME + frame_end - 1
    s.render.image_settings.file_format = 'FFMPEG'
    s.render.ffmpeg.format = 'MPEG4'
    s.render.ffmpeg.codec = 'H264'
    s.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    s.render.ffmpeg.ffmpeg_preset = 'GOOD'
    s.render.filepath = out_mp4

# ------------------- PIPELINE -------------------
safe_clear_scene()
ensure_camera_light()

# 1) Load and smooth
pose = load_pose_json(JSON_PATH)
T = len(pose)
L = len(pose[0]["landmarks"])
arr = np.array([[[lm["x"], lm["y"], lm["z"]] for lm in f["landmarks"]] for f in pose], dtype=np.float32)
arr_s = moving_average(arr, SMOOTH_WINDOW)

# 2) Create driver empties & animate
mp_root, empties = create_driver_empties(L)
key_empties(empties, arr_s, sx=1, sy=1, sz=1)

# 3) Import character
rig = import_mixamo(CHAR_FBX)
print(f"✅ Mixamo rig: {rig.name}")

# 4) Auto-scale/align drivers to character using hip/shoulder centers
# MediaPipe indices (Pose): LShoulder=11, RShoulder=12, LHip=23, RHip=24, Nose=0
mp_LS, mp_RS, mp_LH, mp_RH = 11, 12, 23, 24
# get first-frame positions (already keyed) to estimate scale
bpy.context.scene.frame_set(START_FRAME)
p_ls = empties[mp_LS].matrix_world.translation
p_rs = empties[mp_RS].matrix_world.translation
p_lh = empties[mp_LH].matrix_world.translation
p_rh = empties[mp_RH].matrix_world.translation
mp_shoulder_center = mid(p_ls, p_rs)
mp_hip_center      = mid(p_lh, p_rh)
mp_torso_len = (mp_shoulder_center - mathutils.Vector(mp_hip_center)).length

# Character rig distances
hips = get_bone(rig, "mixamorig:Hips") or get_bone(rig, "Hips")
neck = get_bone(rig, "mixamorig:Neck") or get_bone(rig, "Neck") \
       or get_bone(rig, "mixamorig:Spine2") or get_bone(rig, "Spine2")

if hips and neck:
    rig_hips_w = rig.matrix_world @ hips.head
    rig_neck_w = rig.matrix_world @ neck.head
    rig_torso_len = (rig_neck_w - rig_hips_w).length
    if mp_torso_len > 1e-6 and rig_torso_len > 1e-6:
        scale = rig_torso_len / mp_torso_len
        mp_root.scale = (scale, scale, scale)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # place MP_ROOT over Hips
    mp_root.location = rig_hips_w
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

# 5) Constraint scheme (lightweight, no SciPy):
# Drive hips by location; aim limbs with damped track to landmark empties
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode='POSE')

def b(name): return get_bone(rig, name)

pairs = {
    "hips"    : ("mixamorig:Hips",              None, None),
    "spine"   : ("mixamorig:Spine",             mp_LH, mp_LS),  # rough aim shoulder dir
    "spine2"  : ("mixamorig:Spine2",            mp_LH, mp_LS),
    "neck"    : ("mixamorig:Neck",              mp_LS, 0),      # neck aims toward nose
    "head"    : ("mixamorig:Head",              None, 0),

    "l_arm"   : ("mixamorig:LeftArm",           11, 13),
    "l_fore"  : ("mixamorig:LeftForeArm",       13, 15),
    "l_hand"  : ("mixamorig:LeftHand",          None, 15),

    "r_arm"   : ("mixamorig:RightArm",          12, 14),
    "r_fore"  : ("mixamorig:RightForeArm",      14, 16),
    "r_hand"  : ("mixamorig:RightHand",         None, 16),

    "l_up"    : ("mixamorig:LeftUpLeg",         23, 25),
    "l_low"   : ("mixamorig:LeftLeg",           25, 27),
    "l_foot"  : ("mixamorig:LeftFoot",          None, 27),

    "r_up"    : ("mixamorig:RightUpLeg",        24, 26),
    "r_low"   : ("mixamorig:RightLeg",          26, 28),
    "r_foot"  : ("mixamorig:RightFoot",         None, 28),
}

# Some rigs don’t have mixamorig: prefix – fallbacks
def resolve(name):
    pb = b(name)
    if pb: return pb
    return b(name.split(":")[-1])

for key, (bone_name, src_idx, aim_idx) in pairs.items():
    pb = resolve(bone_name)
    if not pb: continue
    add_clear_constraints(pb)

    # Hips: copy location from hip center (mid of 23,24) – we approximate with MP_ROOT
    if key == "hips":
        add_loc_constraint(pb, mp_root, "")
        continue

    # If we have a driving point for this bone, add damped track to the "aim" target.
    if aim_idx is not None:
        add_track_constraint(pb, bpy.data.objects.get(f"MP_{aim_idx}"), f"MP_{aim_idx}")

# 6) Bake to the character rig and cleanup
frame_end = T
bake_pose(rig, START_FRAME, START_FRAME + frame_end - 1)

# Optional: delete drivers to keep scene clean after bake
bpy.ops.object.mode_set(mode='OBJECT')
for o in list(bpy.data.objects):
    if o.name == "MP_ROOT" or o.name.startswith("MP_"):
        o.select_set(True)
    else:
        o.select_set(False)
bpy.ops.object.delete()

# 7) Export FBX
bpy.ops.object.select_all(action='DESELECT')
rig.select_set(True)
bpy.context.view_layer.objects.active = rig

print(f"\nFBX export starting… '{OUT_FBX}'")
bpy.ops.export_scene.fbx(
    filepath=OUT_FBX,
    use_selection=True,
    bake_anim=True,
    add_leaf_bones=False,
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False
)
print(f"✅ FBX written: {OUT_FBX}")

# 8) (Optional) MP4 preview render
if RENDER_PREVIEW:
    ensure_camera_light()
    setup_render(FPS, OUT_MP4, frame_end)
    print(f"\n🎥 Rendering MP4 preview to '{OUT_MP4}' …")
    bpy.ops.render.render(animation=True)
    print(f"✅ MP4 written: {OUT_MP4}")
