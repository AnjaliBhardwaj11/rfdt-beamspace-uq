#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path

import bpy
import mathutils
import numpy as np

try:
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from uq_config import load_config, resolve_path
except Exception:
    load_config = None
    resolve_path = None


DEFAULT_VISUAL_CFG = {
    "input_models_dir": "meshes_d",
    "output_dataset_dir": "dataset_visual_v2_with_object",
    "num_images": 300,
    "resolution": 800,
    "camera_lens_mm": 20.0,
    "room_min": [0.5, 0.5, 0.0],
    "room_max": [6.5, 4.5, 3.0],
    "cycles_samples": 96,
    "max_bounces": 3,
    "use_denoising": True,
    "denoiser": "OPENIMAGEDENOISE",
    "use_gpu": True,
    "gpu_backend": "CUDA",
    "seed": 1234,
    "test_fraction": 0.1,
    "strategy_split": {
        "perimeter": 0.34,
        "detail": 0.33,
        "topdown": 0.33,
    },
    "focus_orbit": {
        "enabled": True,
        "frames_per_object": 25,
        "keywords": ["chair", "table", "sofa", "tv", "desk"],
    },
}


MATERIAL_COLORS = {
    "walls": (0.85, 0.85, 0.85, 1),
    "floor": (0.25, 0.25, 0.25, 1),
    "ceiling": (0.95, 0.95, 0.95, 1),
    "door": (0.4, 0.2, 0.1, 1),
    "window": (0.7, 0.8, 1.0, 1),
    "furniture": (0.55, 0.35, 0.15, 1),
    "led_tv": (0.05, 0.05, 0.05, 1),
}


INPUT_MODELS_DIR = ""
OUTPUT_DATASET_DIR = ""
NUM_IMAGES = 0
RESOLUTION = 0
CAMERA_LENS_MM = 0.0
ROOM_MIN = mathutils.Vector((0.0, 0.0, 0.0))
ROOM_MAX = mathutils.Vector((0.0, 0.0, 0.0))
CENTER = mathutils.Vector((0.0, 0.0, 0.0))
TEST_FRACTION = 0.1
STRATEGY_SPLIT = {}
FOCUS_ORBIT_CFG = {}
CYCLES_SAMPLES = 96
MAX_BOUNCES = 3
USE_DENOISING = True
DENOISER = "OPENIMAGEDENOISE"
USE_GPU = True
GPU_BACKEND = "CUDA"


def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a visual dataset for RF-3DGS using Blender."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to uq_config.json")
    parser.add_argument("--input-models-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--camera-lens-mm", type=float, default=None)

    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def resolve_local_path(path_str, base_dir):
    if not path_str:
        return ""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def load_visual_config(args):
    cfg = copy.deepcopy(DEFAULT_VISUAL_CFG)
    base_dir = Path.cwd()

    if args.config:
        if load_config:
            merged_cfg, base_dir = load_config(args.config)
            visual_cfg = merged_cfg.get("preprocess", {}).get("visual", {})
            deep_update(cfg, visual_cfg)
        else:
            cfg_path = Path(args.config).expanduser().resolve()
            base_dir = cfg_path.parent
            with cfg_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            visual_cfg = raw.get("preprocess", {}).get("visual", {})
            deep_update(cfg, visual_cfg)

    if args.input_models_dir:
        cfg["input_models_dir"] = args.input_models_dir
    if args.output_dir:
        cfg["output_dataset_dir"] = args.output_dir
    if args.num_images is not None:
        cfg["num_images"] = int(args.num_images)
    if args.resolution is not None:
        cfg["resolution"] = int(args.resolution)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.camera_lens_mm is not None:
        cfg["camera_lens_mm"] = float(args.camera_lens_mm)

    if resolve_path:
        input_dir = resolve_path(cfg["input_models_dir"], base_dir)
        output_dir = resolve_path(cfg["output_dataset_dir"], base_dir)
    else:
        input_dir = resolve_local_path(cfg["input_models_dir"], base_dir)
        output_dir = resolve_local_path(cfg["output_dataset_dir"], base_dir)

    cfg["input_models_dir"] = input_dir
    cfg["output_dataset_dir"] = output_dir
    return cfg


def configure_from_config(cfg):
    global INPUT_MODELS_DIR, OUTPUT_DATASET_DIR, NUM_IMAGES, RESOLUTION
    global ROOM_MIN, ROOM_MAX, CENTER, CAMERA_LENS_MM
    global TEST_FRACTION, STRATEGY_SPLIT, FOCUS_ORBIT_CFG
    global CYCLES_SAMPLES, MAX_BOUNCES, USE_DENOISING, DENOISER
    global USE_GPU, GPU_BACKEND

    INPUT_MODELS_DIR = cfg["input_models_dir"]
    OUTPUT_DATASET_DIR = cfg["output_dataset_dir"]
    NUM_IMAGES = int(cfg["num_images"])
    RESOLUTION = int(cfg["resolution"])
    CAMERA_LENS_MM = float(cfg["camera_lens_mm"])

    ROOM_MIN = mathutils.Vector(tuple(cfg["room_min"]))
    ROOM_MAX = mathutils.Vector(tuple(cfg["room_max"]))
    CENTER = (ROOM_MIN + ROOM_MAX) / 2

    TEST_FRACTION = float(cfg.get("test_fraction", 0.1))
    STRATEGY_SPLIT = cfg.get("strategy_split", {})
    FOCUS_ORBIT_CFG = cfg.get("focus_orbit", {})

    CYCLES_SAMPLES = int(cfg.get("cycles_samples", 96))
    MAX_BOUNCES = int(cfg.get("max_bounces", 3))
    USE_DENOISING = bool(cfg.get("use_denoising", True))
    DENOISER = str(cfg.get("denoiser", "OPENIMAGEDENOISE"))
    USE_GPU = bool(cfg.get("use_gpu", True))
    GPU_BACKEND = str(cfg.get("gpu_backend", "CUDA"))

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def setup_render_engine():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = CYCLES_SAMPLES
    scene.cycles.use_denoising = USE_DENOISING
    scene.cycles.denoiser = DENOISER
    scene.cycles.max_bounces = MAX_BOUNCES

    if USE_GPU:
        scene.cycles.device = 'GPU'
        try:
            prefs = bpy.context.preferences
            cycles_prefs = prefs.addons['cycles'].preferences
            cycles_prefs.refresh_devices()
            cycles_prefs.compute_device_type = GPU_BACKEND

            for device in cycles_prefs.devices:
                if device.type in ('CUDA', 'OPTIX', 'HIP'):
                    device.use = True
                    print(f"Enabled GPU: {device.name} ({device.type})")
        except Exception as e:
            print(f"GPU setup warning: {e}")
            print("Falling back to CPU rendering")
            scene.cycles.device = 'CPU'
    else:
        scene.cycles.device = 'CPU'
    if hasattr(scene.view_settings, 'view_transform'):
        scene.view_settings.view_transform = 'Standard' 
    
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.film_transparent = True

def create_high_feature_material(name, color, is_glass=False):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    bsdf.inputs['Base Color'].default_value = color
    
    if is_glass:
        bsdf.inputs['Transmission Weight'].default_value = 0.65  
        bsdf.inputs['Roughness'].default_value = 0.3 
        noise = nodes.new(type='ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = 50.0 
        links.new(noise.outputs['Fac'], bsdf.inputs['Alpha'])
        mat.blend_method = 'BLEND' 

    else:
        noise_large = nodes.new(type='ShaderNodeTexNoise')
        noise_large.inputs['Scale'].default_value = 15.0
        
        noise_small = nodes.new(type='ShaderNodeTexNoise')
        noise_small.inputs['Scale'].default_value = 100.0
        
        mix_rgb = nodes.new(type='ShaderNodeMixRGB')
        mix_rgb.blend_type = 'ADD'
        mix_rgb.inputs['Fac'].default_value = 0.5
        links.new(noise_large.outputs['Fac'], mix_rgb.inputs['Color1'])
        links.new(noise_small.outputs['Fac'], mix_rgb.inputs['Color2'])
        
        bump = nodes.new(type='ShaderNodeBump')
        bump.inputs['Strength'].default_value = 0.2 
        
        links.new(mix_rgb.outputs['Color'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
        bsdf.inputs['Roughness'].default_value = 0.8 

    return mat

def setup_lighting():
    scene = bpy.context.scene
    
    if not scene.world:
        new_world = bpy.data.worlds.new("World")
        scene.world = new_world

    world = scene.world
    world.use_nodes = True
    
    bg = None
    if 'Background' in world.node_tree.nodes:
        bg = world.node_tree.nodes['Background']
    else:
        bg = world.node_tree.nodes.new('ShaderNodeBackground')
        output = world.node_tree.nodes.new('ShaderNodeOutputWorld')
        world.node_tree.links.new(bg.outputs['Background'], output.inputs['Surface'])
        
    bg.inputs['Color'].default_value = (1, 1, 1, 1)
    bg.inputs['Strength'].default_value = 0.6

    bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
    sun = bpy.context.object
    sun.data.energy = 2.5
    sun.data.angle = 0.5 
    sun.rotation_euler = (math.radians(45), math.radians(15), 0)

    bpy.ops.object.light_add(type='AREA', location=(3.5, 2.5, 2.9)) 
    ceiling_light = bpy.context.object
    ceiling_light.name = "Ceiling_Panel"
    ceiling_light.data.energy = 250.0
    ceiling_light.data.size = 4.0     
    ceiling_light.data.color = (1.0, 0.98, 0.9) # RGB only

def import_models():
    if not os.path.exists(INPUT_MODELS_DIR):
        print(f"Input models directory not found: {INPUT_MODELS_DIR}")
        return
    files = [f for f in os.listdir(INPUT_MODELS_DIR) if f.endswith('.ply')]
    
    for filename in files:
        full_path = os.path.join(INPUT_MODELS_DIR, filename)
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.wm.ply_import(filepath=full_path)
        
        if not bpy.context.selected_objects: continue
        obj = bpy.context.selected_objects[0]
        
        color = (0.5, 0.5, 0.5, 1)
        is_glass = False
        fname_lower = filename.lower()
        for keyword, mapped_color in MATERIAL_COLORS.items():
            if keyword in fname_lower:
                color = mapped_color
                if "glass" in keyword or "window" in keyword: is_glass = True
                break
        
        mat = create_high_feature_material(f"Mat_{filename}", color, is_glass)
        if obj.data.materials: obj.data.materials[0] = mat
        else: obj.data.materials.append(mat)

def look_at(obj, target_pos):
    """Rotates camera to look at target vector."""
    direction = target_pos - obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = rot_quat.to_euler()

def clamp_to_room(position, margin=0.3):
    """Clamps camera position to stay within room boundaries with margin."""
    x = max(ROOM_MIN.x + margin, min(position[0], ROOM_MAX.x - margin))
    y = max(ROOM_MIN.y + margin, min(position[1], ROOM_MAX.y - margin))
    z = max(ROOM_MIN.z + margin, min(position[2], ROOM_MAX.z - margin))
    return mathutils.Vector((x, y, z))

def generate_focus_orbit(target_obj, frames_list, images_dir, start_index, num_frames=30):
    """Generate a tight orbit around a target object."""
    center = target_obj.location
    max_radius_x = min(center.x - ROOM_MIN.x, ROOM_MAX.x - center.x) - 0.5
    max_radius_y = min(center.y - ROOM_MIN.y, ROOM_MAX.y - center.y) - 0.5
    safe_radius = min(max_radius_x, max_radius_y, 1.2)
    safe_radius = max(0.6, safe_radius)
    
    print(f"Generating focus scan for: {target_obj.name} at {center}, radius: {safe_radius:.2f}m")

    for i in range(num_frames):
        t = i / num_frames
        angle = t * 2 * math.pi
        
        desired_z = 2.0 - (t * 1.2)
        current_z = max(ROOM_MIN.z + 0.5, min(desired_z, ROOM_MAX.z - 0.3))
        cam_x = center.x + math.cos(angle) * safe_radius
        cam_y = center.y + math.sin(angle) * safe_radius
        cam = bpy.context.scene.camera
        cam.location = clamp_to_room((cam_x, cam_y, current_z))
        look_target = center + mathutils.Vector((0, 0, 0.5))
        look_at(cam, look_target)
        render_frame(start_index + i, cam, images_dir, frames_list)

def main():
    args = parse_args()
    cfg = load_visual_config(args)
    configure_from_config(cfg)

    random.seed(cfg.get("seed", 1234))
    np.random.seed(cfg.get("seed", 1234))

    reset_scene()
    setup_render_engine()
    import_models()
    setup_lighting()
    
    images_dir = os.path.join(OUTPUT_DATASET_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    cam.data.lens = CAMERA_LENS_MM
    
    frames = []
    
    split = STRATEGY_SPLIT or {"perimeter": 0.34, "detail": 0.33, "topdown": 0.33}
    total_frac = float(split.get("perimeter", 0.0)) + float(split.get("detail", 0.0)) + float(split.get("topdown", 0.0))
    if total_frac <= 0:
        total_frac = 1.0
    perimeter_frac = float(split.get("perimeter", 0.34)) / total_frac
    detail_frac = float(split.get("detail", 0.33)) / total_frac

    perimeter_count = int(NUM_IMAGES * perimeter_frac)
    detail_count = int(NUM_IMAGES * detail_frac)
    topdown_count = max(0, NUM_IMAGES - perimeter_count - detail_count)

    frame_counter = 0

    # Strategy 1: perimeter walk
    for i in range(perimeter_count):
        t = i / max(1, perimeter_count)
        angle = t * 2 * math.pi
        
        # Oval path slightly smaller than room limits
        radius_x = (ROOM_MAX.x - ROOM_MIN.x) * 0.35  # Reduced from 0.4 for safety
        radius_y = (ROOM_MAX.y - ROOM_MIN.y) * 0.35
        
        x = CENTER.x + math.cos(angle) * radius_x
        y = CENTER.y + math.sin(angle) * radius_y
        z = 1.6 # Eye level
        
        cam.location = clamp_to_room((x, y, z))
        
        # Look at a point slightly offset from center to create parallax
        look_target = CENTER + mathutils.Vector((math.cos(angle*2)*0.5, math.sin(angle*2)*0.5, -0.5))
        look_at(cam, look_target)
        
        render_frame(frame_counter, cam, images_dir, frames)
        frame_counter += 1

    # Strategy 2: low detail pass
    for i in range(detail_count):
        x = random.uniform(ROOM_MIN.x + 0.8, ROOM_MAX.x - 0.8)
        y = random.uniform(ROOM_MIN.y + 0.8, ROOM_MAX.y - 0.8)
        z = random.uniform(0.5, 1.0)
        
        cam.location = clamp_to_room((x, y, z))
        
        target_x = x + random.uniform(-1, 1)
        target_y = y + random.uniform(-1, 1)
        target_z = random.uniform(0.8, 1.5)
        
        look_at(cam, mathutils.Vector((target_x, target_y, target_z)))
        render_frame(frame_counter, cam, images_dir, frames)
        frame_counter += 1

    # Strategy 3: top-down filler
    for i in range(topdown_count):
        t = i / max(1, topdown_count)
        x = ROOM_MIN.x + 0.8 + (ROOM_MAX.x - ROOM_MIN.x - 1.6) * t
        y = CENTER.y + math.sin(t * 10 * math.pi) * 1.2
        z = 2.5
        
        cam.location = clamp_to_room((x, y, z))
        
        look_at(cam, mathutils.Vector((x, y, 0.0)))
        render_frame(frame_counter, cam, images_dir, frames)
        frame_counter += 1

    # Strategy 4: object focus orbits
    if FOCUS_ORBIT_CFG.get("enabled", True):
        keywords = FOCUS_ORBIT_CFG.get("keywords", ["chair", "table", "sofa", "tv", "desk"])
        frames_per_object = int(FOCUS_ORBIT_CFG.get("frames_per_object", 25))
        target_objects = []

        for obj in bpy.context.scene.objects:
            if any(k in obj.name.lower() for k in keywords):
                target_objects.append(obj)

        for obj in target_objects:
            generate_focus_orbit(
                target_obj=obj,
                frames_list=frames,
                images_dir=images_dir,
                start_index=frame_counter,
                num_frames=frames_per_object,
            )
            frame_counter += frames_per_object

    # Split: train/test
    num_frames = len(frames)
    if num_frames == 0:
        print("No frames rendered. Skipping transforms export.")
        return

    test_fraction = min(max(TEST_FRACTION, 0.0), 0.5)
    num_test = max(1, int(round(num_frames * test_fraction)))
    num_test = min(num_test, num_frames)
    test_indices = set(np.linspace(0, num_frames - 1, num_test, dtype=int).tolist())
    
    train_frames = [f for i, f in enumerate(frames) if i not in test_indices]
    test_frames = [f for i, f in enumerate(frames) if i in test_indices]
    
    train_data = {"camera_angle_x": cam.data.angle_x, "frames": train_frames}
    test_data = {"camera_angle_x": cam.data.angle_x, "frames": test_frames}
    
    with open(os.path.join(OUTPUT_DATASET_DIR, "transforms_train.json"), 'w') as f:
        json.dump(train_data, f, indent=4)
    
    with open(os.path.join(OUTPUT_DATASET_DIR, "transforms_test.json"), 'w') as f:
        json.dump(test_data, f, indent=4)
    
    print(f"Done! Generated {len(train_frames)} train and {len(test_frames)} test frames")


def render_frame(index, cam, images_dir, frames_list):
    bpy.context.view_layer.update()
    filename = f"frame_{index:04d}.png"
    filepath = os.path.join(images_dir, filename)
    bpy.context.scene.render.filepath = filepath
    
    # Suppress output for speed
    with open(os.devnull, 'w') as fnull:
         bpy.ops.render.render(write_still=True)
            
    # Standard NeRF/3DGS matrix format
    matrix = cam.matrix_world
    frames_list.append({
        "file_path": f"images/frame_{index:04d}",  # No .png extension
        "transform_matrix": [list(row) for row in matrix]
    })
    print(f"Rendered {filename}")

if __name__ == "__main__":
    main()