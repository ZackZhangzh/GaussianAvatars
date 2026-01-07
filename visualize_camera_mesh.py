#!/usr/bin/env python3
"""
Camera-Mesh Visualization Script (DearPyGui Version)

用于验证 train.py 中相机空间与导入 mesh 的位置关系。
读取数据集的方式完全与 train.py 一致。
使用与项目一致的 dearpygui + nvdiffrast 技术栈。

Usage:
    # Interactive 3D viewer
    python visualize_camera_mesh.py --source-path ${SOURCE_PATH} --mesh-path ${MESH_PATH}
    
    # Output single image for camera N (left: original, right: mesh overlay)
    python visualize_camera_mesh.py --source-path ${SOURCE_PATH} --mesh-path ${MESH_PATH} --vis-cam 0

Dependencies (all in requirements.txt):
    - dearpygui, nvdiffrast, torch, numpy, scipy
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal
from PIL import Image
import tyro

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mesh_renderer import NVDiffRenderer
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


@dataclass
class Config:
    """Configuration for Camera-Mesh Visualizer"""
    source_path: str = ""
    """Path to the dataset (same as train.py -s)"""
    mesh_path: str = ""
    """Path to the custom mesh (OBJ format)"""
    vis_cam: Optional[int] = None
    """If set, output a side-by-side image for camera N (left: original, right: mesh overlay)"""
    output_path: Optional[str] = None
    """Output path for vis_cam image (default: vis_cam_{N}.png in source_path)"""
    mesh_opacity: float = 0.5
    """Mesh opacity for overlay (0.0-1.0)"""
    mesh_color: tuple = (0.3, 0.7, 1.0)
    """Mesh color RGB (0.0-1.0)"""
    # Interactive viewer settings
    cam_convention: Literal["opengl", "opencv"] = "opencv"
    """Camera convention"""
    background_color: tuple = (0.2, 0.2, 0.2)
    """Background color"""
    W: int = 1280
    """Window width"""
    H: int = 720
    """Window height"""
    radius: float = 2.0
    """Default camera radius"""
    fovy: float = 30
    """Default field of view"""
    camera_size: float = 0.02
    """Size of camera frustum visualization"""
    show_cameras: bool = True
    """Whether to show camera positions"""


def load_cameras_from_transforms(path: str) -> list:
    """Load camera information from transforms_*.json files."""
    cameras = []
    transform_files = ["transforms_train.json", "transforms_val.json", "transforms_test.json"]
    
    for tf_file in transform_files:
        tf_path = os.path.join(path, tf_file)
        if not os.path.exists(tf_path):
            continue
        
        print(f"  Loading cameras from: {tf_file}")
        with open(tf_path, 'r') as f:
            contents = json.load(f)
        
        frames = contents.get("frames", [])
        fovx_shared = contents.get("camera_angle_x", None)
        
        for idx, frame in enumerate(frames):
            c2w = np.array(frame["transform_matrix"])
            c2w_opencv = c2w.copy()
            c2w_opencv[:3, 1:3] *= -1
            
            w2c = np.linalg.inv(c2w_opencv)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            
            camera_center = c2w_opencv[:3, 3]
            camera_forward = -c2w_opencv[:3, 2]
            camera_up = c2w_opencv[:3, 1]
            camera_right = c2w_opencv[:3, 0]
            
            # Get image path
            file_path = frame.get("file_path", "")
            if file_path and not file_path.endswith(".png"):
                file_path += ".png"
            image_path = os.path.join(path, file_path) if file_path else None
            
            # Get image dimensions
            width = frame.get("w", 512)
            height = frame.get("h", 512)
            
            cameras.append({
                'idx': len(cameras),  # Global index
                'local_idx': idx,  # Index within this file
                'source': tf_file,
                'c2w': c2w,
                'c2w_opencv': c2w_opencv,
                'R': R,
                'T': T,
                'camera_center': camera_center,
                'camera_forward': camera_forward,
                'camera_up': camera_up,
                'camera_right': camera_right,
                'timestep': frame.get('timestep_index', 0),
                'fovx': frame.get('camera_angle_x', fovx_shared),
                'image_path': image_path,
                'width': width,
                'height': height,
            })
    
    print(f"  Loaded {len(cameras)} cameras in total")
    return cameras


def load_mesh(mesh_path: str):
    """Load mesh from OBJ file."""
    try:
        import trimesh
        mesh = trimesh.load(mesh_path, force='mesh')
        print(f"  Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return np.array(mesh.vertices, dtype=np.float32), np.array(mesh.faces, dtype=np.int32)
    except ImportError:
        print("  [WARN] trimesh not installed. Using manual OBJ parsing...")
        return parse_obj_manual(mesh_path)


def parse_obj_manual(obj_path: str):
    """Simple OBJ parser."""
    vertices = []
    faces = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                face_verts = []
                for p in parts[1:4]:
                    v_idx = int(p.split('/')[0]) - 1
                    face_verts.append(v_idx)
                faces.append(face_verts)
    
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def render_mesh_overlay(cam_info: dict, mesh_verts: np.ndarray, mesh_faces: np.ndarray,
                        mesh_renderer: NVDiffRenderer, mesh_color: tuple, mesh_opacity: float) -> np.ndarray:
    """
    Render mesh from a specific camera viewpoint and overlay on original image.
    
    Returns: RGBA overlay image as numpy array [H, W, 4]
    """
    # Get camera parameters
    R = cam_info['R']
    T = cam_info['T']
    fovx = cam_info['fovx']
    width = cam_info['width']
    height = cam_info['height']
    
    # Calculate fovy from fovx
    fovy = 2 * np.arctan(np.tan(fovx / 2) * height / width)
    
    # Build world_view_transform (same as in scene/cameras.py)
    world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).float().cuda()
    
    # Build projection matrix
    projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).float().cuda()
    
    # Full projection
    full_proj_transform = world_view_transform @ projection_matrix
    
    # Create MiniCam
    cam = MiniCam(
        width=width,
        height=height,
        fovy=fovy,
        fovx=fovx,
        znear=0.01,
        zfar=100.0,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
        timestep=0,
    )
    
    # Prepare mesh for rendering
    mesh_verts_gpu = torch.from_numpy(mesh_verts).float().cuda().unsqueeze(0)
    mesh_faces_gpu = torch.from_numpy(mesh_faces).int().cuda()
    
    # Render mesh with transparent background
    out_dict = mesh_renderer.render_from_camera(
        mesh_verts_gpu,
        mesh_faces_gpu,
        cam,
        background_color=[0, 0, 0],
    )
    
    rgba = out_dict['rgba'].squeeze(0).cpu().numpy()  # [H, W, 4]
    
    # Apply mesh color and opacity
    mesh_rgb = rgba[:, :, :3] * np.array(mesh_color)[None, None, :]
    mesh_alpha = rgba[:, :, 3:4] * mesh_opacity
    
    return np.concatenate([mesh_rgb, mesh_alpha], axis=-1)


def generate_vis_cam_image(cfg: Config, cameras: list, mesh_verts: np.ndarray, mesh_faces: np.ndarray):
    """
    Generate side-by-side visualization for a specific camera.
    Left: Original image
    Right: Original image with mesh overlay
    """
    cam_idx = cfg.vis_cam
    
    if cam_idx < 0 or cam_idx >= len(cameras):
        print(f"[ERROR] Camera index {cam_idx} out of range (0-{len(cameras)-1})")
        sys.exit(1)
    
    cam_info = cameras[cam_idx]
    print(f"\n[INFO] Generating visualization for camera {cam_idx}")
    print(f"  Source: {cam_info['source']}")
    print(f"  Image path: {cam_info['image_path']}")
    print(f"  Resolution: {cam_info['width']}x{cam_info['height']}")
    
    # Load original image
    if cam_info['image_path'] and os.path.exists(cam_info['image_path']):
        original_img = np.array(Image.open(cam_info['image_path']).convert('RGB')) / 255.0
        height, width = original_img.shape[:2]
        # Update camera dimensions if needed
        cam_info['width'] = width
        cam_info['height'] = height
    else:
        print(f"[WARN] Original image not found: {cam_info['image_path']}")
        print("[INFO] Creating blank placeholder")
        width, height = cam_info['width'], cam_info['height']
        original_img = np.ones((height, width, 3)) * 0.5  # Gray placeholder
    
    # Initialize mesh renderer
    print("[INFO] Initializing mesh renderer...")
    mesh_renderer = NVDiffRenderer(use_opengl=False)
    
    # Render mesh overlay
    print("[INFO] Rendering mesh overlay...")
    overlay_rgba = render_mesh_overlay(
        cam_info, mesh_verts, mesh_faces, mesh_renderer,
        cfg.mesh_color, cfg.mesh_opacity
    )
    
    # Composite overlay on original image
    overlay_rgb = overlay_rgba[:, :, :3]
    overlay_alpha = overlay_rgba[:, :, 3:4]
    
    composite_img = original_img * (1 - overlay_alpha) + overlay_rgb * overlay_alpha
    
    # Create side-by-side image
    separator_width = 4
    separator = np.ones((height, separator_width, 3)) * 0.8  # Light gray separator
    
    side_by_side = np.concatenate([
        original_img,
        separator,
        composite_img,
    ], axis=1)
    
    # Add labels
    # (Using PIL for text rendering)
    from PIL import ImageDraw, ImageFont
    
    side_by_side_pil = Image.fromarray((side_by_side * 255).astype(np.uint8))
    draw = ImageDraw.Draw(side_by_side_pil)
    
    # Try to use a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw labels with background
    label_left = f"Camera {cam_idx} - Original"
    label_right = f"Camera {cam_idx} - Mesh Overlay"
    
    # Left label
    draw.rectangle([(5, 5), (len(label_left) * 10 + 15, 30)], fill=(0, 0, 0, 180))
    draw.text((10, 8), label_left, fill=(255, 255, 255), font=font)
    
    # Right label
    right_x = width + separator_width + 10
    draw.rectangle([(right_x - 5, 5), (right_x + len(label_right) * 10 + 10, 30)], fill=(0, 0, 0, 180))
    draw.text((right_x, 8), label_right, fill=(255, 255, 255), font=font)
    
    # Determine output path
    if cfg.output_path:
        output_path = cfg.output_path
    else:
        output_path = os.path.join(cfg.source_path, f"vis_cam_{cam_idx}.png")
    
    # Save image
    side_by_side_pil.save(output_path)
    print(f"\n[SUCCESS] Saved visualization to: {output_path}")
    print(f"  Left: Original camera image")
    print(f"  Right: Image with mesh overlay (opacity={cfg.mesh_opacity})")
    
    return output_path


def run_interactive_viewer(cfg: Config, cameras: list, mesh_verts: np.ndarray, mesh_faces: np.ndarray):
    """Run the interactive DearPyGui viewer."""
    # Import viewer components
    from utils.viewer_utils import Mini3DViewer, Mini3DViewerConfig, OrbitCamera
    import dearpygui.dearpygui as dpg
    
    class CameraMeshViewer(Mini3DViewer):
        """Interactive viewer for camera-mesh relationship visualization"""
        
        def __init__(self, cfg, cameras, mesh_verts, mesh_faces):
            self.cfg = cfg
            self.mesh_verts = mesh_verts
            self.mesh_faces = mesh_faces
            self.cameras = cameras
            self.mesh_color = torch.tensor([0.7, 0.8, 0.9, 1.0])
            
            # Convert to torch tensors for GPU rendering
            self.mesh_verts_gpu = torch.from_numpy(mesh_verts).float().cuda().unsqueeze(0)
            self.mesh_faces_gpu = torch.from_numpy(mesh_faces).int().cuda()
            
            # Create camera frustum meshes
            self.camera_meshes = self.create_camera_meshes()
            
            # Initialize mesh renderer
            print("Initializing mesh renderer...")
            self.mesh_renderer = NVDiffRenderer(use_opengl=False)
            
            # Set initial view to center on mesh
            mesh_center = mesh_verts.mean(axis=0)
            mesh_extent = (mesh_verts.max(axis=0) - mesh_verts.min(axis=0)).max()
            cfg.radius = mesh_extent * 2.5
            
            # Initialize parent class
            super().__init__(cfg, 'Camera-Mesh Visualizer')
            
            # Set camera look_at to mesh center
            self.cam.look_at = mesh_center.astype(np.float32)
            
            # Start render loop
            self.run()
        
        def create_camera_meshes(self):
            """Create frustum meshes for all cameras."""
            all_verts = []
            all_faces = []
            all_colors = []
            
            scale = self.cfg.camera_size
            vert_offset = 0
            
            for cam in self.cameras:
                center = cam['camera_center']
                forward = cam['camera_forward']
                up = cam['camera_up']
                right = cam['camera_right']
                
                fov_scale = 0.6
                near = scale
                
                verts = np.array([
                    center,
                    center + forward * near + up * near * fov_scale + right * near,
                    center + forward * near + up * near * fov_scale - right * near,
                    center + forward * near - up * near * fov_scale - right * near,
                    center + forward * near - up * near * fov_scale + right * near,
                ], dtype=np.float32)
                
                faces = np.array([
                    [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
                    [1, 3, 2], [1, 4, 3],
                ], dtype=np.int32) + vert_offset
                
                if 'train' in cam['source']:
                    color = [0.2, 0.8, 0.2, 0.8]
                elif 'val' in cam['source']:
                    color = [0.8, 0.8, 0.2, 0.8]
                else:
                    color = [0.2, 0.4, 0.9, 0.8]
                
                all_verts.append(verts)
                all_faces.append(faces)
                all_colors.extend([color] * len(faces))
                vert_offset += len(verts)
            
            if all_verts:
                return {
                    'verts': torch.from_numpy(np.concatenate(all_verts)).float().cuda().unsqueeze(0),
                    'faces': torch.from_numpy(np.concatenate(all_faces)).int().cuda(),
                    'colors': torch.from_numpy(np.array(all_colors, dtype=np.float32)).float().cuda().unsqueeze(0),
                }
            return None
        
        def define_gui(self):
            super().define_gui()
            
            with dpg.window(label="Controls", tag="_control_window", autosize=True, pos=(10, 10)):
                dpg.add_text(f"Cameras: {len(self.cameras)}")
                dpg.add_text(f"Mesh vertices: {len(self.mesh_verts)}")
                dpg.add_text(f"Mesh faces: {len(self.mesh_faces)}")
                dpg.add_separator()
                
                def callback_show_cameras(sender, app_data):
                    self.cfg.show_cameras = app_data
                    self.need_update = True
                dpg.add_checkbox(label="Show Cameras", default_value=self.cfg.show_cameras,
                               callback=callback_show_cameras)
                
                def callback_camera_size(sender, app_data):
                    self.cfg.camera_size = app_data
                    self.camera_meshes = self.create_camera_meshes()
                    self.need_update = True
                dpg.add_slider_float(label="Camera Size", min_value=0.001, max_value=0.1,
                                   default_value=self.cfg.camera_size, format="%.3f",
                                   callback=callback_camera_size)
                
                def callback_mesh_opacity(sender, app_data):
                    self.mesh_color[3] = app_data
                    self.need_update = True
                dpg.add_slider_float(label="Mesh Opacity", min_value=0.0, max_value=1.0,
                                   default_value=1.0, callback=callback_mesh_opacity)
                
                dpg.add_separator()
                
                def callback_reset_view(sender, app_data):
                    self.cam.reset()
                    self.cam.look_at = self.mesh_verts.mean(axis=0).astype(np.float32)
                    self.need_update = True
                dpg.add_button(label="Reset View", callback=callback_reset_view)
                
                dpg.add_separator()
                dpg.add_text("Controls:", color=(200, 200, 200))
                dpg.add_text("  Left mouse: Rotate", color=(150, 150, 150))
                dpg.add_text("  Right mouse: Pan", color=(150, 150, 150))
                dpg.add_text("  Scroll: Zoom", color=(150, 150, 150))
        
        def render(self):
            cam = MiniCam(
                width=self.W,
                height=self.H,
                fovy=np.radians(self.cam.fovy),
                fovx=np.radians(self.cam.fovx),
                znear=self.cam.znear,
                zfar=self.cam.zfar,
                world_view_transform=torch.from_numpy(self.cam.world_view_transform.T).float().cuda(),
                full_proj_transform=torch.from_numpy(self.cam.full_proj_transform.T).float().cuda(),
                timestep=0,
            )
            
            bg_color = list(self.cfg.background_color)
            
            out_dict = self.mesh_renderer.render_from_camera(
                self.mesh_verts_gpu, self.mesh_faces_gpu, cam, background_color=bg_color)
            rgba = out_dict['rgba'].squeeze(0)
            
            rgb = rgba[:, :, :3] * self.mesh_color[:3].cuda() * self.mesh_color[3].cuda()
            alpha = rgba[:, :, 3:4]
            bg = torch.tensor(bg_color, device='cuda')[None, None, :]
            image = rgb * alpha + bg * (1 - alpha)
            
            if self.cfg.show_cameras and self.camera_meshes is not None:
                cam_out = self.mesh_renderer.render_from_camera(
                    self.camera_meshes['verts'], self.camera_meshes['faces'], cam,
                    background_color=[0, 0, 0], face_colors=self.camera_meshes['colors'])
                cam_rgba = cam_out['rgba'].squeeze(0)
                cam_rgb = cam_rgba[:, :, :3]
                cam_alpha = cam_rgba[:, :, 3:4]
                image = cam_rgb * cam_alpha + image * (1 - cam_alpha)
            
            return image.clamp(0, 1).cpu().numpy()
        
        def run(self):
            while dpg.is_dearpygui_running():
                if self.need_update:
                    self.render_buffer = self.render()
                    dpg.set_value("_texture", self.render_buffer)
                    self.need_update = False
                dpg.render_dearpygui_frame()
            dpg.destroy_context()
    
    viewer = CameraMeshViewer(cfg, cameras, mesh_verts, mesh_faces)


def main():
    cfg = tyro.cli(Config)
    
    # Validate paths
    if not cfg.source_path:
        print("[ERROR] --source-path is required")
        sys.exit(1)
    if not cfg.mesh_path:
        print("[ERROR] --mesh-path is required")
        sys.exit(1)
    if not os.path.exists(cfg.source_path):
        print(f"[ERROR] Source path does not exist: {cfg.source_path}")
        sys.exit(1)
    if not os.path.exists(cfg.mesh_path):
        print(f"[ERROR] Mesh path does not exist: {cfg.mesh_path}")
        sys.exit(1)
    
    print(f"\n{'=' * 60}")
    print("Camera-Mesh Visualizer")
    print(f"{'=' * 60}")
    print(f"Source path: {cfg.source_path}")
    print(f"Mesh path: {cfg.mesh_path}")
    
    # Load data
    print("\nLoading cameras...")
    cameras = load_cameras_from_transforms(cfg.source_path)
    
    print("\nLoading mesh...")
    mesh_verts, mesh_faces = load_mesh(cfg.mesh_path)
    
    # Choose mode based on --vis-cam
    if cfg.vis_cam is not None:
        print(f"\nMode: Single camera visualization (camera {cfg.vis_cam})")
        print(f"{'=' * 60}\n")
        generate_vis_cam_image(cfg, cameras, mesh_verts, mesh_faces)
    else:
        print(f"\nMode: Interactive 3D viewer")
        print(f"{'=' * 60}\n")
        run_interactive_viewer(cfg, cameras, mesh_verts, mesh_faces)


if __name__ == "__main__":
    main()
