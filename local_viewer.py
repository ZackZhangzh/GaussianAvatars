# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

import json
import math
import tyro
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple
from pathlib import Path
import time
import dearpygui.dearpygui as dpg
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import matplotlib

from utils.viewer_utils import Mini3DViewer, Mini3DViewerConfig
from gaussian_renderer import GaussianModel, FlameGaussianModel
from gaussian_renderer import render
from mesh_renderer import NVDiffRenderer


@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


@dataclass
class Config(Mini3DViewerConfig):
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    """Pipeline settings for gaussian splatting rendering"""
    cam_convention: Literal["opengl", "opencv"] = "opencv"
    """Camera convention"""
    point_path: Optional[Path] = None
    """Path to the gaussian splatting file"""
    motion_path: Optional[Path] = None
    """Path to the motion file (npz)"""
    sh_degree: int = 3
    """Spherical Harmonics degree"""
    background_color: tuple[float, float, float] = (1., 1., 1.)
    """default GUI background color"""
    save_folder: Path = Path("./viewer_output")
    """default saving folder"""
    fps: int = 25
    """default fps for recording"""
    keyframe_interval: int = 1
    """default keyframe interval"""
    ref_json: Optional[Path] = None
    """ Path to a reference json file. We copy file paths from a reference json into 
    the exported trajectory json file as placeholders so that `render.py` can directly
    load it like a normal sequence. """
    demo_mode: bool = False
    """The UI will be simplified in demo mode."""
    
    # === LBS功能参数 ===
    lbs: bool = False
    """Enable Linear Blend Skinning control"""
    segment_path: Optional[Path] = None
    """Directory containing segment mesh files (obj/stl)"""
    transform_path: Optional[Path] = None
    """Optional transform file (.npz) to apply to segment meshes"""
    skull_jaw: Tuple[int, int] = (4, 5)
    """Mesh IDs for (skull, jaw), e.g., Segment_4 and Segment_5"""
    lbs_weight_damping: float = 20.0
    """Weight damping factor for LBS (与LBS项目一致)"""
    lbs_proximity_threshold: float = 12.0
    """Proximity threshold for seed selection"""
    lbs_small_step_radius: float = 5.0
    """Radius for graph connectivity in geodesic computation"""

class LocalViewer(Mini3DViewer):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        
        # recording settings
        self.keyframes = []  # list of state dicts of keyframes
        self.all_frames = {}  # state dicts of all frames {key: [num_frames, ...]}
        self.num_record_timeline = 0
        self.playing = False

        print("Initializing 3D Gaussians...")
        self.init_gaussians()

        if self.gaussians.binding is not None:
            # rendering settings
            self.mesh_color = torch.tensor([1, 1, 1, 0.5])
            self.face_colors = None
            print("Initializing mesh renderer...")
            self.mesh_renderer = NVDiffRenderer(use_opengl=False)
        
        # FLAME parameters
        if self.gaussians.binding is not None:
            print("Initializing FLAME parameters...")
            self.reset_flame_param()
        
        # === Segment Mesh功能 (Level 2) ===
        self.segment_meshes = {}  # {id: {'vertices': np.array, 'faces': np.array}}
        self.segment_visible = {}  # {id: bool} - visibility state
        
        if cfg.segment_path is not None:
            print("[Segment] Loading segment meshes...")
            self._load_segments()
        
        # === LBS功能 (Level 3) ===
        self.lbs_controller = None
        self.show_lbs_weights = False  # whether to show weight visualization
        
        if cfg.lbs:
            if cfg.segment_path is None:
                print("[LBS] Error: --lbs requires --segment-path to be specified")
            elif len(self.segment_meshes) == 0:
                print("[LBS] Error: No segments loaded, cannot enable LBS")
            else:
                print("[LBS] Initializing LBS controller...")
                self._init_lbs_controller()
        
        super().__init__(cfg, 'GaussianAvatars - Local Viewer')

        if self.gaussians.binding is not None:
            self.num_timesteps = self.gaussians.num_timesteps
            dpg.configure_item("_slider_timestep", max_value=self.num_timesteps - 1)

            self.gaussians.select_mesh_by_timestep(self.timestep)

    def init_gaussians(self):
        # load gaussians
        if (Path(self.cfg.point_path).parent / "flame_param.npz").exists():
            self.gaussians = FlameGaussianModel(self.cfg.sh_degree)
        else:
            self.gaussians = GaussianModel(self.cfg.sh_degree)

        # selected_fid = self.gaussians.flame_model.mask.get_fid_by_region(['left_half'])
        # selected_fid = self.gaussians.flame_model.mask.get_fid_by_region(['right_half'])
        # unselected_fid = self.gaussians.flame_model.mask.get_fid_except_fids(selected_fid)
        unselected_fid = []
        
        if self.cfg.point_path is not None:
            if self.cfg.point_path.exists():
                self.gaussians.load_ply(self.cfg.point_path, has_target=False, motion_path=self.cfg.motion_path, disable_fid=unselected_fid)
            else:
                raise FileNotFoundError(f'{self.cfg.point_path} does not exist.')
    
    def _load_segments(self):
        """
        Level 2: 加载segment meshes并可视化 (不涉及LBS计算)
        仅需要 --segment-path 参数
        """
        # 确保mesh_renderer已经初始化
        if not hasattr(self, 'mesh_renderer'):
            print("[Segment] Error: mesh_renderer not initialized, cannot load segments")
            return
            
        segment_dir = Path(self.cfg.segment_path)
        if not segment_dir.exists():
            print(f"[Segment] Error: Segment directory not found: {segment_dir}")
            return
        
        # 1. 加载所有segment meshes (.obj 和 .stl)
        print(f"[Segment] Loading meshes from {segment_dir}")
        from utils.lbs import load_mesh_from_file
        
        segment_list = []  # 临时列表
        
        for pattern in ['Segment_*.obj', 'Segment_*.stl']:
            for mesh_file in sorted(segment_dir.glob(pattern)):
                try:
                    segment_id = int(mesh_file.stem.split('_')[1])
                    verts, faces = load_mesh_from_file(mesh_file)
                    
                    # 确保添加batch维度 [N, 3] -> [1, N, 3]
                    verts_batched = torch.from_numpy(verts).float().cuda().unsqueeze(0)
                    faces_batched = torch.from_numpy(faces).long().cuda()
                    
                    segment_data = {
                        'id': segment_id,
                        'name': mesh_file.stem,
                        'verts': verts_batched,
                        'verts_orig': verts_batched.clone(),
                        'faces': faces_batched,
                        'num_verts': verts.shape[0],
                        'num_faces': faces.shape[0],
                        'renderer': NVDiffRenderer(use_opengl=False),  # 独立渲染器
                    }
                    
                    segment_list.append((segment_id, segment_data))
                    self.segment_visible[segment_id] = True  # Default visible
                    
                    print(f"[Segment]   Loaded Segment_{segment_id}: {verts.shape[0]} vertices, {faces.shape[0]} faces")
                    print(f"[Segment]     Batched shape: verts={verts_batched.shape}, faces={faces_batched.shape}")
                    
                except Exception as e:
                    print(f"[Segment]   Failed to load {mesh_file.name}: {e}")
                    import traceback
                    traceback.print_exc()
        
        if len(segment_list) == 0:
            print("[Segment] Warning: No segment meshes found")
            return
        
        # 按ID排序并存入dict
        segment_list.sort(key=lambda x: x[0])
        for seg_id, seg_data in segment_list:
            self.segment_meshes[seg_id] = seg_data
        
        print(f"[Segment] Successfully loaded {len(self.segment_meshes)} segments: {list(self.segment_meshes.keys())}")
        
        # 2. 应用可选的变换矩阵
        if self.cfg.transform_path is not None:
            print(f"[Segment] Applying transform from {self.cfg.transform_path}")
            try:
                transform_data = np.load(self.cfg.transform_path)
                # 假设transform包含rotation和translation
                if 'rotation' in transform_data and 'translation' in transform_data:
                    rot = transform_data['rotation']
                    trans = transform_data['translation']
                    # Apply to all segments
                    for seg_id in self.segment_meshes:
                        verts = self.segment_meshes[seg_id]['verts'][0].cpu().numpy()  # Remove batch dim
                        # 简单的刚体变换
                        verts_transformed = verts @ rot.T + trans
                        self.segment_meshes[seg_id]['verts'] = torch.from_numpy(verts_transformed).float().cuda().unsqueeze(0)
                        self.segment_meshes[seg_id]['verts_orig'] = self.segment_meshes[seg_id]['verts'].clone()
                    print(f"[Segment]   Applied transform to all segments")
                else:
                    print(f"[Segment]   Warning: Transform file missing 'rotation' or 'translation' keys")
            except Exception as e:
                print(f"[Segment] Warning: Failed to apply transform: {e}")
    
    def _init_lbs_controller(self):
        """
        Level 3: 初始化LBS控制器并计算权重
        需要 --segment-path + --lbs + --skull-jaw 参数
        """
        from utils.lbs import LBSController, LBSConfig
        
        # 1. 验证skull和jaw存在
        skull_id, jaw_id = self.cfg.skull_jaw
        if skull_id not in self.segment_meshes or jaw_id not in self.segment_meshes:
            print(f"[LBS] Error: Skull ({skull_id}) or Jaw ({jaw_id}) not found in loaded segments")
            print(f"[LBS]   Available segments: {list(self.segment_meshes.keys())}")
            return
        
        skull_verts = self.segment_meshes[skull_id]['verts'][0].detach().cpu().numpy()  # [N, 3]
        jaw_verts = self.segment_meshes[jaw_id]['verts'][0].detach().cpu().numpy()  # [N, 3]
        print(f"[LBS] Using Segment_{skull_id} as skull ({skull_verts.shape[0]} verts), Segment_{jaw_id} as jaw ({jaw_verts.shape[0]} verts)")
        
        # 2. 获取FLAME skin vertices
        if self.gaussians.binding is None:
            print("[LBS] Warning: No FLAME mesh found, cannot initialize LBS")
            return
        
        skin_verts = self.gaussians.verts.detach().cpu().numpy()[0]  # [N, 3]
        print(f"[LBS] FLAME skin mesh: {len(skin_verts)} vertices")
        
        # 3. 初始化LBS控制器(包含权重计算)
        lbs_config = LBSConfig(
            weight_damping=self.cfg.lbs_weight_damping,
            proximity_threshold=self.cfg.lbs_proximity_threshold,
            small_step_radius=self.cfg.lbs_small_step_radius
        )
        
        try:
            self.lbs_controller = LBSController(
                skin_vertices=skin_verts,
                skull_vertices=skull_verts,
                jaw_vertices=jaw_verts,
                config=lbs_config
            )
            print("[LBS] LBS controller initialized successfully")
        except Exception as e:
            print(f"[LBS] Error: Failed to initialize LBS controller: {e}")
            import traceback
            traceback.print_exc()

    
    def _update_lbs_deformation(self):
        """应用LBS变形到skin和jaw meshes"""
        if not self.cfg.lbs or self.lbs_controller is None:
            return
        
        # 收集变换参数
        translation = np.array([
            dpg.get_value("_slider_lbs_trans_x"),
            dpg.get_value("_slider_lbs_trans_y"),
            dpg.get_value("_slider_lbs_trans_z")
        ])
        
        rotation = (
            np.radians(dpg.get_value("_slider_lbs_rot_pitch")),
            np.radians(dpg.get_value("_slider_lbs_rot_yaw"))
        )
        
        # 应用LBS变形
        jaw_verts, skull_verts, skin_verts = self.lbs_controller.apply_transformation(
            rotation=rotation,
            translation=translation
        )
        
        # 更新FLAME skin mesh
        self.gaussians.verts = torch.from_numpy(skin_verts).float().cuda().unsqueeze(0)  # Add batch dim [1, N, 3]
        
        # 更新segment meshes的顶点
        skull_id, jaw_id = self.cfg.skull_jaw
        self.segment_meshes[skull_id]['verts'] = torch.from_numpy(skull_verts).float().cuda().unsqueeze(0)
        self.segment_meshes[jaw_id]['verts'] = torch.from_numpy(jaw_verts).float().cuda().unsqueeze(0)
        
        self.need_update = True


    def refresh_stat(self):
        if self.last_time_fresh is not None:
            elapsed = time.time() - self.last_time_fresh
            fps = 1 / elapsed
            dpg.set_value("_log_fps", f'{int(fps):<4d}')
        self.last_time_fresh = time.time()
    
    def update_record_timeline(self):
        cycles = dpg.get_value("_input_cycles")
        if cycles == 0:
            self.num_record_timeline = sum([keyframe['interval'] for keyframe in self.keyframes[:-1]])
        else:
            self.num_record_timeline = sum([keyframe['interval'] for keyframe in self.keyframes]) * cycles

        dpg.configure_item("_slider_record_timestep", min_value=0, max_value=self.num_record_timeline-1)

        if len(self.keyframes) <= 0:
            self.all_frames = {}
            return
        else:
            k_x = []

            keyframes = self.keyframes.copy()
            if cycles > 0:
                # pad a cycle at the beginning and the end to ensure smooth transition
                keyframes = self.keyframes * (cycles + 2)
                t_couter = -sum([keyframe['interval'] for keyframe in self.keyframes])
            else:
                t_couter = 0

            for keyframe in keyframes:
                k_x.append(t_couter)
                t_couter += keyframe['interval']
            
            x = np.arange(self.num_record_timeline)
            self.all_frames = {}

            if len(keyframes) <= 1:
                for k in keyframes[0]:
                    k_y = np.concatenate([np.array(keyframe[k])[None] for keyframe in keyframes], axis=0)
                    self.all_frames[k] = np.tile(k_y, (self.num_record_timeline, 1))
            else:
                kind = 'linear' if len(keyframes) <= 3 else 'cubic'
            
                for k in keyframes[0]:
                    if k == 'interval':
                        continue
                    k_y = np.concatenate([np.array(keyframe[k])[None] for keyframe in keyframes], axis=0)
                  
                    interp_funcs = [interp1d(k_x, k_y[:, i], kind=kind, fill_value='extrapolate') for i in range(k_y.shape[1])]

                    y = np.array([interp_func(x) for interp_func in interp_funcs]).transpose(1, 0)
                    self.all_frames[k] = y

    def get_state_dict(self):
        return {
            'rot': self.cam.rot.as_quat(),
            'look_at': np.array(self.cam.look_at),
            'radius': np.array([self.cam.radius]).astype(np.float32),
            'fovy': np.array([self.cam.fovy]).astype(np.float32),
            'interval': self.cfg.fps*self.cfg.keyframe_interval,
        }

    def get_state_dict_record(self):
        record_timestep = dpg.get_value("_slider_record_timestep")
        state_dict = {k: self.all_frames[k][record_timestep] for k in self.all_frames}
        return state_dict

    def apply_state_dict(self, state_dict):
        if 'rot' in state_dict:
            self.cam.rot = R.from_quat(state_dict['rot'])
        if 'look_at' in state_dict:
            self.cam.look_at = state_dict['look_at']
        if 'radius' in state_dict:
            self.cam.radius = state_dict['radius'].item()
        if 'fovy' in state_dict:
            self.cam.fovy = state_dict['fovy'].item()
    
    def parse_ref_json(self):
        if self.cfg.ref_json is None:
            return {}
        else:
            with open(self.cfg.ref_json, 'r') as f:
                ref_dict = json.load(f)

        tid2paths = {}
        for frame in ref_dict['frames']:
            tid = frame['timestep_index']
            if tid not in tid2paths:
                tid2paths[tid] = frame
        return tid2paths
    
    def export_trajectory(self):
        tid2paths = self.parse_ref_json()

        if self.num_record_timeline <= 0:
            return
        
        timestamp = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        traj_dict = {'frames': []}
        timestep_indices = []
        camera_indices = []
        for i in range(self.num_record_timeline):
            # update
            dpg.set_value("_slider_record_timestep", i)
            state_dict = self.get_state_dict_record()
            self.apply_state_dict(state_dict)

            self.need_update = True
            while self.need_update:
                time.sleep(0.001)

            # save image
            save_folder = self.cfg.save_folder / timestamp
            if not save_folder.exists():
                save_folder.mkdir(parents=True)
            path = save_folder / f"{i:05d}.png"
            print(f"Saving image to {path}")
            Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)

            # cache camera parameters
            cx = self.cam.intrinsics[2]
            cy = self.cam.intrinsics[3]
            fl_x = self.cam.intrinsics[0].item() if isinstance(self.cam.intrinsics[0], np.ndarray) else self.cam.intrinsics[0]
            fl_y = self.cam.intrinsics[1].item() if isinstance(self.cam.intrinsics[1], np.ndarray) else self.cam.intrinsics[1]
            h = self.cam.image_height
            w = self.cam.image_width
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2

            c2w = self.cam.pose.copy()  # opencv convention
            c2w[:, [1, 2]] *= -1  # opencv to opengl
            # transform_matrix = np.linalg.inv(c2w).tolist()  # world2cam
            
            timestep_index = self.timestep
            camera_indx = i
            timestep_indices.append(timestep_index)
            camera_indices.append(camera_indx)
            
            tid2paths[timestep_index]['file_path']

            frame = {
                "cx": cx,
                "cy": cy,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "h": h,
                "w": w,
                "camera_angle_x": angle_x,
                "camera_angle_y": angle_y,
                "transform_matrix": c2w.tolist(),
                'timestep_index': timestep_index,
                'camera_indx': camera_indx,
            }
            if timestep_index in tid2paths:
                frame['file_path'] = tid2paths[timestep_index]['file_path']
                frame['fg_mask_path'] = tid2paths[timestep_index]['fg_mask_path']
                frame['flame_param_path'] = tid2paths[timestep_index]['flame_param_path']
            traj_dict['frames'].append(frame)

            # update timestep
            if dpg.get_value("_checkbox_dynamic_record"):
                self.timestep = min(self.timestep + 1, self.num_timesteps - 1)
                dpg.set_value("_slider_timestep", self.timestep)
                self.gaussians.select_mesh_by_timestep(self.timestep)
        
        traj_dict['timestep_indices'] = sorted(list(set(timestep_indices)))
        traj_dict['camera_indices'] = sorted(list(set(camera_indices)))
        
        # save camera parameters
        path = save_folder / f"trajectory.json"
        print(f"Saving trajectory to {path}")
        with open(path, 'w') as f:
            json.dump(traj_dict, f, indent=4)

    def reset_flame_param(self):
        self.flame_param = {
            'expr': torch.zeros(1, self.gaussians.n_expr),
            'rotation': torch.zeros(1, 3),
            'neck': torch.zeros(1, 3),
            'jaw': torch.zeros(1, 3),
            'eyes': torch.zeros(1, 6),
            'translation': torch.zeros(1, 3),
        }

    def define_gui(self):
        super().define_gui()

        # window: rendering options ==================================================================================================
        with dpg.window(label="Render", tag="_render_window", autosize=True):

            with dpg.group(horizontal=True):
                dpg.add_text("FPS:", show=not self.cfg.demo_mode)
                dpg.add_text("0   ", tag="_log_fps", show=not self.cfg.demo_mode)

            dpg.add_text(f"number of points: {self.gaussians._xyz.shape[0]}")
            
            with dpg.group(horizontal=True):
                # show splatting
                def callback_show_splatting(sender, app_data):
                    self.need_update = True
                dpg.add_checkbox(label="show splatting", default_value=True, callback=callback_show_splatting, tag="_checkbox_show_splatting")

                dpg.add_spacer(width=10)

                if self.gaussians.binding is not None:
                    # show mesh
                    def callback_show_mesh(sender, app_data):
                        self.need_update = True
                    dpg.add_checkbox(label="show mesh", default_value=False, callback=callback_show_mesh, tag="_checkbox_show_mesh")

                    # # show original mesh
                    # def callback_original_mesh(sender, app_data):
                    #     self.original_mesh = app_data
                    #     self.need_update = True
                    # dpg.add_checkbox(label="original mesh", default_value=self.original_mesh, callback=callback_original_mesh)
            
            # timestep slider and buttons
            if self.num_timesteps != None:
                def callback_set_current_frame(sender, app_data):
                    if sender == "_slider_timestep":
                        self.timestep = app_data
                    elif sender in ["_button_timestep_plus", "_mvKey_Right"]:
                        self.timestep = min(self.timestep + 1, self.num_timesteps - 1)
                    elif sender in ["_button_timestep_minus", "_mvKey_Left"]:
                        self.timestep = max(self.timestep - 1, 0)
                    elif sender == "_mvKey_Home":
                        self.timestep = 0
                    elif sender == "_mvKey_End":
                        self.timestep = self.num_timesteps - 1

                    dpg.set_value("_slider_timestep", self.timestep)
                    self.gaussians.select_mesh_by_timestep(self.timestep)

                    self.need_update = True
                with dpg.group(horizontal=True):
                    dpg.add_button(label='-', tag="_button_timestep_minus", callback=callback_set_current_frame)
                    dpg.add_button(label='+', tag="_button_timestep_plus", callback=callback_set_current_frame)
                    dpg.add_slider_int(label="timestep", tag='_slider_timestep', width=153, min_value=0, max_value=self.num_timesteps - 1, format="%d", default_value=0, callback=callback_set_current_frame)

            # # render_mode combo
            # def callback_change_mode(sender, app_data):
            #     self.render_mode = app_data
            #     self.need_update = True
            # dpg.add_combo(('rgb', 'depth', 'opacity'), label='render mode', default_value=self.render_mode, callback=callback_change_mode)

            # scaling_modifier slider
            def callback_set_scaling_modifier(sender, app_data):
                self.need_update = True
            dpg.add_slider_float(label="Scale modifier", min_value=0, max_value=1, format="%.2f", width=200, default_value=1, callback=callback_set_scaling_modifier, tag="_slider_scaling_modifier")

            # fov slider
            def callback_set_fovy(sender, app_data):
                self.cam.fovy = app_data
                self.need_update = True
            dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, width=200, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy, tag="_slider_fovy", show=not self.cfg.demo_mode)

            if self.gaussians.binding is not None:
                # visualization options
                def callback_visual_options(sender, app_data):
                    if app_data == 'number of points per face':
                        value, ct = self.gaussians.binding.unique(return_counts=True)
                        ct = torch.log10(ct + 1)
                        ct = ct.float() / ct.max()
                        cmap = matplotlib.colormaps["plasma"]
                        self.face_colors = torch.from_numpy(cmap(ct.cpu())[None, :, :3]).to(self.gaussians.verts)
                    else:
                        self.face_colors = self.mesh_color[:3].to(self.gaussians.verts)[None, None, :].repeat(1, self.gaussians.face_center.shape[0], 1)  # (1, F, 3)
                    
                    dpg.set_value('_checkbox_show_mesh', True)
                    self.need_update = True
                dpg.add_combo(["none", "number of points per face"], default_value="none", label='visualization', width=200, callback=callback_visual_options, tag="_visual_options")

                # mesh_color picker
                def callback_change_mesh_color(sender, app_data):
                    self.mesh_color = torch.tensor(app_data, dtype=torch.float32)  # only need RGB in [0, 1]
                    if dpg.get_value("_visual_options") == 'none':
                        self.face_colors = self.mesh_color[:3].to(self.gaussians.verts)[None, None, :].repeat(1, self.gaussians.face_center.shape[0], 1)
                    self.need_update = True
                dpg.add_color_edit((self.mesh_color*255).tolist(), label="Mesh Color", width=200, callback=callback_change_mesh_color, show=not self.cfg.demo_mode)

            # # bg_color picker
            # def callback_change_bg(sender, app_data):
            #     self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32)  # only need RGB in [0, 1]
            #     self.need_update = True
            # dpg.add_color_edit((self.bg_color*255).tolist(), label="Background Color", width=200, no_alpha=True, callback=callback_change_bg)

            # # near slider
            # def callback_set_near(sender, app_data):
            #     self.cam.znear = app_data
            #     self.need_update = True
            # dpg.add_slider_int(label="near", min_value=1e-8, max_value=2, format="%.2f", default_value=self.cam.znear, callback=callback_set_near, tag="_slider_near")

            # # far slider
            # def callback_set_far(sender, app_data):
            #     self.cam.zfar = app_data
            #     self.need_update = True
            # dpg.add_slider_int(label="far", min_value=1e-3, max_value=10, format="%.2f", default_value=self.cam.zfar, callback=callback_set_far, tag="_slider_far")
            
            # camera
            with dpg.group(horizontal=True):
                def callback_reset_camera(sender, app_data):
                    self.cam.reset()
                    self.need_update = True
                    dpg.set_value("_slider_fovy", self.cam.fovy)
                dpg.add_button(label="reset camera", tag="_button_reset_pose", callback=callback_reset_camera, show=not self.cfg.demo_mode)
                
                def callback_cache_camera(sender, app_data):
                    self.cam.save()
                dpg.add_button(label="cache camera", tag="_button_cache_pose", callback=callback_cache_camera, show=not self.cfg.demo_mode)

                def callback_clear_cache(sender, app_data):
                    self.cam.clear()
                dpg.add_button(label="clear cache", tag="_button_clear_cache", callback=callback_clear_cache, show=not self.cfg.demo_mode)
                
        # window: recording ==================================================================================================
        with dpg.window(label="Record", tag="_record_window", autosize=True, pos=(0, self.H//2)):
            dpg.add_text("Keyframes")
            with dpg.group(horizontal=True):
                # list keyframes
                def callback_set_current_keyframe(sender, app_data):
                    idx = int(dpg.get_value("_listbox_keyframes"))
                    self.apply_state_dict(self.keyframes[idx])

                    record_timestep = sum([keyframe['interval'] for keyframe in self.keyframes[:idx]])
                    dpg.set_value("_slider_record_timestep", record_timestep)

                    self.need_update = True
                dpg.add_listbox(self.keyframes, width=200, tag="_listbox_keyframes", callback=callback_set_current_keyframe)

                # edit keyframes
                with dpg.group():
                    # add
                    def callback_add_keyframe(sender, app_data):
                        if len(self.keyframes) == 0:
                            new_idx = 0
                        else:
                            new_idx = int(dpg.get_value("_listbox_keyframes")) + 1

                        states = self.get_state_dict()
                        
                        self.keyframes.insert(new_idx, states)
                        dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
                        dpg.set_value("_listbox_keyframes", new_idx)

                        self.update_record_timeline()
                    dpg.add_button(label="add", tag="_button_add_keyframe", callback=callback_add_keyframe)

                    # delete
                    def callback_delete_keyframe(sender, app_data):
                        idx = int(dpg.get_value("_listbox_keyframes"))
                        self.keyframes.pop(idx)
                        dpg.configure_item("_listbox_keyframes", items=list(range(len(self.keyframes))))
                        dpg.set_value("_listbox_keyframes", idx-1)

                        self.update_record_timeline()
                    dpg.add_button(label="delete", tag="_button_delete_keyframe", callback=callback_delete_keyframe)

                    # update
                    def callback_update_keyframe(sender, app_data):
                        if len(self.keyframes) == 0:
                            return
                        else:
                            idx = int(dpg.get_value("_listbox_keyframes"))

                        states = self.get_state_dict()
                        states['interval'] = self.cfg.fps*self.cfg.keyframe_interval

                        self.keyframes[idx] = states
                    dpg.add_button(label="update", tag="_button_update_keyframe", callback=callback_update_keyframe)

            with dpg.group(horizontal=True):
                def callback_set_record_cycles(sender, app_data):
                    self.update_record_timeline()
                dpg.add_input_int(label="cycles", tag="_input_cycles", default_value=0, width=70, callback=callback_set_record_cycles)

                def callback_set_keyframe_interval(sender, app_data):
                    self.cfg.keyframe_interval = app_data
                    for keyframe in self.keyframes:
                        keyframe['interval'] = self.cfg.fps*self.cfg.keyframe_interval
                    self.update_record_timeline()
                dpg.add_input_int(label="interval", tag="_input_interval", default_value=self.cfg.keyframe_interval, width=70, callback=callback_set_keyframe_interval)
            
            def callback_set_record_timestep(sender, app_data):
                state_dict = self.get_state_dict_record()
                
                self.apply_state_dict(state_dict)
                self.need_update = True
            dpg.add_slider_int(label="timeline", tag='_slider_record_timestep', width=200, min_value=0, max_value=0, format="%d", default_value=0, callback=callback_set_record_timestep)
            
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="dynamic", default_value=False, tag="_checkbox_dynamic_record")
                dpg.add_checkbox(label="loop", default_value=True, tag="_checkbox_loop_record")
            
            with dpg.group(horizontal=True):
                def callback_play(sender, app_data):
                    self.playing = not self.playing
                    self.need_update = True
                dpg.add_button(label="play", tag="_button_play", callback=callback_play)

                def callback_export_trajectory(sender, app_data):
                    self.export_trajectory()
                dpg.add_button(label="export traj", tag="_button_export_traj", callback=callback_export_trajectory)
            
            def callback_save_image(sender, app_data):
                if not self.cfg.save_folder.exists():
                    self.cfg.save_folder.mkdir(parents=True)
                path = self.cfg.save_folder / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{self.timestep}.png"
                print(f"Saving image to {path}")
                Image.fromarray((np.clip(self.render_buffer, 0, 1) * 255).astype(np.uint8)).save(path)
            with dpg.group(horizontal=True):
                dpg.add_button(label="save image", tag="_button_save_image", callback=callback_save_image)

        # window: FLAME ==================================================================================================
        if self.gaussians.binding is not None:
            with dpg.window(label="FLAME parameters", tag="_flame_window", autosize=True, pos=(self.W-300, 0)):
                def callback_enable_control(sender, app_data):
                    if app_data:
                        self.gaussians.update_mesh_by_param_dict(self.flame_param)
                    else:
                        self.gaussians.select_mesh_by_timestep(self.timestep)
                    self.need_update = True
                dpg.add_checkbox(label="enable control", default_value=False, tag="_checkbox_enable_control", callback=callback_enable_control)

                dpg.add_separator()

                def callback_set_pose(sender, app_data):
                    joint, axis = sender.split('-')[1:3]
                    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
                    self.flame_param[joint][0, axis_idx] = app_data
                    if joint == 'eyes':
                        self.flame_param[joint][0, 3+axis_idx] = app_data
                    if not dpg.get_value("_checkbox_enable_control"):
                        dpg.set_value("_checkbox_enable_control", True)
                    self.gaussians.update_mesh_by_param_dict(self.flame_param)
                    self.need_update = True
                dpg.add_text(f'Joints')
                self.pose_sliders = []
                max_rot = 0.5
                for joint in ['neck', 'jaw', 'eyes']:
                    if joint in self.flame_param:
                        with dpg.group(horizontal=True):
                            dpg.add_slider_float(min_value=-max_rot, max_value=max_rot, format="%.2f", default_value=self.flame_param[joint][0, 0], callback=callback_set_pose, tag=f"_slider-{joint}-x", width=70)
                            dpg.add_slider_float(min_value=-max_rot, max_value=max_rot, format="%.2f", default_value=self.flame_param[joint][0, 1], callback=callback_set_pose, tag=f"_slider-{joint}-y", width=70)
                            dpg.add_slider_float(min_value=-max_rot, max_value=max_rot, format="%.2f", default_value=self.flame_param[joint][0, 2], callback=callback_set_pose, tag=f"_slider-{joint}-z", width=70)
                            self.pose_sliders.append(f"_slider-{joint}-x")
                            self.pose_sliders.append(f"_slider-{joint}-y")
                            self.pose_sliders.append(f"_slider-{joint}-z")
                            dpg.add_text(f'{joint:4s}')
                dpg.add_text('   roll       pitch      yaw')
                
                dpg.add_separator()
                
                def callback_set_expr(sender, app_data):
                    expr_i = int(sender.split('-')[2])
                    self.flame_param['expr'][0, expr_i] = app_data
                    if not dpg.get_value("_checkbox_enable_control"):
                        dpg.set_value("_checkbox_enable_control", True)
                    self.gaussians.update_mesh_by_param_dict(self.flame_param)
                    self.need_update = True
                self.expr_sliders = []
                dpg.add_text(f'Expressions')
                for i in range(5):
                    dpg.add_slider_float(label=f"{i}", min_value=-3, max_value=3, format="%.2f", default_value=0, callback=callback_set_expr, tag=f"_slider-expr-{i}", width=250)
                    self.expr_sliders.append(f"_slider-expr-{i}")

                def callback_reset_flame(sender, app_data):
                    self.reset_flame_param()
                    if not dpg.get_value("_checkbox_enable_control"):
                        dpg.set_value("_checkbox_enable_control", True)
                    self.gaussians.update_mesh_by_param_dict(self.flame_param)
                    self.need_update = True
                    for slider in self.pose_sliders + self.expr_sliders:
                        dpg.set_value(slider, 0)
                dpg.add_button(label="reset FLAME", tag="_button_reset_flame", callback=callback_reset_flame)

        # window: Segment Control (Level 2) ==================================================================================================
        if len(self.segment_meshes) > 0:
            with dpg.window(label="Segment Meshes", tag="_segment_window", autosize=True, pos=(self.W-350, self.H//2)):
                
                dpg.add_text(f"Loaded {len(self.segment_meshes)} segments")
                dpg.add_separator()
                
                # Segment visibility controls
                for seg_id in sorted(self.segment_meshes.keys()):
                    skull_id, jaw_id = self.cfg.skull_jaw
                    label = f"Segment_{seg_id}"
                    if self.cfg.lbs:  # Only show labels if LBS is enabled
                        if seg_id == skull_id:
                            label += " (Skull)"
                        elif seg_id == jaw_id:
                            label += " (Jaw)"
                    
                    def callback_segment_visibility(sender, app_data, user_data):
                        seg_id = user_data
                        self.segment_visible[seg_id] = app_data
                        self.need_update = True
                    
                    dpg.add_checkbox(
                        label=label,
                        default_value=True,
                        callback=callback_segment_visibility,
                        user_data=seg_id,
                        tag=f"_checkbox_segment_{seg_id}"
                    )

        # window: LBS Control (Level 3) ==================================================================================================
        if self.cfg.lbs and self.lbs_controller is not None:
            with dpg.window(label="LBS Control", tag="_lbs_window", autosize=True, pos=(self.W-350, self.H//2 + 250)):
                
                # === 权重可视化 ===
                def callback_show_weights(sender, app_data):
                    self.show_lbs_weights = app_data
                    self.need_update = True
                
                dpg.add_checkbox(
                    label="Show LBS Weights",
                    default_value=False,
                    callback=callback_show_weights,
                    tag="_checkbox_show_lbs_weights"
                )
                
                dpg.add_separator()
                
                # === Jaw变换控制 ===
                with dpg.collapsing_header(label="Jaw Transform", default_open=True):
                    
                    # Translation
                    dpg.add_text("Translation (mm):")
                    
                    def callback_lbs_transform(sender, app_data):
                        self._update_lbs_deformation()
                    
                    dpg.add_slider_float(
                        label="X (Horizontal)",
                        min_value=-10, max_value=10,
                        default_value=0, format="%.1f",
                        callback=callback_lbs_transform,
                        tag="_slider_lbs_trans_x", width=200
                    )
                    dpg.add_slider_float(
                        label="Y (Depth)",
                        min_value=-10, max_value=10,
                        default_value=0, format="%.1f",
                        callback=callback_lbs_transform,
                        tag="_slider_lbs_trans_y", width=200
                    )
                    dpg.add_slider_float(
                        label="Z (Vertical)",
                        min_value=-10, max_value=10,
                        default_value=0, format="%.1f",
                        callback=callback_lbs_transform,
                        tag="_slider_lbs_trans_z", width=200
                    )
                    
                    dpg.add_separator()
                    
                    # Rotation
                    dpg.add_text("Rotation (degrees):")
                    dpg.add_slider_float(
                        label="Pitch (Open/Close)",
                        min_value=-24, max_value=0,
                        default_value=0, format="%.1f",
                        callback=callback_lbs_transform,
                        tag="_slider_lbs_rot_pitch", width=200
                    )
                    dpg.add_slider_float(
                        label="Yaw (Side-to-Side)",
                        min_value=-5, max_value=5,
                        default_value=0, format="%.1f",
                        callback=callback_lbs_transform,
                        tag="_slider_lbs_rot_yaw", width=200
                    )
                
                dpg.add_separator()
                
                # === 重置按钮 ===
                def callback_reset_lbs(sender, app_data):
                    dpg.set_value("_slider_lbs_trans_x", 0)
                    dpg.set_value("_slider_lbs_trans_y", 0)
                    dpg.set_value("_slider_lbs_trans_z", 0)
                    dpg.set_value("_slider_lbs_rot_pitch", 0)
                    dpg.set_value("_slider_lbs_rot_yaw", 0)
                    self._update_lbs_deformation()
                
                dpg.add_button(label="Reset Transform", callback=callback_reset_lbs, width=200)

        # widget-dependent handlers ========================================================================================
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=callback_set_current_frame, tag='_mvKey_Left')
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=callback_set_current_frame, tag='_mvKey_Right')
            dpg.add_key_press_handler(dpg.mvKey_Home, callback=callback_set_current_frame, tag='_mvKey_Home')
            dpg.add_key_press_handler(dpg.mvKey_End, callback=callback_set_current_frame, tag='_mvKey_End')

            def callbackmouse_wheel_slider(sender, app_data):
                delta = app_data
                if dpg.is_item_hovered("_slider_timestep"):
                    self.timestep = min(max(self.timestep - delta, 0), self.num_timesteps - 1)
                    dpg.set_value("_slider_timestep", self.timestep)
                    self.gaussians.select_mesh_by_timestep(self.timestep)
                    self.need_update = True
            dpg.add_mouse_wheel_handler(callback=callbackmouse_wheel_slider)

    def prepare_camera(self):
        @dataclass
        class Cam:
            FoVx = float(np.radians(self.cam.fovx))
            FoVy = float(np.radians(self.cam.fovy))
            image_height = self.cam.image_height
            image_width = self.cam.image_width
            world_view_transform = torch.tensor(self.cam.world_view_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            full_proj_transform = torch.tensor(self.cam.full_proj_transform).float().cuda().T  # the transpose is required by gaussian splatting rasterizer
            camera_center = torch.tensor(self.cam.pose[:3, 3]).cuda()
        return Cam

    @torch.no_grad()
    def run(self):
        print("Running LocalViewer...")

        while dpg.is_dearpygui_running():

            if self.need_update or self.playing:
                cam = self.prepare_camera()

                if dpg.get_value("_checkbox_show_splatting"):
                    # rgb
                    rgb_splatting = render(cam, self.gaussians, self.cfg.pipeline, torch.tensor(self.cfg.background_color).cuda(), scaling_modifier=dpg.get_value("_slider_scaling_modifier"))["render"].permute(1, 2, 0).contiguous()

                    # opacity
                    # override_color = torch.ones_like(self.gaussians._xyz).cuda()
                    # background_color = torch.tensor(self.cfg.background_color).cuda() * 0
                    # rgb_splatting = render(cam, self.gaussians, self.cfg.pipeline, background_color, scaling_modifier=dpg.get_value("_slider_scaling_modifier"), override_color=override_color)["render"].permute(1, 2, 0).contiguous()

                if self.gaussians.binding is not None and dpg.get_value("_checkbox_show_mesh"):
                    # Handle weight visualization for FLAME mesh
                    if self.cfg.lbs and self.show_lbs_weights and self.lbs_controller is not None:
                        # Show LBS weights as heat map
                        weights_jaw = self.lbs_controller.weights[:, 0]  # Jaw weights
                        weights_jaw_tensor = torch.from_numpy(weights_jaw).float().cuda()
                        
                        # Use colormap for visualization
                        cmap = matplotlib.colormaps["jet"]
                        weights_colors = torch.from_numpy(cmap(weights_jaw)[:, :3]).float().cuda()
                        face_colors_weights = weights_colors[self.gaussians.faces].mean(dim=1, keepdim=True)  # (F, 1, 3)
                        
                        out_dict = self.mesh_renderer.render_from_camera(self.gaussians.verts, self.gaussians.faces, cam, face_colors=face_colors_weights)
                    else:
                        # Normal FLAME mesh rendering
                        out_dict = self.mesh_renderer.render_from_camera(self.gaussians.verts, self.gaussians.faces, cam, face_colors=self.face_colors)

                    rgba_mesh = out_dict['rgba'].squeeze(0)  # (H, W, C)
                    rgb_mesh = rgba_mesh[:, :, :3]
                    alpha_mesh = rgba_mesh[:, :, 3:]
                    mesh_opacity = self.mesh_color[3:].cuda()
                
                # Render segment meshes (Level 2+)
                rgb_segments = None
                alpha_segments = None
                if len(self.segment_meshes) > 0:  # Render segments if loaded, regardless of LBS
                    for seg_id in sorted(self.segment_meshes.keys()):
                        if not self.segment_visible.get(seg_id, True):
                            continue
                        
                        seg_data = self.segment_meshes[seg_id]
                        seg_verts = seg_data['verts']  # Already [1, N, 3]
                        seg_faces = seg_data['faces']   # Already [F, 3]
                        seg_renderer = seg_data['renderer']  # Independent renderer
                        
                        # Color assignment
                        skull_id, jaw_id = self.cfg.skull_jaw
                        if seg_id == skull_id:
                            seg_color = torch.tensor([1.0, 0.0, 0.0, 0.5]).cuda()  # Red for skull
                        elif seg_id == jaw_id:
                            seg_color = torch.tensor([0.0, 0.0, 1.0, 0.5]).cuda()  # Blue for jaw
                        else:
                            seg_color = torch.tensor([0.5, 0.5, 0.5, 0.5]).cuda()  # Gray for others
                        
                        # Create face colors: [1, num_faces, 3]
                        seg_face_colors = seg_color[:3].view(1, 1, 3).expand(1, seg_faces.shape[0], 3)
                        
                        try:
                            out_seg = seg_renderer.render_from_camera(seg_verts, seg_faces, cam, face_colors=seg_face_colors)
                            rgba_seg = out_seg['rgba'].squeeze(0)  # [H, W, 4]
                            
                            # Apply alpha from color config
                            rgb_seg = rgba_seg[:, :, :3]
                            alpha_seg = rgba_seg[:, :, 3:] * seg_color[3]  # Apply transparency
                            
                            if rgb_segments is None:
                                rgb_segments = rgb_seg * alpha_seg
                                alpha_segments = alpha_seg
                            else:
                                # Alpha compositing: front-to-back
                                rgb_segments = rgb_segments * (1 - alpha_seg) + rgb_seg * alpha_seg
                                alpha_segments = alpha_segments + alpha_seg * (1 - alpha_segments)
                        except Exception as e:
                            print(f"[Segment] Warning: Failed to render segment {seg_id}: {e}")
                            import traceback
                            traceback.print_exc()

                # Final compositing
                if dpg.get_value("_checkbox_show_splatting") and dpg.get_value("_checkbox_show_mesh"):
                    rgb = rgb_mesh * alpha_mesh * mesh_opacity  + rgb_splatting * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
                elif dpg.get_value("_checkbox_show_splatting") and not dpg.get_value("_checkbox_show_mesh"):
                    rgb = rgb_splatting
                elif not dpg.get_value("_checkbox_show_splatting") and dpg.get_value("_checkbox_show_mesh"):
                    rgb = rgb_mesh
                else:
                    rgb = torch.ones([self.H, self.W, 3]).cuda()
                
                # Composite segment meshes on top
                if rgb_segments is not None and alpha_segments is not None:
                    rgb = rgb * (1 - alpha_segments) + rgb_segments

                self.render_buffer = rgb.cpu().numpy()
                if self.render_buffer.shape[0] != self.H or self.render_buffer.shape[1] != self.W:
                    continue
                dpg.set_value("_texture", self.render_buffer)

                self.refresh_stat()
                self.need_update = False

                if self.playing:
                    record_timestep = dpg.get_value("_slider_record_timestep")
                    if record_timestep >= self.num_record_timeline - 1:
                        if not dpg.get_value("_checkbox_loop_record"):
                            self.playing = False
                        dpg.set_value("_slider_record_timestep", 0)
                    else:
                        dpg.set_value("_slider_record_timestep", record_timestep + 1)
                        if dpg.get_value("_checkbox_dynamic_record"):
                            self.timestep = min(self.timestep + 1, self.num_timesteps - 1)
                            dpg.set_value("_slider_timestep", self.timestep)
                            self.gaussians.select_mesh_by_timestep(self.timestep)

                        state_dict = self.get_state_dict_record()
                        self.apply_state_dict(state_dict)

            dpg.render_dearpygui_frame()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    gui = LocalViewer(cfg)
    gui.run()
