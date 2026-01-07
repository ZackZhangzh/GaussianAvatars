"""
Mesh Alignment to FLAME Canonical Template

This diagnostic tool aligns a custom mesh to the canonical FLAME template using 
rigid Procrustes analysis based on 3D landmarks. It provides visual verification 
outputs to validate the alignment before using the mesh in the full tracking pipeline.

Usage:
    python mesh_align_to_flame.py \
        --mesh-path ../data/MRI/my_scan.obj \
        --lmk-path ../data/MRI/my_landmarks.npy \
        --output-dir ./output/alignment

Author: Generated for VHAP pipeline
Date: 2025-11-13
"""

import sys
import json
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime
from time import time

import numpy as np
import torch
import trimesh
import tyro
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation as R

# Add VHAP to path to import FlameHead
sys.path.insert(0, str(Path(__file__).parent.parent / "VHAP"))
from vhap.model.flame import FlameHead
from vhap.model.lbs import vertices2landmarks

# Add GaussianAvatars to path for dataset loading
sys.path.insert(0, str(Path(__file__).parent.parent / "GaussianAvatars"))
from scene.dataset_readers import readMeshesFromTransforms

# Additional imports for manual fine-tuning
try:
    import dearpygui.dearpygui as dpg
    from utils.viewer_utils import Mini3DViewer, Mini3DViewerConfig
    from mesh_renderer import NVDiffRenderer
except ImportError:
    pass  # Will handle gracefully if missing during execution check



# Landmark format constants
LANDMARK_FORMAT_68_TO_51_INDICES = np.arange(17, 68, dtype=int)  # Remove face contour [0:17]
LANDMARK_FORMAT_70_TO_53_INDICES = np.arange(17, 70, dtype=int)  # Extract static subset


@dataclass
class AlignmentConfig:
    """Configuration for mesh alignment tool.
    
    Aligns source mesh to target using rigid/similarity transformation based on landmarks.
    Supports two target modes:
    - 'flame': Align to FLAME reference (dataset or .obj)
    - 'custom': Align to custom mesh with landmarks
    """
    
    # ========== SOURCE (mesh to be aligned) ==========
    source_mesh: Path
    """Path to source mesh (.obj file) to be aligned"""
    
    source_lmk: Path
    """Path to source landmarks (.npy or .pp file)"""
    
    # ========== TARGET (reference to align to) ==========
    target_mode: str = "flame"
    """Target mode: 'flame' (FLAME reference) or 'custom' (custom mesh)"""
    
    # FLAME target mode parameters
    target_flame_ref: Optional[Path] = None
    """[FLAME mode] Path to FLAME reference: dataset folder (with transforms_train.json) OR .obj file"""
    
    # Custom target mode parameters
    target_mesh: Optional[Path] = None
    """[Custom mode] Path to target mesh (.obj file)"""
    
    target_lmk: Optional[Path] = None
    """[Custom mode] Path to target landmarks (.npy or .pp file)"""
    
    # ========== OUTPUT ==========
    output_dir: Path = Path("./output/alignment")
    """Output directory for visualizations and aligned mesh"""
    
    # ========== FLAME MODEL CONFIG (for flame mode) ==========
    landmark_type: str = "full"
    """Landmark type: 'static' (53 points) or 'full' (70 points with eyes)"""
    
    n_shape: int = 300
    """Number of FLAME shape components"""
    
    n_expr: int = 100
    """Number of FLAME expression components"""
    
    add_teeth: bool = True
    """Whether FLAME model includes teeth"""
    
    # ========== TRANSFORMATION OPTIONS ==========
    enable_scaling: bool = True
    """Whether to compute and apply scaling (similarity transform). If False, uses rigid transform (rotation + translation only)."""
    
    manual_scale: Optional[float] = None
    """Manual scale factor override. If provided, uses this instead of auto-computed scale. Requires enable_scaling=True."""
    
    # ========== VISUALIZATION OPTIONS ==========
    lmk_sphere_radius: float = 0.005
    """Radius of landmark spheres in visualization"""
    
    image_resolution: int = 1024
    """Resolution of output images"""
    
    save_visualizations: bool = True
    """Whether to generate PNG visualizations (requires pyglet)"""

    manual_fine_tune: bool = False
    """Whether to enable manual fine-tuning GUI after initial alignment"""

    load_alignment_path: Optional[Path] = None
    """Optional path to an existing alignment_transform.npz to resume fine-tuning from"""
    
    # ========== DEPRECATED (backward compatibility) ==========
    mesh_path: Optional[Path] = None
    """DEPRECATED: Use source_mesh instead"""
    
    lmk_path: Optional[Path] = None
    """DEPRECATED: Use source_lmk instead"""
    
    flame_reference: Optional[Path] = None
    """DEPRECATED: Use target_flame_ref instead"""
    
    dataset_path: Optional[Path] = None
    """DEPRECATED: Use target_flame_ref instead"""
    
    def __post_init__(self):
        """Validate configuration and handle deprecated parameters."""
        import warnings
        
        # Handle deprecated parameters
        if self.mesh_path is not None:
            warnings.warn("--mesh-path is deprecated, use --source-mesh instead", DeprecationWarning)
            if self.source_mesh == Path(""):  # Check if source_mesh was not set
                self.source_mesh = self.mesh_path
        
        if self.lmk_path is not None:
            warnings.warn("--lmk-path is deprecated, use --source-lmk instead", DeprecationWarning)
            if self.source_lmk == Path(""):
                self.source_lmk = self.lmk_path
        
        if self.flame_reference is not None:
            warnings.warn("--flame-reference is deprecated, use --target-flame-ref instead", DeprecationWarning)
            if self.target_flame_ref is None:
                self.target_flame_ref = self.flame_reference
        
        if self.dataset_path is not None:
            warnings.warn("--dataset-path is deprecated, use --target-flame-ref instead", DeprecationWarning)
            if self.target_flame_ref is None:
                self.target_flame_ref = self.dataset_path
        
        # Validate target mode
        if self.target_mode not in ["flame", "custom"]:
            raise ValueError(
                f"Invalid target_mode: '{self.target_mode}'\n"
                f"  Must be 'flame' or 'custom'\n"
                f"  Use 'flame' to align to FLAME reference\n"
                f"  Use 'custom' to align to custom mesh"
            )
        
        # Validate target mode parameters
        if self.target_mode == "flame":
            if self.target_flame_ref is None:
                raise ValueError(
                    "FLAME target mode requires --target-flame-ref\n"
                    "  Provide either:\n"
                    "    - Path to GaussianAvatars dataset folder (with transforms_train.json)\n"
                    "    - Path to FLAME .obj file"
                )
        elif self.target_mode == "custom":
            if self.target_mesh is None or self.target_lmk is None:
                raise ValueError(
                    "Custom target mode requires both --target-mesh and --target-lmk\n"
                    f"  target_mesh: {self.target_mesh}\n"
                    f"  target_lmk: {self.target_lmk}"
                )
        
        # Validate source inputs
        if not self.source_mesh.exists():
            raise FileNotFoundError(f"Source mesh not found: {self.source_mesh}")
        if not self.source_lmk.exists():
            raise FileNotFoundError(f"Source landmarks not found: {self.source_lmk}")



@dataclass
class AlignmentResult:
    """Complete alignment results for comprehensive output."""
    
    # Input data
    target_mesh_vertices: np.ndarray
    target_mesh_faces: np.ndarray
    target_landmarks: np.ndarray
    source_mesh_vertices: np.ndarray
    source_mesh_faces: np.ndarray
    source_landmarks_original: np.ndarray
    
    # Transformation components
    transform_matrix: np.ndarray  # 4×4
    scale_factor: float
    rotation_matrix: np.ndarray   # 3×3
    translation_vector: np.ndarray  # 3
    
    # Transformed outputs
    source_mesh_aligned_vertices: np.ndarray
    source_landmarks_aligned: np.ndarray
    
    # Quality metrics
    mean_lmk_error: float
    max_lmk_error: float
    std_lmk_error: float
    per_landmark_errors: np.ndarray
    
    # Target mesh metadata (optional, for FLAME mode)
    target_mesh_with_offset_vertices: Optional[np.ndarray] = None
    
    # Timing info
    execution_time: float = 0.0
    timing_breakdown: dict = field(default_factory=dict)


def load_picked_points(filepath: Path) -> np.ndarray:
    """
    Load picked points from MeshLab .pp file format.
    
    Args:
        filepath: Path to .pp file
        
    Returns:
        Array of shape (N, 3) containing 3D points
    """
    def get_num(string):
        """Extract number from string like 'x="123.456"'"""
        pos1 = string.find('"')
        pos2 = string.find('"', pos1 + 1)
        return float(string[pos1 + 1:pos2])
    
    def get_point(str_array):
        """Extract 3D point from array of strings containing x=, y=, z="""
        if 'x=' in str_array[0] and 'y=' in str_array[1] and 'z=' in str_array[2]:
            return [get_num(str_array[0]), get_num(str_array[1]), get_num(str_array[2])]
        else:
            return []
    
    picked_points = []
    with open(filepath, 'r') as f:
        for line in f:
            if 'point' not in line:
                continue
            
            str_parts = line.split()
            if len(str_parts) < 4:
                continue
            
            # Find indices of x=, y=, z= in the split string
            ix = [i for i, s in enumerate(str_parts) if 'x=' in s]
            iy = [i for i, s in enumerate(str_parts) if 'y=' in s]
            iz = [i for i, s in enumerate(str_parts) if 'z=' in s]
            
            if ix and iy and iz:
                point = get_point([str_parts[ix[0]], str_parts[iy[0]], str_parts[iz[0]]])
                if point:
                    picked_points.append(point)
    
    return np.array(picked_points)


def load_user_landmarks(lmk_path: Path, expected_count: Optional[int] = None) -> np.ndarray:
    """
    Load user-provided landmarks from file.
    
    Args:
        lmk_path: Path to landmarks file (.npy or .pp)
        expected_count: Expected number of landmarks (53 or 70), None to skip validation
        
    Returns:
        landmarks: Array of shape (N, 3)
    """
    lmk_path = Path(lmk_path)
    
    if not lmk_path.exists():
        raise FileNotFoundError(f"Landmark file not found: {lmk_path}")
    
    if lmk_path.suffix == '.npy':
        landmarks = np.load(lmk_path)
    elif lmk_path.suffix == '.pp':
        # Load MeshLab picked points file
        landmarks = load_picked_points(lmk_path)
    else:
        raise ValueError(f"Unsupported landmark file format: {lmk_path.suffix}")
    
    # Validate shape
    if landmarks.ndim != 2 or landmarks.shape[1] != 3:
        raise ValueError(f"Expected landmarks shape (N, 3), got {landmarks.shape}")
    
    actual_count = landmarks.shape[0]
    print(f"Loaded {actual_count} landmarks from {lmk_path}")
    
    # Validate expected count if specified
    if expected_count is not None and actual_count != expected_count:
        raise ValueError(
            f"Landmark count mismatch: expected {expected_count}, got {actual_count}\n"
            f"  File: {lmk_path}\n"
            f"  Hint: Check --landmark-type parameter (static=53, full=70)"
        )
    
    return landmarks


def load_dataset_flame_params(dataset_path: Path) -> Optional[dict]:
    """
    Load FLAME parameters from GaussianAvatars dataset.
    
    This loads the first timestep's FLAME parameters from the dataset's
    transforms_train.json, which includes static_offset, shape, etc.
    
    Args:
        dataset_path: Path to GaussianAvatars dataset directory
                     (contains transforms_train.json)
    
    Returns:
        Dictionary with FLAME parameters, or None if loading fails.
        Keys: 'static_offset', 'shape', 'expr', 'rotation', etc.
    """
    if dataset_path is None:
        return None
    
    dataset_path = Path(dataset_path)
    
    # Check for transforms_train.json
    transforms_file = dataset_path / "transforms_train.json"
    if not transforms_file.exists():
        print(f"  ⚠️  transforms_train.json not found in {dataset_path}")
        print(f"     Skipping static_offset loading")
        return None
    
    try:
        # Use GaussianAvatars' readMeshesFromTransforms function
        print(f"  📂 Loading FLAME parameters from {dataset_path}")
        mesh_infos = readMeshesFromTransforms(
            path=str(dataset_path),
            transformsfile="transforms_train.json"
        )
        
        if not mesh_infos:
            print(f"  ⚠️  No mesh parameters found in dataset")
            return None
        
        # Get first timestep's parameters
        first_timestep = min(mesh_infos.keys())
        flame_params = mesh_infos[first_timestep]
        
        print(f"  ✓ Loaded FLAME params from timestep {first_timestep}")
        print(f"     Available keys: {list(flame_params.keys())}")
        
        # Verify static_offset exists
        if 'static_offset' in flame_params:
            offset_shape = flame_params['static_offset'].shape
            print(f"     static_offset shape: {offset_shape}")
        else:
            print(f"  ⚠️  No 'static_offset' found in FLAME parameters")
        
        return flame_params
        
    except Exception as e:
        print(f"  ✗ Failed to load dataset FLAME parameters: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_flame_mesh_from_params(
    flame_model: FlameHead,
    flame_params: dict,
    use_pose: bool = True,
    timestep: int = 0
) -> np.ndarray:
    """
    Generate FLAME mesh vertices from parameters.
    
    Two modes:
    1. Canonical (use_pose=False): v_template + shape + static_offset
    2. Posed (use_pose=True): Apply full FLAME forward with all pose params
    
    Args:
        flame_model: FlameHead instance
        flame_params: Dict with 'shape', 'static_offset', pose params, etc.
        use_pose: If True, applies pose/expression/jaw/etc. (matches local_viewer)
        timestep: Which timestep's pose to use (default 0)
    
    Returns:
        vertices: [V, 3] numpy array
    """
    import torch
    from vhap.model.lbs import blend_shapes
    
    device = flame_model.v_template.device
    
    if not use_pose:
        # OLD CANONICAL MODE (for backward compatibility)
        # Start with template
        vertices = flame_model.v_template.clone()
        
        # Apply shape blendshapes
        if 'shape' in flame_params:
            shape_tensor = torch.from_numpy(flame_params['shape']).float().to(device)
            
            # Ensure shape_tensor is 1D: handle both (N,) and (1, N) formats
            if shape_tensor.dim() > 1:
                shape_tensor = shape_tensor.squeeze()
            
            # Handle dimension mismatch between dataset shape and FLAME shapedirs
            n_shape_available = flame_model.shapedirs.shape[2]
            n_shape_actual = shape_tensor.shape[0]
            
            if n_shape_actual > n_shape_available:
                # Dataset has more dims than FLAME model -> truncate
                print(f"  ⚠️  Truncating shape from {n_shape_actual} to {n_shape_available} dimensions")
                shape_tensor = shape_tensor[:n_shape_available]
            elif n_shape_actual < n_shape_available:
                # Dataset has fewer dims than FLAME model -> pad with zeros
                print(f"  ⚠️  Padding shape from {n_shape_actual} to {n_shape_available} dimensions (zeros)")
                padding = torch.zeros(n_shape_available - n_shape_actual, dtype=shape_tensor.dtype, device=device)
                shape_tensor = torch.cat([shape_tensor, padding], dim=0)
            
            shape_offset = blend_shapes(
                shape_tensor.unsqueeze(0),
                flame_model.shapedirs
            )[0]
            vertices = vertices + shape_offset
            print(f"  ✓ Applied shape blendshapes ({shape_tensor.shape[0]} dims)")
        
        # Apply static_offset
        if 'static_offset' in flame_params:
            static_offset = torch.from_numpy(flame_params['static_offset']).float().to(device)
            
            # Handle batch dimension: (1, V, 3) -> (V, 3)
            if static_offset.dim() == 3 and static_offset.shape[0] == 1:
                static_offset = static_offset.squeeze(0)
            
            vertices = vertices + static_offset
            print(f"  ✓ Applied static_offset")
        
        return vertices.cpu().numpy()
    
    else:
        # NEW POSED MODE (matches local_viewer.py)
        print(f"  🎭 Generating POSED FLAME mesh (timestep {timestep})")
        
        # Prepare all parameters (matching scene/flame_gaussian_model.py:L121-134)
        shape = torch.from_numpy(flame_params['shape']).float().to(device)
        
        # Ensure shape is 1D
        if shape.dim() > 1:
            shape = shape.squeeze()
        
        # Handle pose parameters - need to extract timestep
        def get_param(key, default_shape):
            if key in flame_params:
                param = torch.from_numpy(flame_params[key]).float().to(device)
                # Handle both (T, N) and (N,) formats
                if param.dim() == 1:
                    param = param.unsqueeze(0)  # (N,) -> (1, N)
                # Extract timestep if multi-timestep
                if param.shape[0] > timestep:
                    return param[timestep:timestep+1]
                else:
                    return param[0:1]
            else:
                return torch.zeros(1, default_shape, dtype=torch.float32, device=device)
        
        expr = get_param('expr', 100)  # Default FLAME n_expr
        rotation = get_param('rotation', 3)
        neck_pose = get_param('neck_pose', 3)
        jaw_pose = get_param('jaw_pose', 3)
        eyes_pose = get_param('eyes_pose', 6)
        translation = get_param('translation', 3)
        
        # Get static_offset (no timestep dimension)
        if 'static_offset' in flame_params:
            static_offset = torch.from_numpy(flame_params['static_offset']).float().to(device)
            if static_offset.dim() == 3:
                static_offset = static_offset.squeeze(0)  # (1, V, 3) -> (V, 3)
        else:
            static_offset = None
        
        # Get dynamic_offset for this timestep
        if 'dynamic_offset' in flame_params:
            dyn_offset = torch.from_numpy(flame_params['dynamic_offset']).float().to(device)
            if dyn_offset.dim() == 3 and dyn_offset.shape[0] > timestep:
                dynamic_offset = dyn_offset[timestep:timestep+1]
            else:
                dynamic_offset = torch.zeros(1, flame_model.v_template.shape[0], 3, device=device)
        else:
            dynamic_offset = torch.zeros(1, flame_model.v_template.shape[0], 3, device=device)
        
        print(f"     shape={shape.shape}, expr={expr.shape}, rotation={rotation.shape}")
        print(f"     jaw={jaw_pose.shape}, neck={neck_pose.shape}, eyes={eyes_pose.shape}")
        print(f"     translation={translation[0].cpu().numpy()}")
        
        # Call FLAME forward (matching local_viewer.py exactly)
        with torch.no_grad():
            vertices = flame_model(
                shape[None, ...],  # (1, n_shape)
                expr,  # (1, n_expr)
                rotation,  # (1, 3)
                neck_pose,  # (1, 3)
                jaw_pose,  # (1, 3)
                eyes_pose,  # (1, 6)
                translation,  # (1, 3)
                zero_centered_at_root_node=False,  # ← CRITICAL: match local_viewer
                return_landmarks=False,
                return_verts_cano=False,
                static_offset=static_offset,
                dynamic_offset=dynamic_offset
            )
        
        print(f"  ✓ Generated POSED FLAME mesh (with expression, pose, translation)")
        return vertices[0].cpu().numpy()


def load_flame_reference_with_landmarks(
    flame_reference: Path,
    flame_model: FlameHead,
    landmark_type: str = "full"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load FLAME reference mesh and compute landmarks using topology.
    
    Supports two input types:
    1. Dataset folder (with transforms_train.json) - generates mesh from parameters
    2. OBJ file - loads directly
    
    In both cases, landmarks are computed using vertices2landmarks with FLAME topology.
    
    Args:
        flame_reference: Path to dataset folder or .obj file
        flame_model: FlameHead instance (for topology and mesh generation)
        landmark_type: "full" (70 points) or "static" (53 points)
    
    Returns:
        (vertices, landmarks): Both as [N, 3] numpy arrays
    """
    import torch
    import trimesh
    
    flame_reference = Path(flame_reference)
    
    # Determine flame_vertices based on input type
    if flame_reference.is_dir():
        # Method 1: Generate from dataset
        print(f"📂 Loading FLAME reference from dataset: {flame_reference}")
        
        # Check for transforms_train.json
        if not (flame_reference / "transforms_train.json").exists():
            raise ValueError(
                f"Not a valid GaussianAvatars dataset folder: {flame_reference}\n"
                f"Expected file: {flame_reference / 'transforms_train.json'}"
            )
        
        # Load FLAME parameters
        flame_params = load_dataset_flame_params(flame_reference)
        if not flame_params:
            raise ValueError(f"Failed to load FLAME parameters from {flame_reference}")
        
        # Generate mesh from parameters (POSED mode to match local_viewer.py)
        flame_vertices = generate_flame_mesh_from_params(
            flame_model, 
            flame_params,
            use_pose=True,  # ← Use POSED FLAME (with expression, jaw, etc.)
            timestep=0  # ← Same as local_viewer initialization
        )
        
    elif flame_reference.suffix.lower() == '.obj':
        # Method 2: Load from OBJ file
        print(f"📄 Loading FLAME reference from OBJ: {flame_reference}")
        
        mesh = trimesh.load(flame_reference, process=False)
        flame_vertices = mesh.vertices
        
        # Verify topology compatibility
        expected_verts = flame_model.v_template.shape[0]
        if len(flame_vertices) != expected_verts:
            print(f"  ⚠️  Warning: Vertex count mismatch!")
            print(f"     OBJ has {len(flame_vertices)} vertices")
            print(f"     FLAME template has {expected_verts} vertices")
            print(f"     Landmarks may be incorrect if topology differs!")
        
    else:
        raise ValueError(
            f"flame_reference must be a dataset folder or .obj file.\n"
            f"Got: {flame_reference}\n"
            f"  Is directory: {flame_reference.is_dir()}\n"
            f"  Suffix: {flame_reference.suffix}"
        )
    
    # Compute landmarks using FLAME topology (same for both methods)
    print(f"  🎯 Computing FLAME landmarks using topology ({landmark_type})")
    
    verts_tensor = torch.from_numpy(flame_vertices).float().unsqueeze(0)
    
    # FlameHead only has full_lmk_* attributes (70 landmarks)
    # For static mode, we compute all 70 then extract subset
    lmk_faces_idx = flame_model.full_lmk_faces_idx
    lmk_bary_coords = flame_model.full_lmk_bary_coords
    
    flame_landmarks_full = vertices2landmarks(
        verts_tensor,
        flame_model.faces,
        lmk_faces_idx,
        lmk_bary_coords
    )
    
    flame_landmarks_full = flame_landmarks_full[0].cpu().numpy()
    
    # Extract static subset if needed (remove first 17 contour points)
    if landmark_type == "static":
        flame_landmarks = flame_landmarks_full[17:]  # 70 → 53 (remove face contour)
        print(f"  ✓ Computed {len(flame_landmarks)} FLAME landmarks (static subset from 70)")
    else:
        flame_landmarks = flame_landmarks_full
        print(f"  ✓ Computed {len(flame_landmarks)} FLAME landmarks (full)")
    
    return flame_vertices, flame_landmarks


def load_flame_model_and_landmarks(
    n_shape: int = 300,
    n_expr: int = 100,
    add_teeth: bool = True,
    landmark_type: str = "full",
    dataset_flame_params: Optional[dict] = None
) -> Tuple[FlameHead, np.ndarray, Optional[np.ndarray]]:
    """
    Load FLAME model and compute canonical landmark positions.
    
    This replicates the exact initialization used in track_nersemble_v2.py
    and export_as_nerf_dataset.py to ensure consistency with VHAP pipeline.
    
    Uses the exact same vertices2landmarks() function from vhap.model.lbs
    to guarantee identical landmark computation.
    
    Args:
        n_shape: Number of shape components
        n_expr: Number of expression components
        add_teeth: Whether to include teeth in the model
        landmark_type: 'static' (53 points) or 'full' (70 points with eyes)
        dataset_flame_params: Optional FLAME parameters from dataset (includes static_offset)
        
    Returns:
        flame_model: Initialized FlameHead model
        flame_landmarks: Array of shape (N, 3) with canonical landmark positions
        vertices_with_offset: Array of shape (V, 3) with static_offset applied, or None
    """
    # Validate landmark_type
    if landmark_type not in ["static", "full"]:
        raise ValueError(f"Invalid landmark_type '{landmark_type}'. Must be 'static' or 'full'.")
    
    # Save current directory and change to VHAP root for FLAME model loading
    import os
    original_dir = Path.cwd()
    vhap_root = Path(__file__).parent.parent / "VHAP"
    
    try:
        os.chdir(vhap_root)
        
        # Initialize FLAME model (same as in VHAP scripts)
        flame_model = FlameHead(n_shape, n_expr, add_teeth=add_teeth)
        print(f"Loaded FLAME model with {flame_model.v_template.shape[0]} vertices")
        print(f"Using landmark type: '{landmark_type}'")
    finally:
        # Restore original directory
        os.chdir(original_dir)
    
    # Load landmark embeddings based on type
    lmk_embedding_path = vhap_root / "asset/flame/landmark_embedding_with_eyes.npy"
    
    if not lmk_embedding_path.exists():
        raise FileNotFoundError(f"Landmark embedding not found: {lmk_embedding_path}")
    
    lmk_data = np.load(lmk_embedding_path, allow_pickle=True, encoding='latin1').item()
    
    # Select appropriate landmark embedding
    if landmark_type == "static":
        # Static landmarks: 53 points (no eyes)
        lmk_face_idx = torch.tensor(lmk_data['static_lmk_faces_idx'], dtype=torch.long)
        lmk_b_coords = torch.tensor(lmk_data['static_lmk_bary_coords'], dtype=torch.float32)
    else:  # landmark_type == "full"
        # Full landmarks: 70 points (with eyes) - use from flame_model
        lmk_face_idx = flame_model.full_lmk_faces_idx
        lmk_b_coords = flame_model.full_lmk_bary_coords
    
    # Compute landmarks on canonical FLAME mesh using the EXACT method from VHAP
    with torch.no_grad():
        # Get canonical vertices (neutral expression, zero pose)
        batch_size = 1
        vertices = flame_model.v_template.unsqueeze(0)  # (1, V, 3)
        
        # Prepare face indices and barycentric coordinates with batch dimension
        if landmark_type == "static":
            # Static landmarks need batch dimension added
            lmk_faces_idx_batched = lmk_face_idx.unsqueeze(0).repeat(batch_size, 1)
            lmk_bary_coords_batched = lmk_b_coords.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            # Full landmarks already have batch dimension in first axis
            lmk_faces_idx_batched = lmk_face_idx.repeat(batch_size, 1)
            lmk_bary_coords_batched = lmk_b_coords.repeat(batch_size, 1, 1)
        
        # Use VHAP's vertices2landmarks function for 100% consistency
        flame_landmarks = vertices2landmarks(
            vertices,                    # (B, V, 3)
            flame_model.faces,          # (F, 3)
            lmk_faces_idx_batched,      # (B, N)
            lmk_bary_coords_batched     # (B, N, 3)
        )
        
        # Convert to numpy: (1, N, 3) -> (N, 3)
        flame_landmarks = flame_landmarks[0].numpy()
    
    print(f"Computed {len(flame_landmarks)} FLAME canonical landmarks")
    
    # Optionally compute vertices with static_offset
    vertices_with_offset = None
    if dataset_flame_params is not None and 'static_offset' in dataset_flame_params:
        static_offset = dataset_flame_params['static_offset']
        static_offset_torch = torch.from_numpy(static_offset).float()
        
        # Ensure same number of vertices
        num_verts = flame_model.v_template.shape[0]
        if static_offset_torch.shape[0] != num_verts:
            print(f"  ⚠️  static_offset vertex count mismatch: {static_offset_torch.shape[0]} vs {num_verts}")
            # Pad or trim as needed
            if static_offset_torch.shape[0] < num_verts:
                padding = torch.zeros((num_verts - static_offset_torch.shape[0], 3))
                static_offset_torch = torch.cat([static_offset_torch, padding], dim=0)
                print(f"     Padded static_offset to {num_verts} vertices")
            else:
                static_offset_torch = static_offset_torch[:num_verts]
                print(f"     Trimmed static_offset to {num_verts} vertices")
        
        # Apply static_offset (matching FlameHead.forward logic)
        vertices_with_offset = (flame_model.v_template +static_offset_torch).numpy()
        offset_norm = np.linalg.norm(static_offset_torch.numpy(), axis=1).mean()
        print(f"  ✓ Applied static_offset (mean magnitude: {offset_norm:.6f})")
    
    return flame_model, flame_landmarks, vertices_with_offset


def compute_rigid_alignment(
    source_lmks: np.ndarray,
    target_lmks: np.ndarray,
    enable_scaling: bool = True,
    manual_scale: Optional[float] = None
) -> Tuple[np.ndarray, dict]:
    """
    Compute transformation (scale + rotation + translation) using Procrustes analysis.
    
    This aligns source landmarks to target landmarks, minimizing the sum of squared distances.
    Based on the rigid alignment step in fit_scan_53lmk.py.
    
    Args:
        source_lmks: Source landmarks (N, 3) - e.g., user's mesh landmarks
        target_lmks: Target landmarks (N, 3) - e.g., FLAME canonical landmarks
        enable_scaling: If True, computes similarity transform (with scale).
                       If False, uses rigid transform (rotation + translation only, scale=1.0)
        manual_scale: Manual scale factor override. If provided, uses this instead of 
                     auto-computed scale. Requires enable_scaling=True.
        
    Returns:
        transform_matrix: 4x4 homogeneous transformation matrix
        components: Dictionary with decomposed transformation components
    """
    assert source_lmks.shape == target_lmks.shape, \
        f"Landmark count mismatch: source {source_lmks.shape[0]} vs target {target_lmks.shape[0]}"
    
    # Validate parameters
    if manual_scale is not None and not enable_scaling:
        raise ValueError("manual_scale can only be used when enable_scaling=True")
    
    # Step 1: Center both point clouds
    source_center = source_lmks.mean(axis=0)
    target_center = target_lmks.mean(axis=0)
    source_centered = source_lmks - source_center
    target_centered = target_lmks - target_center
    
    # Step 2: Determine scale factor
    if manual_scale is not None:
        # Use manual scale
        scale = float(manual_scale)
        print(f"Using manual scale factor: {scale:.6f}")
    elif enable_scaling:
        # Compute scale from landmarks
        source_scale = np.sqrt((source_centered ** 2).sum())
        target_scale = np.sqrt((target_centered ** 2).sum())
        scale = target_scale / source_scale
        print(f"Computed scale factor: {scale:.6f}")
    else:
        # No scaling (rigid transform)
        scale = 1.0
        print(f"Scaling disabled (rigid transform, scale=1.0)")
    
    # Step 3: Compute rotation using SVD
    # For rigid transform without scale, we need to normalize the point clouds first
    if enable_scaling or manual_scale is not None:
        # Similarity transform: use original centered points
        H = source_centered.T @ target_centered
    else:
        # Rigid transform: normalize before computing rotation
        source_norm = source_centered / np.sqrt((source_centered ** 2).sum())
        target_norm = target_centered / np.sqrt((target_centered ** 2).sum())
        H = source_norm.T @ target_norm
    
    U, S, Vt = np.linalg.svd(H)
    R_matrix = Vt.T @ U.T
    
    # Ensure proper rotation matrix (det(R) = 1, not -1 for reflection)
    if np.linalg.det(R_matrix) < 0:
        Vt[-1, :] *= -1
        R_matrix = Vt.T @ U.T
    
    print(f"Rotation matrix determinant: {np.linalg.det(R_matrix):.6f}")
    
    # Step 4: Compose into 4x4 transformation matrix
    # T(x) = s * R * x + t
    # where t = target_center - s * R * source_center
    transform = np.eye(4)
    transform[:3, :3] = scale * R_matrix
    translation = target_center - scale * R_matrix @ source_center
    transform[:3, 3] = translation
    
    # Verify alignment quality and compute per-landmark errors
    transformed_source = (scale * R_matrix @ source_lmks.T).T + translation
    per_lmk_errors = np.sqrt(((transformed_source - target_lmks) ** 2).sum(axis=1))
    mean_error = per_lmk_errors.mean()
    max_error = per_lmk_errors.max()
    std_error = per_lmk_errors.std()
    
    print(f"Mean landmark alignment error: {mean_error:.6f}")
    print(f"Max landmark alignment error: {max_error:.6f}")
    print(f"Std landmark alignment error: {std_error:.6f}")
    
    # Decompose transformation for detailed output
    components = {
        'scale': float(scale),
        'rotation_matrix': R_matrix,
        'translation': translation,
        'mean_error': float(mean_error),
        'max_error': float(max_error),
        'std_error': float(std_error),
        'per_landmark_errors': per_lmk_errors
    }
    
    return transform, components


def analyze_landmark_orientation(landmarks: np.ndarray) -> dict:
    """Analyze FLAME landmarks distribution and suggest a default camera eye.

    Strategy:
    - Compute per-axis standard deviation; the axis with the smallest spread is
      assumed to be the depth (face forward/back) axis for canonical FLAME.
    - Use the centroid sign along that axis to choose which side the camera
      should sit on (positive or negative).
    - Set camera distance relative to the bounding-box size so the mesh is
      comfortably framed.

    Returns:
        A dict with keys: 'axis' (one of 'x','y','z'), 'stds', 'means', 'eye'
        where 'eye' is a dict suitable for Plotly camera.eye (x,y,z floats).
    """
    if landmarks is None or len(landmarks) == 0:
        return {'axis': 'z', 'stds': None, 'means': None, 'eye': dict(x=1.5, y=1.5, z=1.2)}

    # Ensure numpy array
    lm = np.asarray(landmarks)
    centroid = lm.mean(axis=0)
    stds = lm.std(axis=0)
    mins = lm.min(axis=0)
    maxs = lm.max(axis=0)
    extents = maxs - mins

    # Axis with smallest spread -> assume depth (face forward/back)
    axis_idx = int(np.argmin(stds))
    axis_char = 'xyz'[axis_idx]

    # Determine sign: use centroid along axis (positive -> camera on positive side)
    sign = 1.0 if centroid[axis_idx] >= 0 else -1.0

    # Frame size: use diagonal of bounding box
    bbox_diag = float(np.linalg.norm(extents))
    if bbox_diag <= 0:
        bbox_diag = 1.0

    # Camera distance factor (tunable) — ensure mesh comfortably fits
    distance = bbox_diag * 2.0

    # Build eye vector (center on other coords)
    eye = dict(x=0.0, y=0.0, z=0.0)
    # Put the camera along chosen axis at computed distance
    coord = sign * distance
    if axis_char == 'x':
        eye['x'] = coord
        # Slight offset for nicer frontal view
        eye['y'] = 0.0
        eye['z'] = distance * 0.25
    elif axis_char == 'y':
        eye['y'] = coord
        eye['x'] = 0.0
        eye['z'] = distance * 0.25
    else:  # 'z'
        eye['z'] = coord
        eye['x'] = 0.0
        eye['y'] = distance * 0.15

    return {
        'axis': axis_char,
        'stds': stds.tolist(),
        'means': centroid.tolist(),
        'eye': eye,
        'bbox_diag': bbox_diag
    }


def create_interactive_html_viewer(
    result: AlignmentResult,
    output_path: Path,
    show_flame: bool = True,
    show_user_mesh: bool = True
) -> None:
    """
    Create interactive HTML 3D viewer using Plotly.
    
    Features:
    - Mouse drag to rotate, scroll to zoom
    - Hover over landmarks to see index and alignment error
    - Toggle layers via legend
    - Error heatmap on landmarks
    - Export PNG button built-in
    
    Args:
        result: AlignmentResult with all mesh and landmark data
        output_path: Path to save HTML file
        show_flame: Whether to include FLAME mesh
        show_user_mesh: Whether to include user mesh
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  ⚠️  Plotly not available, skipping HTML viewer generation")
        print("     Install with: pip install plotly")
        return
    
    fig = go.Figure()
    
    # Layer 1: Target Template (light gray, semi-transparent)
    if show_flame:
        fig.add_trace(go.Mesh3d(
            x=result.target_mesh_vertices[:, 0],
            y=result.target_mesh_vertices[:, 1],
            z=result.target_mesh_vertices[:, 2],
            i=result.target_mesh_faces[:, 0],
            j=result.target_mesh_faces[:, 1],
            k=result.target_mesh_faces[:, 2],
            color='lightgray',
            opacity=0.3,
            name='Target Mesh',
            hoverinfo='skip',
            lighting=dict(ambient=0.8, diffuse=0.5, specular=0.1),
            lightposition=dict(x=100, y=100, z=100)
        ))
    
    # Layer 2: Aligned Source Mesh (green, semi-transparent)
    if show_user_mesh:
        fig.add_trace(go.Mesh3d(
            x=result.source_mesh_aligned_vertices[:, 0],
            y=result.source_mesh_aligned_vertices[:, 1],
            z=result.source_mesh_aligned_vertices[:, 2],
            i=result.source_mesh_faces[:, 0],
            j=result.source_mesh_faces[:, 1],
            k=result.source_mesh_faces[:, 2],
            color='lightgreen',
            opacity=0.6,
            name='Aligned Source Mesh',
            hoverinfo='skip',
            lighting=dict(ambient=0.9, diffuse=0.6, specular=0.2)
        ))
    
    # Layer 3: Target Landmarks (blue with index labels)
    errors_mm = result.per_landmark_errors * 1000  # Convert to mm
    
    fig.add_trace(go.Scatter3d(
        x=result.target_landmarks[:, 0],
        y=result.target_landmarks[:, 1],
        z=result.target_landmarks[:, 2],
        mode='markers+text',
        name='Target Landmarks',
        text=[str(i) for i in range(len(result.target_landmarks))],
        # Place target labels below the points for clearer separation
        textposition='bottom center',
        textfont=dict(size=8, color='darkblue'),
        marker=dict(
            size=8,
            color='blue',
            opacity=0.8,
            line=dict(color='darkblue', width=1)
        ),
        hovertemplate=(
            '<b>Target Landmark %{customdata[0]}</b><br>' +
            'Position: (%{x:.4f}, %{y:.4f}, %{z:.4f})<br>' +
            '<extra></extra>'
        ),
        customdata=np.column_stack([
            np.arange(len(result.target_landmarks)),
            errors_mm
        ])
    ))
    
    # Layer 4: Source Aligned Landmarks (green with index labels)
    fig.add_trace(go.Scatter3d(
        x=result.source_landmarks_aligned[:, 0],
        y=result.source_landmarks_aligned[:, 1],
        z=result.source_landmarks_aligned[:, 2],
        mode='markers+text',
        name='Source Aligned Landmarks',
        text=[str(i) for i in range(len(result.source_landmarks_aligned))],
        # Place source labels above the points to avoid overlap with target labels
        textposition='top center',
        textfont=dict(size=8, color='darkgreen'),
        marker=dict(
            size=8,
            color='green',
            opacity=0.8,
            line=dict(color='darkgreen', width=1)
        ),
        hovertemplate=(
            '<b>Source Landmark %{customdata[0]}</b><br>' +
            'Position: (%{x:.4f}, %{y:.4f}, %{z:.4f})<br>' +
            'Alignment Error: %{customdata[1]:.4f} mm<br>' +
            '<extra></extra>'
        ),
        customdata=np.column_stack([
            np.arange(len(result.source_landmarks_aligned)),
            errors_mm
        ])
    ))
    
    # Layer 5: Error vectors (connecting corresponding landmarks)
    # Create lines connecting target landmark to aligned source landmark
    for i in range(len(result.target_landmarks)):
        fig.add_trace(go.Scatter3d(
            x=[result.target_landmarks[i, 0], result.source_landmarks_aligned[i, 0]],
            y=[result.target_landmarks[i, 1], result.source_landmarks_aligned[i, 1]],
            z=[result.target_landmarks[i, 2], result.source_landmarks_aligned[i, 2]],
            mode='lines',
            line=dict(
                color=errors_mm[i],
                width=2,
                colorscale='RdYlGn_r',  # Red=high error, Green=low error
                cmin=errors_mm.min(),
                cmax=errors_mm.max()
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add colorbar for error visualization
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(
            size=0.1,
            color=[0],
            colorscale='RdYlGn_r',
            cmin=errors_mm.min(),
            cmax=errors_mm.max(),
            colorbar=dict(
                title='Alignment<br>Error (mm)',
                thickness=15,
                len=0.7,
                x=1.02
            ),
            showscale=True
        ),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Determine default camera based on target landmark distribution
    try:
        orientation_info = analyze_landmark_orientation(result.target_landmarks)
        camera_eye = orientation_info.get('eye', dict(x=1.5, y=1.5, z=1.2))
        print(f"  Info: Landmark orientation axis='{orientation_info.get('axis')}', stds={orientation_info.get('stds')}")
    except Exception:
        camera_eye = dict(x=1.5, y=1.5, z=1.2)

    # Layout configuration
    fig.update_layout(
        title=dict(
            text=f'Interactive Landmark Alignment Viewer<br>' +
                 f'<sub>Mean Error: {result.mean_lmk_error*1000:.3f}mm | ' +
                 f'Max Error: {result.max_lmk_error*1000:.3f}mm | ' +
                 f'Landmarks: {len(result.target_landmarks)}</sub>',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            aspectmode='data',
            camera=dict(
                eye=dict(x=float(camera_eye.get('x', 1.5)),
                         y=float(camera_eye.get('y', 1.5)),
                         z=float(camera_eye.get('z', 1.2))),
                center=dict(x=0, y=0, z=0)
            ),
            xaxis=dict(title='X', backgroundcolor='rgb(230, 230, 230)', gridcolor='white'),
            yaxis=dict(title='Y', backgroundcolor='rgb(230, 230, 230)', gridcolor='white'),
            zaxis=dict(title='Z', backgroundcolor='rgb(230, 230, 230)', gridcolor='white')
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=0, t=80),
        hovermode='closest'
    )
    
    # Save as self-contained HTML (offline usable)
    fig.write_html(
        str(output_path),
        include_plotlyjs=True,  # Embed plotly.js for offline use
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'landmark_alignment',
                'height': 1080,
                'width': 1920,
                'scale': 2
            },
            'modeBarButtonsToAdd': ['hoverclosest', 'hovercompare']
        }
    )
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ {output_path} ({file_size_mb:.2f} MB)")
    print(f"    → Open in browser to interact with 3D visualization")
    print(f"    → Hover landmarks to see index and error")
    print(f"    → Use camera button to export high-res PNG")


def save_comprehensive_outputs(
    cfg: AlignmentConfig,
    result: AlignmentResult
) -> None:
    """
    Save all alignment outputs in organized directory structure.
    
    Directory structure:
        output_dir/
        ├── meshes/
        │   ├── flame_canonical.obj
        │   ├── user_mesh_original.obj
        │   └── user_mesh_aligned.obj
        ├── landmarks/
        │   ├── flame_landmarks.npy
        │   ├── user_landmarks_original.npy
        │   ├── user_landmarks_aligned.npy
        │   └── per_landmark_errors.npy
        ├── transform/
        │   ├── transform_matrix.npy
        │   ├── transform_matrix.txt
        │   └── transform_params.json
        └── logs/
            └── alignment_report.txt
    
    Args:
        cfg: Configuration object
        result: AlignmentResult object with all data
    """
    output_dir = cfg.output_dir
    
    # Create subdirectories
    mesh_dir = output_dir / "meshes"
    lmk_dir = output_dir / "landmarks"
    transform_dir = output_dir / "transform"
    
    for d in [mesh_dir, lmk_dir, transform_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving outputs to: {output_dir}")
    
    # 1. Save meshes
    _save_meshes(mesh_dir, result)
    
    # 2. Save landmarks
    _save_landmarks(lmk_dir, result)
    
    # 3. Save transformation
    _save_transformation(transform_dir, result)
    
    # 4. Save alignment report (save at root for visibility)
    _save_alignment_report(output_dir, cfg, result)
    
    print(f"  ✓ Saved alignment report to {output_dir / 'alignment_report.txt'}")


def _save_meshes(mesh_dir: Path, result: AlignmentResult) -> None:
    """Save mesh files in OBJ format without material files."""
    
    def export_obj_without_material(vertices, faces, filepath):
        """Export OBJ file without generating .mtl file."""
        with open(filepath, 'w') as f:
            f.write("# Generated by mesh_align_to_flame.py\n")
            f.write(f"# Vertices: {len(vertices)}\n")
            f.write(f"# Faces: {len(faces)}\n\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        # Ensure no .mtl file exists
        mtl_path = filepath.with_suffix('.mtl')
        if mtl_path.exists():
            mtl_path.unlink()
    
    # Target canonical mesh (pure template, no offset)
    export_obj_without_material(
        result.target_mesh_vertices,
        result.target_mesh_faces,
        mesh_dir / "target_mesh_template.obj"
    )
    
    # Target mesh with static_offset (if available, for FLAME mode)
    if result.target_mesh_with_offset_vertices is not None:
        export_obj_without_material(
            result.target_mesh_with_offset_vertices,
            result.target_mesh_faces,
            mesh_dir / "target_mesh_with_offset.obj"
        )
        print(f"  ✓ {mesh_dir / 'target_mesh_with_offset.obj'} (matches FlameGaussian training)")
    
    # Source mesh original
    export_obj_without_material(
        result.source_mesh_vertices,
        result.source_mesh_faces,
        mesh_dir / "source_mesh_original.obj"
    )
    
    # Source mesh aligned
    export_obj_without_material(
        result.source_mesh_aligned_vertices,
        result.source_mesh_faces,
        mesh_dir / "source_mesh_aligned.obj"
    )
    
    print(f"  ✓ Saved 3 mesh files to {mesh_dir}/")


def _save_landmarks(lmk_dir: Path, result: AlignmentResult) -> None:
    """Save landmark arrays in NPY format."""
    
    np.save(lmk_dir / "target_landmarks.npy", result.target_landmarks)
    
    np.save(lmk_dir / "source_landmarks_original.npy", result.source_landmarks_original)
    
    np.save(lmk_dir / "source_landmarks_aligned.npy", result.source_landmarks_aligned)
    
    np.save(lmk_dir / "per_landmark_errors.npy", result.per_landmark_errors)
    
    print(f"  ✓ Saved 4 landmark files to {lmk_dir}/")


def _save_transformation(transform_dir: Path, result: AlignmentResult) -> None:
    """Save transformation matrix and decomposed parameters."""
    
    # Save comprehensive transform data for train.py in .npz format
    import datetime
    transform_file = transform_dir / "alignment_transform.npz"
    
    # Compute inverse for potential reverse transformation
    inverse_transform = np.linalg.inv(result.transform_matrix)
    
    np.savez(
        transform_file,
        # Primary transform (4x4 matrix)
        transform_matrix=result.transform_matrix,
        inverse_transform=inverse_transform,
        # Decomposed components
        scale=result.scale_factor,
        rotation=result.rotation_matrix,
        translation=result.translation_vector,
        # Metadata
        mean_error=result.mean_lmk_error,
        max_error=result.max_lmk_error,
        std_error=result.std_lmk_error,
        timestamp=str(datetime.datetime.now()),
    )
    print(f"  ✓ {transform_file} (for train.py --mesh-transform-path)")
    
    # Also save as human-readable text
    txt_file = transform_dir / "transform_matrix.txt"
    with open(txt_file, 'w') as f:
        f.write("4×4 Homogeneous Transformation Matrix\n")
        f.write("=" * 50 + "\n\n")
        f.write("T = \n")
        for row in result.transform_matrix:
            f.write("  [ " + "  ".join(f"{v:12.8f}" for v in row) + " ]\n")
        f.write("\n")
        f.write("Applies as: x_aligned = T @ [x, y, z, 1]^T\n")
        f.write(f"\nScale factor: {result.scale_factor:.6f}\n")
        f.write(f"Mean landmark error: {result.mean_lmk_error:.6f}\n")
    print(f"  ✓ {txt_file}")


def _save_alignment_report(log_dir: Path, cfg: AlignmentConfig, result: AlignmentResult) -> None:
    """Generate comprehensive alignment report."""
    
    report_path = log_dir / "alignment_report.txt"
    
    # Error distribution statistics
    errors = result.per_landmark_errors
    n_total = len(errors)
    n_under_005 = np.sum(errors < 0.005)
    n_under_010 = np.sum(errors < 0.010)
    n_under_015 = np.sum(errors < 0.015)
    
    # Convert rotation to Euler angles for readability
    rot = R.from_matrix(result.rotation_matrix)
    euler_deg = rot.as_euler('xyz', degrees=True)
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FLAME Mesh Alignment Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Input files
        f.write("INPUT FILES\n")
        f.write("-" * 80 + "\n")
        f.write(f"User Mesh:      {cfg.mesh_path}\n")
        f.write(f"User Landmarks: {cfg.lmk_path}\n")
        f.write(f"FLAME Model:    FlameHead (n_shape={cfg.n_shape}, n_expr={cfg.n_expr}, ")
        f.write(f"add_teeth={cfg.add_teeth})\n\n")
        
        # Mesh statistics
        f.write("MESH STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Source Mesh (Original):  {len(result.source_mesh_vertices):,} vertices, ")
        f.write(f"{len(result.source_mesh_faces):,} faces\n")
        f.write(f"Target Mesh:             {len(result.target_mesh_vertices):,} vertices, ")
        f.write(f"{len(result.target_mesh_faces):,} faces\n")
        f.write(f"Landmarks:               {len(result.target_landmarks)} points\n\n")
        
        # Alignment results
        f.write("ALIGNMENT RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Scale Factor:          {result.scale_factor:.6f}\n")
        f.write(f"Rotation (Euler XYZ):  [{euler_deg[0]:7.3f}°, {euler_deg[1]:7.3f}°, ")
        f.write(f"{euler_deg[2]:7.3f}°]\n")
        f.write(f"Translation:           [{result.translation_vector[0]:9.6f}, ")
        f.write(f"{result.translation_vector[1]:9.6f}, {result.translation_vector[2]:9.6f}]\n\n")
        
        # Quality metrics
        f.write("QUALITY METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean Landmark Error:   {result.mean_lmk_error:.6f}\n")
        f.write(f"Max Landmark Error:    {result.max_lmk_error:.6f}\n")
        f.write(f"Std Landmark Error:    {result.std_lmk_error:.6f}\n\n")
        
        f.write("Landmark Error Distribution:\n")
        f.write(f"  < 0.005:  {n_under_005:3d} landmarks ({100*n_under_005/n_total:5.1f}%)\n")
        f.write(f"  < 0.010:  {n_under_010:3d} landmarks ({100*n_under_010/n_total:5.1f}%)\n")
        f.write(f"  < 0.015:  {n_under_015:3d} landmarks ({100*n_under_015/n_total:5.1f}%)\n\n")
        
        # Execution time
        f.write("EXECUTION TIME\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total:                 {result.execution_time:.3f} seconds\n")
        if result.timing_breakdown:
            for key, val in result.timing_breakdown.items():
                f.write(f"  - {key:20s} {val:.3f} sec\n")
        f.write("\n")
        
        f.write(f"  Result: Aligned source mesh saved as:\n")
        f.write(f"          {cfg.output_dir}/meshes/source_mesh_aligned.obj\n")
        f.write("=" * 80 + "\n")
    
    print(f"  ✓ {report_path}")



def visualize_results(
    cfg: AlignmentConfig,
    result: AlignmentResult
) -> None:
    """
    Create and save visualization outputs.
    
    Generates visualization files in visualizations/ subdirectory:
    1. landmarks_indexed.png - Static 2D projection with landmark indices (matplotlib)
    2. landmarks_interactive.html - Interactive 3D viewer with mesh+landmarks (plotly)
    
    Note: Trimesh 3D rendering removed (requires GUI environment).
    Use landmarks_interactive.html for interactive 3D mesh visualization.
    
    Args:
        cfg: Configuration object
        result: AlignmentResult with all mesh and landmark data
    """
    if not cfg.save_visualizations:
        print("Skipping visualizations (disabled in config)")
        return
    
    vis_dir = cfg.output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations in: {vis_dir}")
    
    # Note: Trimesh 3D rendering removed (requires GUI environment)
    # Use landmarks_interactive.html for 3D mesh visualization instead
    
    # Scene 4: Indexed landmarks visualization (static) - unified space with correspondences
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot target landmarks (blue)
        ax.scatter(
            result.target_landmarks[:, 0],
            result.target_landmarks[:, 1],
            result.target_landmarks[:, 2],
            c='blue', s=80, alpha=0.7, label='Target landmarks', marker='o', edgecolors='darkblue', linewidths=1.5
        )
        
        # Plot aligned source landmarks (green)
        ax.scatter(
            result.source_landmarks_aligned[:, 0],
            result.source_landmarks_aligned[:, 1],
            result.source_landmarks_aligned[:, 2],
            c='green', s=80, alpha=0.7, label='Aligned source landmarks', marker='^', edgecolors='darkgreen', linewidths=1.5
        )
        
        # Draw correspondence lines between matching landmarks
        errors_mm = result.per_landmark_errors * 1000
        for i in range(len(result.target_landmarks)):
            # Color code lines by error magnitude
            error = errors_mm[i]
            if error < 5:
                color = 'lightgreen'
                alpha = 0.3
            elif error < 10:
                color = 'orange'
                alpha = 0.5
            else:
                color = 'red'
                alpha = 0.7
            
            ax.plot(
                [result.target_landmarks[i, 0], result.source_landmarks_aligned[i, 0]],
                [result.target_landmarks[i, 1], result.source_landmarks_aligned[i, 1]],
                [result.target_landmarks[i, 2], result.source_landmarks_aligned[i, 2]],
                color=color, alpha=alpha, linewidth=1.0
            )
        
        # Add landmark indices (show every 5th to avoid clutter)
        for i in range(0, len(result.target_landmarks), 5):
            # Target landmark index (blue)
            ax.text(
                result.target_landmarks[i, 0],
                result.target_landmarks[i, 1],
                result.target_landmarks[i, 2],
                str(i), fontsize=7, color='darkblue', fontweight='bold'
            )
            # Source landmark index (green, slightly offset)
            ax.text(
                result.source_landmarks_aligned[i, 0],
                result.source_landmarks_aligned[i, 1],
                result.source_landmarks_aligned[i, 2] + 0.02,
                str(i), fontsize=7, color='darkgreen', fontweight='bold'
            )
        
        ax.set_title(
            f'Landmark Alignment Correspondence (n={len(result.target_landmarks)})\n'
            f'Mean Error: {result.mean_lmk_error*1000:.3f}mm | Max: {result.max_lmk_error*1000:.3f}mm',
            fontsize=12, fontweight='bold'
        )
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'landmarks_indexed.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {vis_dir / 'landmarks_indexed.png'}")
        
    except ImportError:
        print(f"  ✗ Could not render landmarks_indexed.png: matplotlib not available")
    except Exception as e:
        print(f"  ✗ Could not render landmarks_indexed.png: {e}")
    
    # Scene 5: Interactive HTML 3D viewer
    try:
        create_interactive_html_viewer(
            result=result,
            output_path=vis_dir / 'landmarks_interactive.html',
            show_flame=True,
            show_user_mesh=True
        )
    except Exception as e:
        print(f"  ✗ Could not generate interactive HTML viewer: {e}")


class FineTuningViewer(Mini3DViewer):
    """
    Interactive viewer for manual fine-tuning of mesh alignment.
    Allows real-time adjustment of scale, rotation, and translation.
    """
    def __init__(self, cfg: AlignmentConfig, result: AlignmentResult):
        # Initialize base viewer configuration
        try:
            import dearpygui.dearpygui as dpg
            from dataclasses import dataclass
            # We don't import Mini3DViewerConfig globally to avoid circular deps risks 
            # if utils aren't perfectly set up, but assumed available here.
        except ImportError:
            raise ImportError("Manual fine-tuning requires 'dearpygui' and 'utils.viewer_utils'.")

        # Define a local config class that includes cam_convention
        # (Mini3DViewer expects cfg.cam_convention, but base Mini3DViewerConfig doesn't have it)
        @dataclass
        class FineTuningConfig(Mini3DViewerConfig):
             cam_convention: str = "opencv"

        # Create a standard config for the viewer
        viewer_cfg = FineTuningConfig(
            W=1600, 
            H=900, 
            radius=1.5, 
            fovy=30,
            cam_convention="opencv"
        )
        
        # Initialize state variables BEFORE super().__init__ because define_gui needs them
        self.align_cfg = cfg
        self.result = result
        
        # Initialize transformation parameters from the auto-alignment result
        self.scale = float(result.scale_factor)
        
        # Convert rotation matrix to Euler angles (degrees) for intuitive sliders
        r = R.from_matrix(result.rotation_matrix)
        self.rot_euler = r.as_euler('xyz', degrees=True).tolist() # [x, y, z]
        
        self.translation = result.translation_vector.tolist() # [x, y, z]

        # History for Undo/Redo
        self.history = []
        self.initial_state = {
            'scale': self.scale,
            'rot': list(self.rot_euler),
            'trans': list(self.translation)
        }
        
        # Call base class init (which calls define_gui)
        super().__init__(viewer_cfg, title="Manual Mesh Alignment Fine-Tuning")
        
        # Initialize renderer
        print("  Initializing NVDiffRenderer for fine-tuning...")
        self.mesh_renderer = NVDiffRenderer(use_opengl=False) # Use CUDA rasterizer
        
        # Prepare Meshes for Rendering
        # 1. Target Reference (Static, Light Gray, Transparent)
        self.target_verts = torch.from_numpy(result.target_mesh_vertices).float().cuda().unsqueeze(0) # [1, V, 3]
        faces = result.target_mesh_faces.numpy() if hasattr(result.target_mesh_faces, 'numpy') else result.target_mesh_faces
        if isinstance(faces, torch.Tensor): faces = faces.cpu().numpy()
        self.target_faces = torch.from_numpy(faces).int().cuda()
        
        self.target_color = torch.tensor([0.7, 0.7, 0.7, 0.5]).cuda() # RGBA
        self.target_face_colors = self.target_color[:3].view(1, 1, 3).expand(1, self.target_faces.shape[0], 3)
        
        # 2. Source Mesh (Dynamic, Light Green, Transparent)
        self.source_verts_orig = torch.from_numpy(result.source_mesh_vertices).float().cuda().unsqueeze(0) # [1, V, 3]
        faces_u = result.source_mesh_faces
        self.source_faces = torch.from_numpy(faces_u).int().cuda()
        
        self.source_color = torch.tensor([0.2, 0.8, 0.2, 0.6]).cuda() # RGBA
        self.source_face_colors = self.source_color[:3].view(1, 1, 3).expand(1, self.source_faces.shape[0], 3)
        
        # Current transformed source vertices (will be updated)
        self.source_verts_transformed = self.source_verts_orig.clone()
        
        # Initial update
        self.update_transform_logical()
        
        print("  Fine-tuning viewer initialized. Please consult the GUI window.")

    def update_transform_logical(self):
        """Update internal transformation matrices and mesh vertices based on current parameters."""
        # 1. Reconstruct Rotation Matrix
        r = R.from_euler('xyz', self.rot_euler, degrees=True)
        self.current_R = r.as_matrix()
        
        # 2. Reconstruct Transform Matrix (T(x) = s * R * x + t)
        self.current_transform = np.eye(4)
        self.current_transform[:3, :3] = self.scale * self.current_R
        self.current_transform[:3, 3] = self.translation
        
        # 3. Apply to User Mesh Vertices (for rendering)
        # v_new = (T @ v_old^T)^T
        T_torch = torch.from_numpy(self.current_transform).float().cuda()
        
        # Homogeneous coordinate transform
        # source_verts_orig: [1, V, 3] -> [V, 3]
        v = self.source_verts_orig[0] 
        v_homo = torch.cat([v, torch.ones_like(v[:, :1])], dim=1) # [V, 4]
        v_new = (T_torch @ v_homo.T).T # [V, 4]
        
        self.source_verts_transformed = v_new[:, :3].unsqueeze(0) # [1, V, 3]

    def define_gui(self):
        super().define_gui()
        
        # Create Control Window
        with dpg.window(label="Fine-Tuning Controls", tag="_control_window", width=350, height=500, pos=(20, 20)):
            dpg.add_text("Manual Alignment Fine-Tuning", color=(0, 255, 255))
            dpg.add_separator()
            
            dpg.add_text("Instructions:")
            dpg.add_text("- Adjust sliders to align Green Mesh (Source) to Gray Mesh (Target)")
            dpg.add_text("- Use Mouse Left/Right to Rotate/Pan Camera")
            dpg.add_text("- Scroll to Zoom")
            dpg.add_separator()
            
            # --- SCALE ---
            dpg.add_text("Scale")
            # speed=0.005 for fine control
            dpg.add_drag_float(
                label="Factor", 
                default_value=self.scale, 
                speed=0.005,
                min_value=0.1, 
                max_value=3.0, 
                callback=self.callback_update_params, 
                tag="_slider_scale"
            )
            dpg.add_separator()
            
            # --- ROTATION ---
            dpg.add_text("Rotation (Euler XYZ)")
            # Using drag_float instead of slider for better sensitivity (speed=0.1 degrees)
            dpg.add_drag_float(label="Pitch (X)", default_value=self.rot_euler[0], speed=0.1, callback=self.callback_update_params, tag="_slider_rot_x")
            dpg.add_drag_float(label="Yaw (Y)",   default_value=self.rot_euler[1], speed=0.1, callback=self.callback_update_params, tag="_slider_rot_y")
            dpg.add_drag_float(label="Roll (Z)",  default_value=self.rot_euler[2], speed=0.1, callback=self.callback_update_params, tag="_slider_rot_z")
            dpg.add_separator()
            
            # --- TRANSLATION ---
            dpg.add_text("Translation (Meters)")
            # Adaptive range based on initial translation + margin
            t_margin = 0.2 # Reduced default margin for finer control visually
            tx, ty, tz = self.translation
            # Use drag_float for infinite range and better sensitivity control
            # speed=0.001 means 1mm per pixel drag
            dpg.add_drag_float(label="X", default_value=tx, speed=0.0005, callback=self.callback_update_params, tag="_slider_trans_x")
            dpg.add_drag_float(label="Y", default_value=ty, speed=0.0005, callback=self.callback_update_params, tag="_slider_trans_y")
            dpg.add_drag_float(label="Z", default_value=tz, speed=0.0005, callback=self.callback_update_params, tag="_slider_trans_z")
            dpg.add_separator()
            
            def callback_save(sender, app_data):
                dpg.stop_dearpygui()
                
            with dpg.group(horizontal=True):
                dpg.add_button(label="UNDO (Ctrl+Z)", callback=self.callback_undo, width=105, height=30)
                dpg.add_button(label="RESET", callback=self.callback_reset, width=105, height=30)
                dpg.add_button(label="SAVE", callback=callback_save, width=105, height=30)

        # Register keyboard handlers
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Z, callback=self.callback_key_press)
            dpg.add_key_press_handler(dpg.mvKey_Left, callback=self.callback_key_press)
            dpg.add_key_press_handler(dpg.mvKey_Right, callback=self.callback_key_press)
            dpg.add_key_press_handler(dpg.mvKey_Up, callback=self.callback_key_press)
            dpg.add_key_press_handler(dpg.mvKey_Down, callback=self.callback_key_press)
            # Shift modifiers are checked inside callback

        # Register mouse wheel handler for UI controls
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self.callback_mouse_wheel_ui)

    def callback_mouse_wheel_ui(self, sender, app_data):
        delta = app_data # typically 1.0 or -1.0
        
        # Map tags to (attribute_name, index, step)
        controls = {
            "_slider_scale":   ('scale', None, 0.005),
            "_slider_rot_x":   ('rot_euler', 0, 0.1),
            "_slider_rot_y":   ('rot_euler', 1, 0.1),
            "_slider_rot_z":   ('rot_euler', 2, 0.1),
            "_slider_trans_x": ('translation', 0, 0.0005),
            "_slider_trans_y": ('translation', 1, 0.0005),
            "_slider_trans_z": ('translation', 2, 0.0005),
        }

        for tag, (attr, idx, step) in controls.items():
            if dpg.is_item_hovered(tag):
                # Apply change
                change = step * delta
                
                # Push history only on first significant change? 
                # For wheel, distinct steps are clear, so let's try to be smart or just update.
                # Ideally, we should push history.
                # But wheel can fire rapidly. Let's just update for now to keep it smooth.
                
                if attr == 'scale':
                    self.scale += change
                    self.scale = max(0.1, min(3.0, self.scale))
                    dpg.set_value(tag, self.scale)
                elif attr == 'rot_euler':
                    self.rot_euler[idx] += change
                    dpg.set_value(tag, self.rot_euler[idx])
                elif attr == 'translation':
                    self.translation[idx] += change
                    dpg.set_value(tag, self.translation[idx])
                
                self.update_transform_logical()
                self.need_update = True
                return # Handled, stop checking

    def callback_key_press(self, sender, app_data):
        # Handle Ctrl+Z for Undo
        if dpg.is_key_down(dpg.mvKey_Control) and app_data == dpg.mvKey_Z:
            self.callback_undo(None, None)
            return

        # Handle Arrow Keys for Fine Translation
        # Step size: 0.5mm (0.0005m) normally, 5mm (0.005m) with Shift
        step = 0.005 if dpg.is_key_down(dpg.mvKey_Shift) else 0.0005
        
        changed = False
        if app_data == dpg.mvKey_Left:
            self.push_history() # Save state before move
            self.translation[0] -= step # X-
            changed = True
        elif app_data == dpg.mvKey_Right:
            self.push_history()
            self.translation[0] += step # X+
            changed = True
        elif app_data == dpg.mvKey_Up:
            self.push_history()
            self.translation[1] += step # Y+
            changed = True
        elif app_data == dpg.mvKey_Down:
            self.push_history()
            self.translation[1] -= step # Y-
            changed = True
            
        if changed:
            # Update GUI sliders to match internal state
            dpg.set_value("_slider_trans_x", self.translation[0])
            dpg.set_value("_slider_trans_y", self.translation[1])
            self.update_transform_logical()
            self.need_update = True

    def push_history(self):
        """Save current state to history stack."""
        state = {
            'scale': self.scale,
            'rot': list(self.rot_euler),
            'trans': list(self.translation)
        }
        self.history.append(state)
        # Limit history size
        if len(self.history) > 20: 
            self.history.pop(0)

    def callback_undo(self, sender, app_data):
        if not self.history:
            print("  [UI] Nothing to undo.")
            return
            
        prev_state = self.history.pop()
        self.scale = prev_state['scale']
        self.rot_euler = list(prev_state['rot'])
        self.translation = list(prev_state['trans'])
        
        # Sync GUI
        dpg.set_value("_slider_scale", self.scale)
        dpg.set_value("_slider_rot_x", self.rot_euler[0])
        dpg.set_value("_slider_rot_y", self.rot_euler[1])
        dpg.set_value("_slider_rot_z", self.rot_euler[2])
        dpg.set_value("_slider_trans_x", self.translation[0])
        dpg.set_value("_slider_trans_y", self.translation[1])
        dpg.set_value("_slider_trans_z", self.translation[2])
        
        self.update_transform_logical()
        self.need_update = True
        print("  [UI] Undo performed.")

    def callback_reset(self, sender, app_data):
        self.push_history() # Save current before reset
        
        self.scale = self.initial_state['scale']
        self.rot_euler = list(self.initial_state['rot'])
        self.translation = list(self.initial_state['trans'])
        
        dpg.set_value("_slider_scale", self.scale)
        dpg.set_value("_slider_rot_x", self.rot_euler[0])
        dpg.set_value("_slider_rot_y", self.rot_euler[1])
        dpg.set_value("_slider_rot_z", self.rot_euler[2])
        dpg.set_value("_slider_trans_x", self.translation[0])
        dpg.set_value("_slider_trans_y", self.translation[1])
        dpg.set_value("_slider_trans_z", self.translation[2])
        
        self.update_transform_logical()
        self.need_update = True
        print("  [UI] Reset to initial state.")

    def callback_update_params_with_history(self, sender, app_data):
        # We only push history on 'activation' (start of drag) or check if value changed signficantly?
        # DPG callbacks fire continuously during drag.
        # Ideally we want history on "mouse release" or "start edit".
        # But for simplicity, we might not push history *during* drag via this callback 
        # unless we can detect 'start'. 
        # A simple hack: push history if the value is 'stable' or just rely on manual 'Snapshot' button?
        # Better: let's not spam history on drag. 
        # We will add a helper 'callback_start_edit' if possible, but DPG is limited.
        # Alternative: The user explicitly uses undo for 'steps'. 
        # If we spam history, undo becomes tedious (undoing 0.001 changes).
        # Let's try to only push when the *sender* changes (focus logic).
        # OR: Just simple "Push History" button? No, that's bad UX.
        # Let's implement a 'lazy' history: save state ONLY when starting interaction? 
        # Hard with standard callbacks.
        # Compromise: We won't auto-push on drag. We push on Key press and Button clicks.
        # For sliders, we can't easily undo 'half a drag'. 
        # We will add checks: if time since last history push > 1 sec? No.
        # Let's just update values here.
        self.callback_update_params(sender, app_data)

    def callback_update_params(self, sender, app_data):
        self.scale = dpg.get_value("_slider_scale")
        self.rot_euler[0] = dpg.get_value("_slider_rot_x")
        self.rot_euler[1] = dpg.get_value("_slider_rot_y")
        self.rot_euler[2] = dpg.get_value("_slider_rot_z")
        self.translation[0] = dpg.get_value("_slider_trans_x")
        self.translation[1] = dpg.get_value("_slider_trans_y")
        self.translation[2] = dpg.get_value("_slider_trans_z")
        
        self.update_transform_logical()
        self.need_update = True

    def prepare_camera(self):
        """Adapter to convert OrbitCamera to what Gaussian Splatting / NVDiffRenderer expects."""
        @dataclass
        class Cam:
            FoVx = float(np.radians(self.cam.fovx))
            FoVy = float(np.radians(self.cam.fovy))
            image_height = self.cam.image_height
            image_width = self.cam.image_width
            world_view_transform = torch.tensor(self.cam.world_view_transform).float().cuda().T
            full_proj_transform = torch.tensor(self.cam.full_proj_transform).float().cuda().T
            camera_center = torch.tensor(self.cam.pose[:3, 3]).cuda()
        return Cam

    def run(self):
        """Main Render Loop"""
        print("  Starting GUI render loop...")
        while dpg.is_dearpygui_running():
            
            if self.need_update:
                cam = self.prepare_camera()
                
                # Render Target (Static Reference)
                out_target = self.mesh_renderer.render_from_camera(
                    self.target_verts, 
                    self.target_faces, 
                    cam, 
                    face_colors=self.target_face_colors
                )
                rgba_target = out_target['rgba'].squeeze(0) # [H, W, 4]
                
                # Render Source Mesh (Dynamic)
                out_source = self.mesh_renderer.render_from_camera(
                    self.source_verts_transformed,
                    self.source_faces,
                    cam,
                    face_colors=self.source_face_colors
                )
                rgba_source = out_source['rgba'].squeeze(0) # [H, W, 4]
                
                # Composite
                # Simple alpha compositing: Source over Target over White Background
                bg_color = torch.tensor([1.0, 1.0, 1.0, 1.0]).cuda().view(1, 1, 4)
                
                # Mix Source over Target
                # out = source * source_a + target * target_a * (1 - source_a)
                # But we have pre-multiplied alpha or not? setup in render_from_camera seems to return [RGB, A]
                
                # Let's do simple manual composition
                # 1. Background
                # USE cam snapshot dimensions to ensure match with renderer output
                canvas = bg_color.expand(cam.image_height, cam.image_width, 4).clone()
                
                if canvas.shape[0] != rgba_target.shape[0] or canvas.shape[1] != rgba_target.shape[1]:
                    print(f"Warning: Resize detected. Canvas: {canvas.shape}, Target: {rgba_target.shape}")
                    # Skip frame if mismatch (though using cam.* should prevent this)
                    canvas = bg_color.expand(rgba_target.shape[0], rgba_target.shape[1], 4).clone()

                # 2. Add Target
                target_alpha = rgba_target[..., 3:] * self.target_color[3]
                canvas = rgba_target * target_alpha + canvas * (1 - target_alpha)
                
                # 3. Add Source
                source_alpha = rgba_source[..., 3:] * self.source_color[3]
                canvas = rgba_source * source_alpha + canvas * (1 - source_alpha)
                
                # Convert to numpy for display
                # DPG requires contiguous float32 buffer
                buffer = canvas[..., :3].cpu().numpy()
                self.render_buffer = np.ascontiguousarray(buffer, dtype=np.float32)
                
                # Safety check for dimensions before passing to DPG
                # DPG texture was created with (self.W, self.H)
                # Our buffer is (H, W, 3). Total elements must match.
                expected_size = self.W * self.H * 3
                if self.render_buffer.size != expected_size:
                    print(f"Warning: Buffer size mismatch. W={self.W} H={self.H}, Buffer={self.render_buffer.shape}")
                    # Skip update to avoid crash
                else: 
                    # Update DPG texture
                    dpg.set_value("_texture", self.render_buffer)
                    
                self.need_update = False
            
            dpg.render_dearpygui_frame()

    def get_final_transform(self):
        """Return the final transformation matrix and components."""
        return self.current_transform, {
            'scale': self.scale,
            'rotation_matrix': self.current_R,
            'translation': np.array(self.translation)
        }


def main(cfg: AlignmentConfig) -> None:
    """
    Main execution function with comprehensive output generation.
    
    Args:
        cfg: Configuration object with all parameters
    """
    start_time = time()
    timing = {}
    
    print("="*80)
    print("Mesh Alignment Tool - Source → Target")
    print("="*80)
    print(f"Source mesh:      {cfg.source_mesh}")
    print(f"Source landmarks: {cfg.source_lmk}")
    print(f"Target mode:      {cfg.target_mode}")
    if cfg.target_mode == "flame":
        print(f"FLAME reference:  {cfg.target_flame_ref}")
    else:
        print(f"Target mesh:      {cfg.target_mesh}")
        print(f"Target landmarks: {cfg.target_lmk}")
    print(f"Output dir:       {cfg.output_dir}")
    print("="*80 + "\n")
    
    # ========== STEP 1: Load Target (FLAME or Custom) ==========
    print(f"\n[1/6] Loading target mesh and landmarks...")
    t0 = time()
    
    if cfg.target_mode == "flame":
        # MODE 1: FLAME Reference
        print(f"  Using FLAME reference: {cfg.target_flame_ref}")
        
        # Load FlameHead model (for topology and/or mesh generation)
        flame_model, _, _ = load_flame_model_and_landmarks(
            cfg.n_shape, cfg.n_expr, cfg.add_teeth, cfg.landmark_type, None
        )
        
        # Load reference mesh and compute landmarks
        target_vertices, target_landmarks = load_flame_reference_with_landmarks(
            cfg.target_flame_ref, flame_model, cfg.landmark_type
        )
        target_faces = flame_model.faces.numpy() if isinstance(flame_model.faces, torch.Tensor) else flame_model.faces
        
        print(f"  ✓ Loaded FLAME target: {len(target_vertices):,} vertices, {len(target_faces):,} faces")
        print(f"  ✓ Computed {len(target_landmarks)} FLAME landmarks")
        
    else:
        # MODE 2: Custom Mesh
        print(f"  Using custom target: {cfg.target_mesh}")
        
        # Load target mesh
        target_mesh = trimesh.load(cfg.target_mesh, process=False)
        target_vertices = target_mesh.vertices
        target_faces = target_mesh.faces
        
        # Load target landmarks
        target_landmarks = load_user_landmarks(cfg.target_lmk, expected_count=None)
        
        print(f"  ✓ Loaded custom target: {len(target_vertices):,} vertices, {len(target_faces):,} faces")
        print(f"  ✓ Loaded {len(target_landmarks)} target landmarks")
    
    timing['Target loading'] = time() - t0
    
    #  ========== STEP 2: Load Source Mesh ==========
    print(f"\n[2/6] Loading source mesh from {cfg.source_mesh}...")
    t0 = time()
    source_mesh = trimesh.load(cfg.source_mesh, process=False)
    print(f"  ✓ Loaded source mesh: {len(source_mesh.vertices):,} vertices, {len(source_mesh.faces):,} faces")
    timing['Source mesh loading'] = time() - t0
    
    # ========== STEP 3: Load Source Landmarks with Adaptive Conversion ==========
    print(f"\n[3/6] Loading source landmarks from {cfg.source_lmk}...")
    t0 = time()
    
    # Load landmarks without validation first
    source_landmarks = load_user_landmarks(cfg.source_lmk, expected_count=None)
    source_count = len(source_landmarks)
    target_count = len(target_landmarks)
    
    print(f"  Detected source: {source_count} landmarks, target: {target_count} landmarks")
    print(f"  Alignment strategy: {cfg.landmark_type}")
    
    # ========== Alignment Strategy-Based Conversion ==========
    if cfg.landmark_type == "static":
        # STATIC MODE: Flexible - convert both to 51 facial points (remove contour)
        print(f"\n  📐 Static alignment mode: converting both to 51 facial points...")
        
        # Convert source to 51 points
        if source_count == 68:
            print(f"    Source: 68 → 51 (removing contour [0:17])")
            source_landmarks = source_landmarks[LANDMARK_FORMAT_68_TO_51_INDICES]
        elif source_count == 70:
            print(f"    Source: 70 → 51 (extracting facial subset [17:68])")
            source_landmarks = source_landmarks[LANDMARK_FORMAT_68_TO_51_INDICES]
        elif source_count == 53:
            print(f"    Source: 53 → 51 (removing 2 eyeball centers)")
            source_landmarks = source_landmarks[:51]  # First 51 are facial points
        elif source_count == 51:
            print(f"    Source: 51 ✓ (already facial points)")
        else:
            raise ValueError(
                f"❌ Unsupported source landmark count for static mode: {source_count}\n"
                f"  Supported: 51, 53, 68, 70\n"
                f"  File: {cfg.source_lmk}"
            )
        
        # Convert target to 51 points
        if target_count == 68:
            print(f"    Target: 68 → 51 (removing contour [0:17])")
            target_landmarks = target_landmarks[LANDMARK_FORMAT_68_TO_51_INDICES]
        elif target_count == 70:
            print(f"    Target: 70 → 51 (extracting facial subset [17:68])")
            target_landmarks = target_landmarks[LANDMARK_FORMAT_68_TO_51_INDICES]
        elif target_count == 53:
            print(f"    Target: 53 → 51 (removing 2 eyeball centers)")
            target_landmarks = target_landmarks[:51]
        elif target_count == 51:
            print(f"    Target: 51 ✓ (already facial points)")
        else:
            raise ValueError(
                f"❌ Unsupported target landmark count for static mode: {target_count}\n"
                f"  Supported: 51, 53, 68, 70\n"
                f"  Note: Target landmarks were loaded from: "
                f"{cfg.target_lmk if cfg.target_mode == 'custom' else 'FLAME model'}"
            )
        
        print(f"  ✅ Final alignment: {len(source_landmarks)} source ↔ {len(target_landmarks)} target points")
        
    elif cfg.landmark_type == "full":
        # FULL MODE: Strict - require 68 points from both
        print(f"\n  🎯 Full alignment mode: requires 68 points from both source and target...")
        
        if source_count != 68:
            raise ValueError(
                f"❌ Full mode requires 68-point source landmarks\n"
                f"  Got: {source_count} points\n"
                f"  File: {cfg.source_lmk}\n"
                f"  💡 Suggestion: Use --landmark-type static for flexible alignment"
            )
        
        if target_count != 68:
            raise ValueError(
                f"❌ Full mode requires 68-point target landmarks\n"
                f"  Got: {target_count} points\n"
                f"  Target: {cfg.target_lmk if cfg.target_mode == 'custom' else 'FLAME (use 70-point format)'}\n"
                f"  💡 Suggestion: Use --landmark-type static for flexible alignment"
            )
        
        print(f"  ✅ Perfect match: 68 source ↔ 68 target points")
    
    else:
        raise ValueError(f"Invalid landmark_type: {cfg.landmark_type} (must be 'static' or 'full')")
    
    timing['Landmark loading'] = time() - t0
    
    # ========== STEP 4: Compute Rigid Alignment ==========
    print(f"\n[4/6] Computing rigid alignment (Procrustes analysis)...")
    t0 = time()
    transform_matrix, components = compute_rigid_alignment(
        source_landmarks, 
        target_landmarks,
        enable_scaling=cfg.enable_scaling,
        manual_scale=cfg.manual_scale
    )
    timing['Alignment computation'] = time() - t0
    
    # Apply transformation
    aligned_vertices = trimesh.transformations.transform_points(
        source_mesh.vertices, transform_matrix
    )
    aligned_landmarks = trimesh.transformations.transform_points(
        source_landmarks, transform_matrix
    )
    
    # ========== STEP 5: Package Results ==========
    print(f"\n[5/6] Packaging alignment results...")
    result = AlignmentResult(
        # Input data
        target_mesh_vertices=target_vertices,
        target_mesh_faces=target_faces,
        target_landmarks=target_landmarks,
        source_mesh_vertices=source_mesh.vertices.copy(),
        source_mesh_faces=source_mesh.faces.copy(),
        source_landmarks_original=source_landmarks,
        # Transformation
        transform_matrix=transform_matrix,
        scale_factor=components['scale'],
        rotation_matrix=components['rotation_matrix'],
        translation_vector=components['translation'],
        # Transformed outputs
        source_mesh_aligned_vertices=aligned_vertices,
        source_landmarks_aligned=aligned_landmarks,
        # Quality metrics
        mean_lmk_error=components['mean_error'],
        max_lmk_error=components['max_error'],
        std_lmk_error=components['std_error'],
        per_landmark_errors=components['per_landmark_errors'],
        # Target mesh metadata (optional, for FLAME mode)
        target_mesh_with_offset_vertices=None,
        # Timing
        execution_time=0.0,  # Will be updated
        timing_breakdown=timing
    )

    # [RESUME FROM EXISTING ALIGNMENT]
    if cfg.load_alignment_path is not None:
        if cfg.load_alignment_path.exists():
            print(f"\nLoading existing alignment from: {cfg.load_alignment_path}")
            try:
                data = np.load(cfg.load_alignment_path)
                
                # Check required keys
                required = ['scale', 'rotation', 'translation', 'transform_matrix']
                if all(k in data for k in required):
                    # Override result metrics
                    result.scale_factor = float(data['scale'])
                    result.rotation_matrix = data['rotation']
                    result.translation_vector = data['translation']
                    result.transform_matrix = data['transform_matrix']
                    
                    # Apply transform to mesh and landmarks
                    result.source_mesh_aligned_vertices = trimesh.transformations.transform_points(
                        result.source_mesh_vertices, result.transform_matrix
                    )
                    result.source_landmarks_aligned = trimesh.transformations.transform_points(
                        result.source_landmarks_original, result.transform_matrix
                    )
                    
                    # Re-calc errors
                    per_lmk_errors = np.sqrt(((result.source_landmarks_aligned - result.target_landmarks) ** 2).sum(axis=1))
                    result.per_landmark_errors = per_lmk_errors
                    result.mean_lmk_error = per_lmk_errors.mean()
                    result.max_lmk_error = per_lmk_errors.max()
                    result.std_lmk_error = per_lmk_errors.std()
                    
                    print(f"  ✓ Loaded successfully. Mean Error: {result.mean_lmk_error:.6f}")
                else:
                    print(f"  ❌ Invalid .npz file. Missing keys: {[k for k in required if k not in data]}")
                    print("  Proceeding with auto-alignment result instead.")
            except Exception as e:
                print(f"  ❌ Failed to load alignment: {e}")
        else:
            print(f"  ❌ File not found: {cfg.load_alignment_path}")
    
    # [MANUAL FINE-TUNING STEP]
    if cfg.manual_fine_tune:
        print("\n" + "="*80)
        print("MANUAL FINE-TUNING ENABLED")
        print("="*80)
        print("  Launching interactive GUI for fine-tuning...")
        print("  Close the GUI window to save changes and continue.")
        
        try:
            viewer = FineTuningViewer(cfg, result)
            viewer.run() # Blocking call
            
            # Retrieve updated transformation
            new_transform, new_components = viewer.get_final_transform()
            
            print("\n  ✓ Fine-tuning complete. Updating results...")
            
            # Update Result Object
            result.transform_matrix = new_transform
            result.scale_factor = new_components['scale']
            result.rotation_matrix = new_components['rotation_matrix']
            result.translation_vector = new_components['translation']
            
            # Re-apply transformation to vertices (Critical for correct output)
            result.user_mesh_aligned_vertices = trimesh.transformations.transform_points(
                result.user_mesh_vertices, new_transform
            )
            result.user_landmarks_aligned = trimesh.transformations.transform_points(
                result.user_landmarks_original, new_transform
            )
            
            # Re-compute errors
            per_lmk_errors = np.sqrt(((result.user_landmarks_aligned - result.flame_landmarks) ** 2).sum(axis=1))
            result.per_landmark_errors = per_lmk_errors
            result.mean_lmk_error = per_lmk_errors.mean()
            result.max_lmk_error = per_lmk_errors.max()
            result.std_lmk_error = per_lmk_errors.std()
            
            print(f"  New Mean Error: {result.mean_lmk_error:.6f} (may vary from auto-alignment)")
            
        except ImportError as e:
            print(f"  ❌ Failed to launch fine-tuning GUI: {e}")
            print("  Make sure 'dearpygui' is installed and you are running in a GUI environment.")
        except Exception as e:
            print(f"  ❌ Error during fine-tuning: {e}")
            import traceback
            traceback.print_exc()

    # Step 6: Save comprehensive outputs

    t0 = time()
    save_comprehensive_outputs(cfg, result)
    timing['Output generation'] = time() - t0
    
    # Generate visualizations (optional)
    if cfg.save_visualizations:
        t0 = time()
        visualize_results(cfg, result)
        timing['Visualization'] = time() - t0
    
    # Update total execution time and re-save report
    total_time = time() - start_time
    result.execution_time = total_time
    result.timing_breakdown = timing
    
    # Re-generate report with complete timing
    _save_alignment_report(cfg.output_dir, cfg, result)
    
    # Final summary
    print("\n" + "="*60)
    print("✓ Alignment Complete!")
    print("="*60)
    print(f"  Results saved to: {cfg.output_dir.absolute()}")
    print(f"  Report:           {cfg.output_dir}/alignment_report.txt")
    print(f"  Visualizations:   {cfg.output_dir}/visualizations/")
    print(f"  Aligned Mesh:     {cfg.output_dir}/meshes/user_mesh_aligned.obj")
    print("="*60)


if __name__ == "__main__":
    cfg = tyro.cli(AlignmentConfig)
    main(cfg)
